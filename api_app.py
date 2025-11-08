#!/usr/bin/env python3
"""
api_app.py

FastAPI recommender using FAISS + sentence-transformers with optional Groq reranking.

Usage:
  uvicorn api_app:app --reload --port 8000

Requirements:
  pip install fastapi uvicorn python-dotenv pandas numpy faiss-cpu sentence-transformers requests groq
  # If groq not needed, skip installing it.
"""
import os
import time
import json
import logging
import re
import math
from collections import defaultdict
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# optional Groq
try:
    from groq import Groq
    GROQ_PKG = True
except Exception:
    GROQ_PKG = False

# ---------------- Config & logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
DATA_DIR = os.getenv("DATA_DIR", "./data")
FAISS_PATH = os.path.join(DATA_DIR, "catalog.faiss")
CATALOG_CSV = os.path.join(DATA_DIR, "catalog_indexed.csv")
INDEX_META = os.path.join(DATA_DIR, "index_meta.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # change if needed
USE_GROQ_EXPLAIN = os.getenv("GROQ_EXPLAIN", "true").lower() in ("1","true","yes")

# Reranker weights (fallback)
W_SIM = float(os.getenv("W_SIM", 0.6))
W_OVERLAP = float(os.getenv("W_OVERLAP", 0.25))
W_TYPE = float(os.getenv("W_TYPE", 0.15))

# ---------------- App init ----------------
app = FastAPI(title="SHL GenAI Recommender (Groq + FAISS)")

# CORS (open for local dev; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# root route (friendly)
@app.get("/")
def root():
    return {"message": "API running. Use /health, /recommend, /batch_export or open /docs for interactive UI."}

# ---------------- Load model, data, index ----------------
logging.info("Loading embedding model: %s", EMBED_MODEL)
model = SentenceTransformer(EMBED_MODEL)

if not os.path.exists(CATALOG_CSV) or not os.path.exists(FAISS_PATH):
    raise RuntimeError("catalog_indexed.csv or catalog.faiss not found in data dir. Run build_index.py first.")

df = pd.read_csv(CATALOG_CSV).fillna("")
logging.info("Loaded catalog rows: %d", len(df))

index = faiss.read_index(FAISS_PATH)
logging.info("Loaded FAISS index. ntotal=%d", int(index.ntotal))

# Try to detect use_cosine from index_meta
USE_COSINE = False
if os.path.exists(INDEX_META):
    try:
        with open(INDEX_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
            USE_COSINE = bool(meta.get("use_cosine", False))
    except Exception:
        USE_COSINE = False
logging.info("Index meta: use_cosine=%s", USE_COSINE)

# Groq client setup
groq_client = None
if GROQ_API_KEY and GROQ_PKG:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logging.info("Groq client initialized.")
    except Exception as e:
        groq_client = None
        logging.warning("Groq client initialization failed: %s", e)
else:
    if GROQ_API_KEY and not GROQ_PKG:
        logging.warning("GROQ_API_KEY set but 'groq' package not installed.")
    else:
        logging.info("Groq not configured; using fallback reranker only.")

# ---------------- Request/response models ----------------
class RecommendRequest(BaseModel):
    query: Optional[str] = None
    url: Optional[str] = None
    k: Optional[int] = 10

class Assessment(BaseModel):
    assessment_name: str
    url: str
    score: float
    rationale: Optional[str] = None

class RecommendResponse(BaseModel):
    query: str
    predictions: List[Assessment]

class BatchExportRequest(BaseModel):
    queries: List[str]
    k: Optional[int] = 10

# ---------------- Helpers ----------------
def fetch_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        text = r.text
        # quick HTML->text
        text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", text)
        text = re.sub(r"(?s)<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

def normalize_scores(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-12:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def top_n_candidates(query: str, top_n: int = 50):
    q_emb = model.encode([query], convert_to_numpy=True)
    if USE_COSINE:
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(q_emb.astype("float32"), top_n)
    candidates = []
    for dist, idx in zip(D[0].tolist(), I[0].tolist()):
        row = df.iloc[int(idx)]
        candidates.append({
            "idx": int(idx),
            "assessment_name": row.get("assessment_name",""),
            "url": row.get("url",""),
            "description": row.get("description",""),
            "test_type": row.get("test_type",""),
            "raw_faiss_score": float(dist)
        })
    return candidates

# ---------------- Fallback reranker ----------------
def fallback_rerank(query: str, candidates: list):
    if not candidates:
        return []
    texts = [ (c["assessment_name"] + " " + c["description"] + " " + c.get("url","")) for c in candidates ]
    emb = model.encode(texts, convert_to_numpy=True)
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    # cosine sim or L2-based proxy
    def cos_sim(a,b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a,b) / denom) if denom > 1e-12 else 0.0
    sims = np.array([cos_sim(q_emb,e) for e in emb], dtype="float32")
    sims_n = normalize_scores(sims)

    q_tokens = set([t.lower() for t in re.findall(r"\w+", query) if len(t)>2])
    overlaps = []
    for c in candidates:
        text_tokens = set([t.lower() for t in re.findall(r"\w+", (c.get("assessment_name","") + " " + c.get("description",""))) if len(t)>2])
        overlaps.append(len(q_tokens & text_tokens)/max(1,len(q_tokens)))
    overlaps_n = normalize_scores(np.array(overlaps, dtype="float32"))

    q_lower = query.lower()
    wants_P = bool(re.search(r"\b(personality|behaviour|behavioral|situational|soft skill|competenc)\b", q_lower))
    wants_K = bool(re.search(r"\b(knowledge|aptitude|verbal|numerical|logical|cognitive|technical|coding)\b", q_lower))
    type_scores = []
    for c in candidates:
        t = (c.get("test_type") or "").upper()
        s = 0.0
        if wants_P and "P" in t: s += 1.0
        if wants_K and "K" in t: s += 1.0
        type_scores.append(s)
    type_n = normalize_scores(np.array(type_scores, dtype="float32"))

    combined = W_SIM * sims_n + W_OVERLAP * overlaps_n + W_TYPE * type_n
    for i,c in enumerate(candidates):
        c["score"] = float(combined[i])
        c["sim_component"] = float(sims_n[i])
        c["overlap_component"] = float(overlaps_n[i])
        c["type_component"] = float(type_n[i])
    return sorted(candidates, key=lambda x: x["score"], reverse=True)

# ---------------- Groq reranker ----------------
def groq_rerank(query: str, candidates: list, model_name: str = GROQ_MODEL, include_rationales: bool = USE_GROQ_EXPLAIN):
    if groq_client is None:
        raise RuntimeError("Groq client not initialized or GROQ_API_KEY missing.")
    if not candidates:
        return []

    max_candidates = min(15, len(candidates))
    items = []
    for i, c in enumerate(candidates[:max_candidates], start=1):
        short = (c["assessment_name"] + " — " + (c["description"][:250] or "")).strip()
        items.append(f"{i}. {short}\nURL: {c['url']}\n")

    prompt = (
        "You are an expert matching job descriptions to assessment products. "
        "Given the job description and a numbered list of candidate assessments, rate each candidate from 0 to 100 for relevance. "
        "Return ONLY valid JSON array like: [{\"index\":1, \"score\":87, \"rationale\":\"one-line reason\"}, ...]. "
        "If you cannot provide a rationale, set it to an empty string.\n\n"
        f"Job description:\n'''{query}'''\n\nCandidates:\n" + "\n".join(items)
    )

    start = time.time()
    resp = groq_client.chat.completions.create(
        messages=[{"role":"user","content":prompt}],
        model=model_name,
        temperature=0.0,
        max_tokens=800
    )
    elapsed = time.time() - start
    logging.info("Groq reranker call finished in %.2f s", elapsed)

    text = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(text)
    except Exception:
        # best-effort parsing: find lines with index and score
        parsed = []
        for line in text.splitlines():
            m = re.search(r"(\d+)[^0-9]+(\d{1,3}(?:\.\d+)?)", line)
            if m:
                parsed.append({"index": int(m.group(1)), "score": float(m.group(2)), "rationale": ""})
    # apply scores
    for p in parsed:
        idx = p.get("index")
        sc = p.get("score", 0)
        rat = p.get("rationale", "") if isinstance(p.get("rationale",""), str) else ""
        if idx and 1 <= idx <= max_candidates:
            candidates[idx-1]["score"] = float(sc) / 100.0
            candidates[idx-1]["rationale"] = rat
    # ensure other candidates have score field
    for i,c in enumerate(candidates):
        if "score" not in c:
            c["score"] = 0.0
            c["rationale"] = ""
    return sorted(candidates, key=lambda x: x.get("score",0.0), reverse=True)

# ---------------- Balancing ----------------
def enforce_balancing(candidates_sorted: list, k: int):
    buckets = defaultdict(list)
    for c in candidates_sorted:
        t = (c.get("test_type") or "").upper()
        if "K" in t and "P" in t:
            buckets["K+P"].append(c)
        elif "K" in t:
            buckets["K"].append(c)
        elif "P" in t:
            buckets["P"].append(c)
        else:
            buckets["OTHER"].append(c)
    selection = []
    # greedy: try to include from each bucket in order
    for bucket_name in ["K", "P", "K+P", "OTHER"]:
        while len(selection) < k and buckets[bucket_name]:
            selection.append(buckets[bucket_name].pop(0))
    if len(selection) < k:
        remaining = [c for c in candidates_sorted if c not in selection]
        selection.extend(remaining[:k - len(selection)])
    return selection[:k]

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status":"ok", "catalog_size": int(len(df)), "faiss_count": int(index.ntotal)}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not req.query and not req.url:
        raise HTTPException(status_code=400, detail="Provide 'query' or 'url'")

    query_text = req.query
    if not query_text:
        query_text = fetch_text_from_url(req.url)

    # initial retrieval pool
    pool = top_n_candidates(query_text, top_n=50)

    # choose reranker
    try:
        if groq_client:
            start = time.time()
            reranked = groq_rerank(query_text, pool)
            logging.info("Used Groq reranker (%.2fs)", time.time() - start)
        else:
            start = time.time()
            reranked = fallback_rerank(query_text, pool)
            logging.info("Used fallback reranker (%.2fs)", time.time() - start)
    except Exception as e:
        logging.warning("Reranker failed (%s). Falling back to offline reranker.", e)
        reranked = fallback_rerank(query_text, pool)

    # balancing and final selection
    k = int(req.k or 10)
    final = enforce_balancing(reranked, k)

    results = []
    for c in final:
        results.append({
            "assessment_name": c.get("assessment_name",""),
            "url": c.get("url",""),
            "score": float(c.get("score", 0.0)),
            "rationale": (c.get("rationale","") if USE_GROQ_EXPLAIN else "")
        })
    return {"query": query_text, "predictions": results}

@app.post("/batch_export")
def batch_export(req: BatchExportRequest):
    queries = req.queries or []
    k = int(req.k or 10)
    if not queries:
        raise HTTPException(status_code=400, detail="`queries` must be a non-empty list")
    rows = []
    for q in queries:
        pool = top_n_candidates(q, top_n=50)
        try:
            if groq_client:
                reranked = groq_rerank(q, pool)
            else:
                reranked = fallback_rerank(q, pool)
        except Exception:
            reranked = fallback_rerank(q, pool)
        final = enforce_balancing(reranked, k=k)
        for c in final:
            rows.append({"Query": q, "Assessment_url": c.get("url")})
    out_csv = os.path.join(DATA_DIR, "predictions.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return {"predictions_csv": out_csv, "rows": len(rows)}

# ---------------- Run guard ----------------
if __name__ == "__main__":
    print("Run with: uvicorn api_app:app --reload --port 8000")
