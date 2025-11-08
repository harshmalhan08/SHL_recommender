#!/usr/bin/env python3
"""
Improved mapper: map ground-truth URLs to product pages with robust fetching.

Features:
 - Retries requests (configurable)
 - Optional Playwright render fallback (if playwright installed)
 - If fetching fails, falls back to mapping from Query text
 - Pre-encodes catalog embeddings for fast fallback similarity search
"""
import os, json, argparse, time, random, logging
from typing import List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import faiss

# Optional Playwright
USE_PLAYWRIGHT = True
try:
    if USE_PLAYWRIGHT:
        from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# Config (tweak if needed)
DATA_DIR = "data"
CATALOG_INDEXED_CSV = os.path.join(DATA_DIR, "catalog_indexed.csv")
FAISS_PATH = os.path.join(DATA_DIR, "catalog.faiss")
INDEX_META = os.path.join(DATA_DIR, "index_meta.json")
GROUND_TRUTH = os.path.join(DATA_DIR, "test_queries.csv")
OUT_PATH = os.path.join(DATA_DIR, "test_queries_mapped.csv")
MODEL_NAME = "all-MiniLM-L6-v2"

HTTP_TIMEOUT = 25            # seconds per request (increased)
HTTP_RETRIES = 3             # number of attempts
RETRY_BACKOFF = 1.2          # multiplier
SLEEP_BETWEEN_FETCHES = 0.15 # polite delay between fetches
TRUNCATE_CHARS = 20000       # limit page text length

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_index_and_meta():
    if not os.path.exists(CATALOG_INDEXED_CSV) or not os.path.exists(FAISS_PATH):
        raise FileNotFoundError("Ensure catalog_indexed.csv and catalog.faiss exist in data/ (run build_index.py first).")
    df = pd.read_csv(CATALOG_INDEXED_CSV).fillna("")
    index = faiss.read_index(FAISS_PATH)
    use_cosine = False
    if os.path.exists(INDEX_META):
        try:
            with open(INDEX_META, "r", encoding="utf-8") as f:
                meta = json.load(f)
                use_cosine = bool(meta.get("use_cosine", False))
        except Exception:
            use_cosine = False
    return df, index, use_cosine

def fetch_page_text_requests(url: str, timeout=HTTP_TIMEOUT, retries=HTTP_RETRIES):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SHL-mapper/1.0)"}
    attempt = 0
    backoff = 1.0
    while attempt < retries:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            text = r.text
            # crude HTML -> plaintext
            import re
            text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", text)
            text = re.sub(r"(?s)<[^>]*>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text[:TRUNCATE_CHARS]
        except Exception as e:
            logging.debug("requests attempt %d failed for %s: %s", attempt+1, url, e)
            attempt += 1
            time.sleep(backoff)
            backoff *= RETRY_BACKOFF
    return ""

def fetch_page_text_playwright(url: str):
    if sync_playwright is None:
        return ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=35000, wait_until="load")
                page.wait_for_timeout(900)
                html = page.content()
            finally:
                browser.close()
        # strip tags
        import re
        text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
        text = re.sub(r"(?s)<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text[:TRUNCATE_CHARS]
    except Exception as e:
        logging.debug("playwright fetch failed for %s: %s", url, e)
        return ""

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64):
    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((n, dim), dtype="float32")
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        batch = texts[i:j]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        if emb.dtype != np.float32:
            emb = emb.astype("float32")
        embeddings[i:j] = emb
    return embeddings

def normalize_rows(x: np.ndarray):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def robust_map(k: int = 5, force_query_fallback: bool = False):
    df_catalog, index, use_cosine = load_index_and_meta()
    logging.info("Catalog items: %d, index use_cosine=%s", len(df_catalog), use_cosine)

    # load ground truth robustly
    try:
        gt_df = pd.read_csv(GROUND_TRUTH).fillna("")
    except Exception:
        # tolerant parse fallback (split on first comma)
        rows = []
        with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header = 0
        if len(lines) and "query" in lines[0].lower() and "relevant" in lines[0].lower():
            header = 1
        for line in lines[header:]:
            if not line.strip(): continue
            idxc = line.find(",")
            if idxc == -1: continue
            q = line[:idxc].strip().strip('"').strip("'")
            urls = line[idxc+1:].strip().strip('"').strip("'")
            rows.append({"Query": q, "Relevant_urls": urls})
        gt_df = pd.DataFrame(rows)

    # prepare model
    logging.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # Pre-encode catalog texts only once (if needed for fallback)
    catalog_texts = df_catalog["text"].fillna("").astype(str).tolist()
    catalog_emb = embed_texts(model, catalog_texts, batch_size=64)
    if use_cosine:
        catalog_emb = normalize_rows(catalog_emb)

    out_rows = []
    for i, row in gt_df.iterrows():
        query = str(row.get("Query", "")).strip()
        relevant_raw = str(row.get("Relevant_urls", "")).strip()
        gt_urls = [u.strip().rstrip("/") for u in relevant_raw.split("|") if u.strip()]
        logging.info("[%d/%d] Query: %s", i+1, len(gt_df), (query[:80] + "...") if len(query) > 80 else query)
        mapped = []

        targets = gt_urls[:] if gt_urls else ["__QUERY_TEXT__"]

        for t in targets:
            page_text = ""
            if t == "__QUERY_TEXT__" or force_query_fallback:
                page_text = query
                logging.debug("Using query text as target fallback.")
            else:
                # try requests with retries
                page_text = fetch_page_text_requests(t)
                if not page_text and sync_playwright:
                    logging.info("Requests failed; trying Playwright render for %s", t)
                    page_text = fetch_page_text_playwright(t)
                if not page_text:
                    logging.warning("No text for target %s after retries. Will fallback to Query text.", t)
                    page_text = query

            # embed target
            emb = model.encode([page_text], convert_to_numpy=True).astype("float32")
            if use_cosine:
                emb = normalize_rows(emb)

            # search
            try:
                D, I = index.search(emb, k)
            except Exception as e:
                logging.warning("Index search failed (%s). Falling back to cosine dot with pre-encoded catalog.", e)
                # fallback: compute dot product similarity with precomputed catalog_emb
                qv = emb[0]
                if use_cosine:
                    sims = np.dot(catalog_emb, qv)
                else:
                    # for L2, compute negative L2 distance as proxy
                    sims = -np.linalg.norm(catalog_emb - qv, axis=1)
                top_idx = np.argsort(-sims)[:k]
                I = np.expand_dims(top_idx, axis=0)
                D = np.expand_dims(sims[top_idx], axis=0)

            # map to urls
            hits = []
            for dist, idx_hit in zip(D[0].tolist(), I[0].tolist()):
                try:
                    url_hit = df_catalog.iloc[int(idx_hit)]["url"]
                except Exception:
                    url_hit = ""
                hits.append({"url": url_hit, "score": float(dist)})
            for h in hits:
                if h["url"] and h["url"] not in mapped:
                    mapped.append(h["url"])

            time.sleep(SLEEP_BETWEEN_FETCHES + random.random() * 0.05)

        out_rows.append({
            "Query": query,
            "Relevant_urls": relevant_raw,
            "Mapped_products": "|".join(mapped[:k])
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_PATH, index=False)
    logging.info("Wrote mapping to: %s", OUT_PATH)
    return OUT_PATH

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="how many product urls to map per ground-truth URL")
    ap.add_argument("--force-query-fallback", action="store_true", help="always map using query text instead of trying to fetch ground-truth URLs")
    args = ap.parse_args()
    res = robust_map(k=args.k, force_query_fallback=args.force_query_fallback)
    print("Done. Mapped file:", res)
