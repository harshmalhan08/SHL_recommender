#!/usr/bin/env python3
"""
build_index.py

Batched embedding + FAISS index builder.

Usage:
  python build_index.py --data-dir ./data --model all-MiniLM-L6-v2 --batch-size 128

Outputs (in data-dir):
  - catalog_indexed.csv   (original CSV + `text` column)
  - embeddings.npy        (numpy float32 array of embeddings)
  - catalog.faiss         (faiss index of embeddings)

Requirements:
  pip install sentence-transformers faiss-cpu pandas numpy tqdm
  # On Windows if faiss-cpu pip fails:
  pip install faiss-cpu --extra-index-url https://download.pytorch.org/whl/cpu
"""

import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

def make_text_column(df, title_col="assessment_name", desc_col="description", tags_col="tags"):
    def join_row(r):
        parts = []
        if title_col in r and pd.notna(r[title_col]):
            parts.append(str(r[title_col]))
        if desc_col in r and pd.notna(r[desc_col]):
            parts.append(str(r[desc_col]))
        if tags_col in r and pd.notna(r[tags_col]):
            # if tags stored as semicolon separated, include them spaced
            tags = str(r[tags_col])
            if ";" in tags:
                parts.append(" ".join([t.strip() for t in tags.split(";") if t.strip()]))
            else:
                parts.append(tags)
        return " \n ".join([p for p in parts if p])
    return df.apply(join_row, axis=1)

def encode_in_batches(model, texts, batch_size=128, show_progress=True):
    n = len(texts)
    embeddings = np.zeros((n, model.get_sentence_embedding_dimension()), dtype="float32")
    iterator = range(0, n, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(n + batch_size - 1)//batch_size, desc="Embedding batches")
    for i in iterator:
        j = min(i + batch_size, n)
        batch_texts = texts[i:j]
        emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        if emb.dtype != np.float32:
            emb = emb.astype("float32")
        embeddings[i:j] = emb
    return embeddings

def build_faiss_index(embeddings, use_cosine=True):
    """
    Build FAISS index.
    If use_cosine=True: normalize embeddings to unit length and use IndexFlatIP (inner product == cosine).
    Otherwise use IndexFlatL2.
    """
    if use_cosine:
        # normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, embeddings
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data", help="Folder containing catalog.csv (output directory)")
    p.add_argument("--csv-name", default="catalog.csv", help="Input catalog CSV filename")
    p.add_argument("--out-csv", default="catalog_indexed.csv", help="Output CSV with text column")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformers model name")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding")
    p.add_argument("--use-cosine", action="store_true", help="Normalize embeddings and use InnerProduct (cosine) matching")
    p.add_argument("--save-embeddings", action="store_true", help="Also save embeddings.npy (optional)")
    args = p.parse_args()

    data_dir = args.data_dir
    in_csv_path = os.path.join(data_dir, args.csv_name)
    if not os.path.exists(in_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {in_csv_path}")

    print("Loading catalog CSV:", in_csv_path)
    df = pd.read_csv(in_csv_path)
    print(f"Loaded {len(df)} rows.")

    print("Creating text column (title + description + tags)...")
    # Ensure columns exist
    for col in ["assessment_name", "description", "tags"]:
        if col not in df.columns:
            df[col] = ""

    df["text"] = make_text_column(df, "assessment_name", "description", "tags")
    out_csv_path = os.path.join(data_dir, args.out_csv)
    df.to_csv(out_csv_path, index=False)
    print("Wrote indexed CSV:", out_csv_path)

    print("Loading embedding model:", args.model)
    model = SentenceTransformer(args.model)

    texts = df["text"].fillna("").astype(str).tolist()
    print(f"Encoding {len(texts)} items in batches (batch_size={args.batch_size}) ...")
    embeddings = encode_in_batches(model, texts, batch_size=args.batch_size, show_progress=True)

    # Build FAISS index (cosine by default if requested)
    if args.use_cosine:
        print("Normalizing embeddings for cosine similarity and building IndexFlatIP ...")
    else:
        print("Building L2 (IndexFlatL2) index ...")

    index, final_embeddings = build_faiss_index(embeddings, use_cosine=args.use_cosine)

    faiss_path = os.path.join(data_dir, "catalog.faiss")
    faiss.write_index(index, faiss_path)
    print("Saved FAISS index to:", faiss_path)

    if args.save_embeddings:
        emb_path = os.path.join(data_dir, "embeddings.npy")
        np.save(emb_path, final_embeddings)
        print("Saved embeddings to:", emb_path)

    # Save a small metadata file
    meta = {
        "num_items": int(len(df)),
        "embedding_dim": int(final_embeddings.shape[1]),
        "use_cosine": bool(args.use_cosine),
        "model": args.model
    }
    with open(os.path.join(data_dir, "index_meta.json"), "w", encoding="utf-8") as f:
        import json as _j
        _j.dump(meta, f, indent=2)
    print("Wrote index_meta.json")

    print("Done. Catalog indexed:", meta)

if __name__ == "__main__":
    main()
