#!/usr/bin/env python3
"""
evaluate_mapped.py

Reads data/test_queries_mapped.csv (Query, Relevant_urls, Mapped_products)
Calls /recommend for each Query and computes Recall@K using Mapped_products as ground-truth.

Usage:
  python evaluate_mapped.py --api http://127.0.0.1:8000/recommend --k 10
"""

import os
import csv
import argparse
import requests
from statistics import mean

MAPPED_CSV = os.path.join("data", "test_queries_mapped.csv")
DEFAULT_API = "http://127.0.0.1:8000/recommend"

def load_mapped(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Ensure keys exist
            q = r.get("Query","").strip()
            mapped = r.get("Mapped_products","").strip()
            mapped_list = [u.strip().rstrip("/") for u in mapped.split("|") if u.strip()]
            rows.append({"Query": q, "Mapped": mapped_list, "Raw": r})
    return rows

def recall_at_k(preds, truths):
    if not truths:
        return 0.0
    hits = len(set(preds) & set(truths))
    return hits / len(truths)

def call_api(api_url, query, k):
    try:
        resp = requests.post(api_url, json={"query": query, "k": k}, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        preds = [p.get("url","").rstrip("/") for p in j.get("predictions",[])]
        return preds
    except Exception as e:
        print("  [error] API call failed:", e)
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(MAPPED_CSV):
        print("Mapped CSV not found:", MAPPED_CSV)
        return

    rows = load_mapped(MAPPED_CSV)
    print(f"Loaded {len(rows)} mapped queries from {MAPPED_CSV}")

    recalls = []
    report_rows = []
    for i, r in enumerate(rows, start=1):
        q = r["Query"]
        truths = [u.rstrip("/") for u in r["Mapped"]]
        print(f"\n[{i}/{len(rows)}] Query: {q[:100]}")
        preds = call_api(args.api, q, args.k)
        if not preds:
            print("  No predictions returned (API error). Counted as 0 recall.")
            recalls.append(0.0)
            report_rows.append({"Query": q, "Recall@K": 0.0, "Predictions": "", "GroundTruth": "|".join(truths)})
            continue
        r_at_k = recall_at_k(preds[:args.k], truths)
        recalls.append(r_at_k)
        print(f"  Recall@{args.k}: {r_at_k:.3f}")
        for rank,u in enumerate(preds[:args.k], start=1):
            print(f"    {rank}. {u}")
        report_rows.append({"Query": q, "Recall@K": r_at_k, "Predictions": "|".join(preds[:args.k]), "GroundTruth": "|".join(truths)})

    if recalls:
        m = mean(recalls)
        print(f"\n✅ Mean Recall@{args.k}: {m:.4f} (n={len(recalls)})")
    else:
        print("No queries evaluated.")

    out_csv = os.path.join("data","eval_report.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query","Recall@K","Predictions","GroundTruth"])
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)
    print("Wrote evaluation report to:", out_csv)

if __name__ == '__main__':
    main()
