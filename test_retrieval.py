# test_retrieval.py
import faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/catalog_indexed.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/catalog.faiss")

q = "I need a coding and logical reasoning assessment for backend developers"
q_emb = model.encode([q])
# If you built L2 index (default), do NOT normalize; if you used --use-cosine normalize the vector.
# This build used L2, so we skip normalization.
D, I = index.search(q_emb.astype("float32"), k=10)
for rank, idx in enumerate(I[0], start=1):
    print(f"{rank}. {df.iloc[idx]['assessment_name']} - {df.iloc[idx]['url']}")
