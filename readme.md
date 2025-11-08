# SHL GenAI Assessment Recommender

## Overview
This system recommends SHL assessments based on job descriptions using FAISS semantic retrieval and Groq LLM reranking.

## Steps to Run
1. `pip install -r requirements.txt`
2. `python build_index.py`
3. `uvicorn api_app:app --reload --port 8000`
4. Open `frontend/index.html` and query SHL products.

## Features
- Semantic search with Sentence-Transformers
- LLM reranker via Groq API
- Interactive web UI with Tailwind
- Batch export support
