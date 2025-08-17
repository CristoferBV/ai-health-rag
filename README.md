# Mini Healthcare RAG (LangGraph + FastAPI)

- Ingest public PDFs/MD/TXT into a Chroma vector DB.
- Orchestrate a retrieve → rerank → answer graph with LangGraph.
- Serve a FastAPI endpoint: POST /ask {"question": "..."} → answer with citations.

## Quickstart
1) Put 3–5 files into `data/` (no PHI).
2) `python -m venv .venv && source .venv/bin/activate`
3) `pip install -r requirements.txt`
4) `python ingest/ingest.py`
5) Set `OPENAI_API_KEY` (optional; falls back to local embeddings).
6) `uvicorn app.api:app --reload --port 8080`
7) `curl -X POST http://127.0.0.1:8080/ask -H "Content-Type: application/json" -d '{"question":"What are the key recommendations?"}'`

## Docker
`docker build -t mini-rag .`
`docker run -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY mini-rag`

## Notes
- LangGraph handles deterministic multi-step flow and shared state.
- Returns short citations for transparency.
- Ready to deploy on Cloud Run with a single command once the image is pushed.
