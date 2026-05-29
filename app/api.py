import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from cli.keyword_search import InvertedIndex
import uvicorn

app = FastAPI(title="My RAG Search API")

def get_allowed_origins(default_origin: str = "https://raymondww.github.io") -> list[str]:
    raw_origins = os.getenv("ALLOWED_ORIGINS")
    if raw_origins is None or not raw_origins.strip():
        raw_origins = default_origin
    stripped = [origin.strip() for origin in raw_origins.split(",")]
    return [origin for origin in stripped if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["GET"],
    allow_headers=["*"],
)

index = InvertedIndex()

@app.on_event("startup")
def load_index():
    try:
        index.load()
        print("Index loaded successfully.")
    except FileNotFoundError:
        print("Warning: index not found. Run build first.")

@app.get("/")
def root():
    return {"status": "ok", "docs": len(index.docmap)}

@app.get("/search")
def search(query: str = Query(..., min_length=1), limit: int = Query(5, ge=1, le=20)):
    results = index.bm25_search(query, limit)
    return {"query": query, "results": results}

@app.get("/health")
def health():
    return {"status": "healthy", "indexed_docs": len(index.docmap)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
