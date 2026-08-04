from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from rag_engine.keyword_search import InvertedIndex

app = FastAPI(title="Keyword Search Pipeline Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

index = InvertedIndex()
index.build()


@app.get("/api/search")
def search(q: str, limit: int = 5):
    return index.search_with_trace(q, limit)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
