import os
import re
import json
import numpy as np
from typing import TypedDict
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.documents_map: dict = {}
    
    @staticmethod
    def semantic_chunking(query: str, max_chunk_size: int = 100, overlap: int = 0) -> list:
        sentences = re.split(r"(?<=[.!?])\s+", query)
        step = max_chunk_size - overlap

        chunks = []
        start = 0
        while start < len(sentences):
            end = start + max_chunk_size
            chunks.append(sentences[start:end])
            if end >= len(sentences):
                break
            start += step

        return chunks
                
    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        embedding: np.ndarray = self.model_name.encode([text], convert_to_numpy=True)
        return embedding[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc

        movies_list = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model_name.encode(movies_list, convert_to_numpy=True, show_progress_bar=True)
        np.save("cache/embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc

        if os.path.exists("cache/embeddings.npy"):
            self.embeddings = np.load("cache/embeddings.npy")
            if self.embeddings and self.embeddings.shape[0] == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        scored = [
            (cosine_similarity(query_embedding, doc_embedding), doc)
            for doc_embedding, doc in zip(self.embeddings, self.documents)
        ]

        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in scored[:limit]
        ]

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.chunks = []
        self.metadata = []
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc
        
        for doc_idx, doc in enumerate(self.documents):
            if len(doc['description']) != 0:
                description = doc['description']
                semantic_chunking = self.semantic_chunking(description, max_chunk_size=4, overlap=1)
                for idx,chunk in enumerate(semantic_chunking):
                    chunk_text = ' '.join(chunk)
                    self.metadata.append({
                        "movie_idx": doc_idx,
                        "chunk_idx": idx,
                        "total_chunks": len(semantic_chunking),
                    })
                    self.chunks.append(chunk_text)
        self.chunk_embeddings = self.model_name.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        self.chunk_metadata = self.metadata
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(self.chunks)}, f, indent=2)
            
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None or self.documents is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        embedding = self.generate_embedding(query)
        chunk_scores = []
        for chunk_idx, chunk in enumerate(self.chunk_embeddings):
            chunk_result = {
                "chunk_idx" : chunk_idx,
                "movie_idx" : self.chunk_metadata[chunk_idx]["movie_idx"],
                "score" : cosine_similarity(embedding, chunk)
             }
            chunk_scores.append(chunk_result)
        best_chunk = {}
        for c in chunk_scores:
            movie_idx = c["movie_idx"]
            if movie_idx not in best_chunk or c["score"] > best_chunk[movie_idx]["score"]:
                best_chunk[movie_idx] = c
        best_chunk_list = list(best_chunk.values())
        best_chunk_list.sort(key=lambda x: x["score"], reverse=True)
        best_chunk_list = best_chunk_list[:limit]
        results = []
        from utils.search_utils import format_search_result
        for c in best_chunk_list:
            doc = self.documents[c["movie_idx"]]
            results.append(format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"][:100],
                score=c["score"],
                metadata={"chunk_idx": c["chunk_idx"]},
            ))
        return results

def verify_model() -> None:
    semantic_search = SemanticSearch()
    model = semantic_search.model_name
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str) -> None:
    embedding = SemanticSearch().generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    embedding = SemanticSearch().generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    from utils.search_utils import get_movies
    movies = get_movies()
    semantic_search = SemanticSearch()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")