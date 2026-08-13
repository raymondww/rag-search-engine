from PIL import Image
from sentence_transformers import SentenceTransformer

from rag_engine.semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents: list[dict] | None = None, model_name: str = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = (
            self.model.encode(self.texts, show_progress_bar=True) if self.texts else []
        )

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        scored = []
        for doc, text_embedding in zip(self.documents, self.text_embeddings):
            score = cosine_similarity(image_embedding, text_embedding)
            scored.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "score": float(score),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]


def verify_image_embedding(image_path: str):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
    return embedding


def image_search_command(image_path: str) -> list[dict]:
    from utils.search_utils import get_movies

    movies = get_movies()
    search = MultimodalSearch(movies)
    return search.search_with_image(image_path)
