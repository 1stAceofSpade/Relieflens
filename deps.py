import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

INCIDENTS_COLLECTION = "incidents"
MEMORY_COLLECTION = "memory"

EMBED_DIM = 384
MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(MODEL_NAME)

def embed_text(text: str):
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()
