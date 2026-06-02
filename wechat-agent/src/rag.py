import os
os.environ["OTEL_SDK_DISABLED"] = "true"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from config import VECTORDB_DIR, EMBEDDING_MODEL, EMBEDDING_DIM, RAG_TOP_K


class RAGRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=str(VECTORDB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="chat_history",
            metadata={"hnsw:space": "cosine"},
        )

    def add_messages(self, messages: List[str], ids: List[str]):
        if not messages:
            return
        embeddings = self.embedder.encode(messages, show_progress_bar=False).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=messages,
            ids=ids,
        )

    def search(self, query: str, k: int = RAG_TOP_K) -> List[str]:
        query_emb = self.embedder.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
        )
        return results["documents"][0] if results["documents"] else []

    def count(self) -> int:
        return self.collection.count()
