import math
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
import uuid

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2): return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = sum(a * a for a in v1) ** 0.5
    norm_b = sum(b * b for b in v2) ** 0.5
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot_product / (norm_a * norm_b)

class SemanticDB:
    """
    Vector Database substitute using Neural Embeddings and Cosine Similarity.
    Allows JARVIS to store and retrieve long-term facts, preferences, and historical
    solutions based on true semantic meaning.
    """
    def __init__(self, storage_path: Path, embedder_func: Callable[[str], List[float]] = None):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Dict[str, Any]] = {}  # id -> {text, metadata, embedding}
        self.embedder_func = embedder_func
        self._load()

    def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        doc_id = str(uuid.uuid4())
        vec = self.embedder_func(text) if self.embedder_func else []
        self.documents[doc_id] = {"text": text, "metadata": metadata or {}, "embedding": vec}
        self._save()
        return doc_id

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.documents or not self.embedder_func: return []
        
        query_vec = self.embedder_func(query)
        if not query_vec:
            return []
            
        results = []
        for doc_id, doc in self.documents.items():
            doc_vec = doc.get("embedding", [])
            if not doc_vec: continue
                
            sim = cosine_similarity(query_vec, doc_vec)
            if sim > 0:
                results.append((sim, {"id": doc_id, **doc}))
                
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

    def _load(self):
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            # Migrate old TF-IDF DBs to true Vectors
            migrated = False
            for doc_id, doc in self.documents.items():
                if "embedding" not in doc and self.embedder_func:
                    doc["embedding"] = self.embedder_func(doc["text"])
                    migrated = True
            
            if migrated:
                self._save()
