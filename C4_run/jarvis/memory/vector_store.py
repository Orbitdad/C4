"""
FAISS-backed Structured Vector Memory Store for C4.

Stores three categories of memory: user, system, task.
Each entry has: content, type, importance (0.0–1.0), timestamp, and a FAISS vector.

Persistence:
    - FAISS index  → data/memory/vector_store.faiss
    - Metadata     → data/memory/vector_store_meta.json
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy FAISS import — degrade gracefully if not installed
_faiss = None

def _get_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            logger.warning(
                "[VectorMemoryStore] faiss-cpu not installed. "
                "Run: pip install faiss-cpu. Falling back to brute-force search."
            )
    return _faiss


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single structured memory record."""
    id: str
    content: str
    type: str  # "user" | "system" | "task"
    importance: float  # 0.0 – 1.0
    timestamp: str  # ISO-8601
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)  # Embeddings stored in FAISS, not JSON
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=d.get("id", ""),
            content=d.get("content", ""),
            type=d.get("type", "task"),
            importance=float(d.get("importance", 0.5)),
            timestamp=d.get("timestamp", ""),
            embedding=d.get("embedding"),
        )


# ---------------------------------------------------------------------------
# System Seeds — injected on first boot
# ---------------------------------------------------------------------------

SYSTEM_SEEDS: List[Tuple[str, str, float]] = [
    ("qwen2:7b is the primary brain model for reasoning and planning.", "system", 1.0),
    ("deepseek-coder:6.7b is used exclusively for code generation and debugging tasks.", "system", 1.0),
    ("nomic-embed-text is used for all embeddings and semantic memory operations.", "system", 1.0),
    ("C4 must never claim to have done something it has not actually done.", "system", 1.0),
    ("Always respond concisely. Maximum 3 sentences unless the user explicitly asks for more detail.", "system", 0.9),
    ("C4 addresses its creator as 'sir'. Tone is calm, precise, authoritative.", "system", 0.9),
    ("For simple web tasks, prefer Vanilla HTML/CSS/JS. Use React only when explicitly requested.", "system", 0.85),
    ("All systems run fully local — no cloud APIs for core reasoning.", "system", 1.0),
]


# ---------------------------------------------------------------------------
# Vector Memory Store
# ---------------------------------------------------------------------------

class VectorMemoryStore:
    """
    FAISS-backed persistent vector database for structured memory entries.

    Uses ``nomic-embed-text`` via the provided ``embed_fn`` callable.
    Falls back to brute-force NumPy cosine search if FAISS is not installed.
    """

    def __init__(
        self,
        storage_dir: Path,
        embed_fn: Callable[[str], List[float]],
        dimension: int = 768,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embed_fn = embed_fn
        self.dimension = dimension

        self._index_path = self.storage_dir / "vector_store.faiss"
        self._meta_path = self.storage_dir / "vector_store_meta.json"

        # id → MemoryEntry (minus embeddings)
        self._entries: Dict[str, MemoryEntry] = {}
        # Ordered list of ids matching FAISS row order
        self._id_order: List[str] = []
        # FAISS index (or None if unavailable)
        self._index = None
        # Raw vectors (NumPy fallback)
        self._vectors: Optional[np.ndarray] = None

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        mem_type: str = "task",
        importance: float = 0.5,
        entry_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Embed and store a new memory entry. Returns the created entry."""
        vec = self._safe_embed(content)
        if vec is None:
            logger.warning("[VectorMemoryStore] Embedding failed — storing without vector.")

        entry = MemoryEntry(
            id=entry_id or f"mem-{uuid.uuid4().hex[:12]}",
            content=content,
            type=mem_type,
            importance=max(0.0, min(1.0, importance)),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._entries[entry.id] = entry
        self._id_order.append(entry.id)

        if vec is not None:
            arr = np.array([vec], dtype=np.float32)
            faiss = _get_faiss()
            if faiss and self._index is not None:
                self._index.add(arr)
            else:
                # NumPy fallback
                if self._vectors is not None:
                    self._vectors = np.vstack([self._vectors, arr])
                else:
                    self._vectors = arr

        self._save()
        logger.debug(f"[VectorMemoryStore] Stored [{entry.type}] (imp={entry.importance:.2f}): {entry.content[:60]}...")
        return entry

    def search(
        self,
        query: str,
        top_k: int = 5,
        type_filter: Optional[List[str]] = None,
    ) -> List[Tuple[float, MemoryEntry]]:
        """
        Semantic search. Returns ``[(similarity_score, MemoryEntry), ...]``
        sorted descending by similarity.
        """
        if not self._entries:
            return []

        vec = self._safe_embed(query)
        if vec is None:
            return []

        query_arr = np.array([vec], dtype=np.float32)
        n = len(self._id_order)

        faiss = _get_faiss()
        if faiss and self._index is not None and self._index.ntotal > 0:
            # FAISS path
            search_k = min(top_k * 3, n)  # Over-fetch for post-filtering
            distances, indices = self._index.search(query_arr, search_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= n:
                    continue
                eid = self._id_order[idx]
                entry = self._entries.get(eid)
                if entry is None:
                    continue
                if type_filter and entry.type not in type_filter:
                    continue
                # FAISS IndexFlatIP returns inner product (cosine for normalized vecs)
                results.append((float(dist), entry))
            return results[:top_k]
        elif self._vectors is not None and len(self._vectors) > 0:
            # NumPy brute-force fallback
            sims = self._cosine_batch(query_arr, self._vectors)
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            results = []
            for idx, sim in ranked:
                if idx >= n:
                    continue
                eid = self._id_order[idx]
                entry = self._entries.get(eid)
                if entry is None:
                    continue
                if type_filter and entry.type not in type_filter:
                    continue
                results.append((float(sim), entry))
                if len(results) >= top_k:
                    break
            return results

        return []

    def find_duplicates(self, content: str, threshold: float = 0.92) -> List[MemoryEntry]:
        """Return existing entries whose similarity to ``content`` exceeds ``threshold``."""
        results = self.search(content, top_k=3)
        return [entry for sim, entry in results if sim >= threshold]

    def count(self) -> int:
        return len(self._entries)

    def get_all(self, mem_type: Optional[str] = None) -> List[MemoryEntry]:
        """Return all entries, optionally filtered by type."""
        entries = list(self._entries.values())
        if mem_type:
            entries = [e for e in entries if e.type == mem_type]
        return entries

    def seed_system_memories(self) -> int:
        """
        Populate default system memories if the store is fresh.
        Returns the number of seeds written.
        """
        existing_system = [e for e in self._entries.values() if e.type == "system"]
        if existing_system:
            return 0  # Already seeded
        count = 0
        for content, mem_type, importance in SYSTEM_SEEDS:
            self.add(content, mem_type=mem_type, importance=importance)
            count += 1
        logger.info(f"[VectorMemoryStore] Seeded {count} system memories on first boot.")
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        # Save metadata JSON
        meta = {
            "id_order": self._id_order,
            "entries": {eid: e.to_dict() for eid, e in self._entries.items()},
            "dimension": self.dimension,
        }
        try:
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[VectorMemoryStore] Failed to save metadata: {e}")

        # Save FAISS index
        faiss = _get_faiss()
        if faiss and self._index is not None:
            try:
                faiss.write_index(self._index, str(self._index_path))
            except Exception as e:
                logger.error(f"[VectorMemoryStore] Failed to save FAISS index: {e}")
        elif self._vectors is not None:
            # Save raw vectors as .npy fallback
            npy_path = self.storage_dir / "vector_store_vectors.npy"
            try:
                np.save(str(npy_path), self._vectors)
            except Exception as e:
                logger.error(f"[VectorMemoryStore] Failed to save vectors: {e}")

    def _load(self) -> None:
        # Load metadata
        if self._meta_path.is_file():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._id_order = meta.get("id_order", [])
                self._entries = {}
                for eid, edata in meta.get("entries", {}).items():
                    self._entries[eid] = MemoryEntry.from_dict(edata)
                self.dimension = meta.get("dimension", self.dimension)
            except Exception as e:
                logger.error(f"[VectorMemoryStore] Failed to load metadata: {e}")
                self._entries = {}
                self._id_order = []

        # Load FAISS index
        faiss = _get_faiss()
        if faiss and self._index_path.is_file():
            try:
                self._index = faiss.read_index(str(self._index_path))
                logger.info(f"[VectorMemoryStore] Loaded FAISS index: {self._index.ntotal} vectors.")
                return
            except Exception as e:
                logger.error(f"[VectorMemoryStore] Failed to load FAISS index: {e}")

        # Try NumPy fallback
        npy_path = self.storage_dir / "vector_store_vectors.npy"
        if npy_path.is_file():
            try:
                self._vectors = np.load(str(npy_path))
                logger.info(f"[VectorMemoryStore] Loaded NumPy vectors: {len(self._vectors)} rows.")
                return
            except Exception as e:
                logger.error(f"[VectorMemoryStore] Failed to load vectors: {e}")

        # Initialize fresh index
        if faiss:
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"[VectorMemoryStore] Created fresh FAISS index (dim={self.dimension}).")
        else:
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)
            logger.info(f"[VectorMemoryStore] Using NumPy fallback (dim={self.dimension}).")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _safe_embed(self, text: str) -> Optional[np.ndarray]:
        """Embed text, returning a normalized float32 vector or None."""
        try:
            raw = self.embed_fn(text)
            if not raw:
                return None
            vec = np.array(raw, dtype=np.float32)

            # Auto-detect dimension on first successful embed
            if vec.shape[0] != self.dimension:
                if self.count() == 0:
                    logger.info(
                        f"[VectorMemoryStore] Auto-adjusting dimension: {self.dimension} → {vec.shape[0]}"
                    )
                    self.dimension = vec.shape[0]
                    faiss = _get_faiss()
                    if faiss:
                        self._index = faiss.IndexFlatIP(self.dimension)
                    else:
                        self._vectors = np.empty((0, self.dimension), dtype=np.float32)
                else:
                    logger.warning(
                        f"[VectorMemoryStore] Dimension mismatch: got {vec.shape[0]}, expected {self.dimension}"
                    )
                    return None

            # L2-normalize for cosine similarity via inner product
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            logger.error(f"[VectorMemoryStore] Embedding error: {e}")
            return None

    @staticmethod
    def _cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between a single query and a matrix of vectors."""
        if matrix.shape[0] == 0:
            return np.array([])
        # Normalize
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return (m_norms @ q_norm.T).flatten()
