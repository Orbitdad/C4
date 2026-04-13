"""
Face Recognizer: Embedding comparison and identity matching.

Architecture Decision:
    Deliberately separated from IdentityManager. This class ONLY compares
    embedding vectors (pure math), making it easily unit-testable and
    replaceable with a different metric (e.g., Euclidean, FAISS) without
    touching the identity store.

Performance Notes:
    Cosine similarity on 512-dim float32 vectors is O(N) per identity.
    For <= 100 registered users this is negligible (<1ms). For larger
    deployments, replace with FAISS ANN search.

Threshold Tuning:
    MATCH_THRESHOLD = 0.45 is conservative (low false-positive risk).
    ArcFace similarity ranges 0.0 (completely different) to 1.0 (identical).
    Empirically: same person across sessions ~0.55-0.85, strangers <0.35.
    Raise to 0.50 for stricter matching; lower to 0.40 for noisier environments.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Compares a face embedding against a collection of known embeddings.
    Returns the best-matching identity ID and its confidence score.
    """

    # Minimum cosine similarity to accept a match (tunable)
    MATCH_THRESHOLD: float = 0.45

    def __init__(self, threshold: float = MATCH_THRESHOLD) -> None:
        self.threshold = threshold

    def compare(
        self,
        probe_embedding: np.ndarray,
        gallery: List[Tuple[str, List[np.ndarray]]],
    ) -> Tuple[Optional[str], float]:
        """
        Compare a probe embedding against a gallery of known identities.

        Args:
            probe_embedding: 512-dim float32 ArcFace embedding of the detected face.
            gallery: List of (identity_id, [embeddings]) tuples from IdentityManager.

        Returns:
            (identity_id, confidence) if match found above threshold.
            (None, 0.0) if no match — treat as UNKNOWN_USER.
        """
        if probe_embedding is None or len(probe_embedding) == 0:
            return None, 0.0

        probe_norm = self._normalize(probe_embedding)
        best_id: Optional[str] = None
        best_score: float = 0.0

        for identity_id, embeddings in gallery:
            if not embeddings:
                continue
            # Score against each stored embedding, take the maximum.
            # Multi-embedding per user naturally handles angle/lighting variance.
            for stored_emb in embeddings:
                score = float(np.dot(probe_norm, self._normalize(stored_emb)))
                if score > best_score:
                    best_score = score
                    best_id = identity_id

        if best_score >= self.threshold:
            logger.debug(
                f"[FaceRecognizer] Matched '{best_id}' with confidence {best_score:.3f}"
            )
            return best_id, best_score

        logger.debug(
            f"[FaceRecognizer] No match. Best score was {best_score:.3f} < threshold {self.threshold}"
        )
        return None, best_score

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector to unit length for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            return embedding
        return embedding / norm

    def compute_average_embedding(
        self, embeddings: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute the centroid of multiple embeddings for a user.
        Used to collapse N registration samples into a single representative vector.
        """
        if not embeddings:
            return None
        stacked = np.stack([self._normalize(e) for e in embeddings], axis=0)
        mean = stacked.mean(axis=0)
        return self._normalize(mean)
