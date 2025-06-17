"""
ticket_similarity.py
====================

Light-weight, in-memory similarity engine.

• Uses SentenceTransformers to create *unit-norm* embeddings
  so that inner-product == cosine similarity.
• Designed to be **read-only** at inference time; batch updates
  should be done via the trainer job.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Ticket:
    id: str
    title: str
    description: str
    requester_email: Optional[str] = None
    created_at: Optional[datetime] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    department_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    def get_combined_text(self) -> str:
        return f"{self.title}. {self.description}"


# --------------------------------------------------------------------------- #
#  Model caching and authentication setup
# --------------------------------------------------------------------------- #
def _get_model_cache_dir() -> str:
    """Get the model cache directory, creating it if needed."""
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir

def _get_hf_token() -> Optional[str]:
    """Get Hugging Face token from environment or token file."""
    # Try environment variable first
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    
    # Try token file
    token_file = os.getenv("HF_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        return Path(token_file).read_text().strip()
    
    return None

def _create_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Create SentenceTransformer with authentication and caching."""
    cache_dir = _get_model_cache_dir()
    token = _get_hf_token()
    
    kwargs = {
        "cache_folder": cache_dir,
    }
    
    # Check if we're in offline mode
    is_offline = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1" or os.getenv("HF_HUB_OFFLINE", "0") == "1"
    
    if not is_offline and token:
        kwargs["token"] = token
        logger.info("Using Hugging Face authentication token")
    elif not is_offline:
        logger.warning("No Hugging Face token found - may encounter rate limits")
    else:
        logger.info("Running in offline mode - using cached models only")
    
    logger.info("Loading SentenceTransformer model %s with cache dir %s", model_name, cache_dir)
    
    try:
        return SentenceTransformer(model_name, **kwargs)
    except Exception as e:
        if is_offline:
            logger.error("Failed to load model %s in offline mode. Ensure model is cached: %s", model_name, e)
        raise


# --------------------------------------------------------------------------- #
#  Main detector
# --------------------------------------------------------------------------- #
class TicketSimilarityDetector:
    """
    Brute-force cosine similarity detector.

    Note: For large corpora use the FAISS-based ANN path instead;
    this class is still useful for small offline analysis and for
    generating the training snapshot.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
    ) -> None:
        self.model = _create_sentence_transformer(model_name)
        self.similarity_threshold: float = similarity_threshold

        self.tickets: List[Ticket] = []
        self.ticket_index: Dict[str, int] = {}  # id → position
        self.embeddings_matrix: Optional[np.ndarray] = None

    # -----------------------------  ingestion  ----------------------------- #
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a *unit-norm* embedding (normalize_embeddings=True)
        so we can use dot-product as cosine similarity later.
        """
        return self.model.encode(
            [text], normalize_embeddings=True
        )[0]  # shape: (dim,)

    def add_ticket(self, ticket: Ticket) -> None:
        if ticket.embedding is None:
            ticket.embedding = self._generate_embedding(ticket.get_combined_text())

        self.ticket_index[ticket.id] = len(self.tickets)
        self.tickets.append(ticket)
        self._update_embeddings_matrix()

    def add_tickets_batch(self, tickets: List[Ticket]) -> None:
        texts: List[str] = [
            t.get_combined_text() for t in tickets if t.embedding is None
        ]
        if texts:
            new_embs = self.model.encode(texts, normalize_embeddings=True)
            emb_iter = iter(new_embs)
            for t in tickets:
                if t.embedding is None:
                    t.embedding = next(emb_iter)

        start = len(self.tickets)
        for offset, t in enumerate(tickets):
            self.ticket_index[t.id] = start + offset
            self.tickets.append(t)

        self._update_embeddings_matrix()

    def _update_embeddings_matrix(self) -> None:
        if self.tickets:
            self.embeddings_matrix = np.vstack([t.embedding for t in self.tickets])

    # -----------------------------  querying  ------------------------------ #
    def find_similar_tickets(
        self, query_ticket: Ticket, top_k: int = 5
    ) -> List[Tuple[Ticket, float]]:
        if not self.tickets:
            return []

        if query_ticket.embedding is None:
            query_ticket.embedding = self._generate_embedding(
                query_ticket.get_combined_text()
            )

        # Filter tickets to only consider those with the same department_id
        if query_ticket.department_id is not None:
            # Find indices of tickets with the same department_id
            same_dept_indices = [
                i for i, ticket in enumerate(self.tickets)
                if ticket.department_id == query_ticket.department_id
            ]
            
            if not same_dept_indices:
                return []
                
            # Get embeddings for same department tickets only
            same_dept_embeddings = self.embeddings_matrix[same_dept_indices]
            sims = cosine_similarity(
                query_ticket.embedding.reshape(1, -1), same_dept_embeddings
            )[0]
            
            # Sort by similarity and map back to original indices
            sorted_indices = np.argsort(sims)[::-1]
            
            results: List[Tuple[Ticket, float]] = []
            for i in sorted_indices[:top_k]:
                if sims[i] >= self.similarity_threshold:
                    original_idx = same_dept_indices[i]
                    results.append((self.tickets[original_idx], float(sims[i])))
            return results
        else:
            # If no department_id, fall back to original behavior
            sims = cosine_similarity(
                query_ticket.embedding.reshape(1, -1), self.embeddings_matrix
            )[0]
            idxs = np.argsort(sims)[::-1]

            results: List[Tuple[Ticket, float]] = []
            for i in idxs[:top_k]:
                if sims[i] >= self.similarity_threshold:
                    results.append((self.tickets[i], float(sims[i])))
            return results

    def find_potential_duplicates(
        self, min_similarity: float | None = None
    ) -> List[Tuple[Ticket, Ticket, float]]:
        min_similarity = min_similarity or self.similarity_threshold
        if len(self.tickets) < 2:
            return []

        sim_matrix = cosine_similarity(self.embeddings_matrix)
        duplicates: List[Tuple[Ticket, Ticket, float]] = []
        n = len(self.tickets)
        for i in range(n):
            for j in range(i + 1, n):
                # Only compare tickets from the same department
                ticket_i = self.tickets[i]
                ticket_j = self.tickets[j]
                
                # Skip if tickets don't have the same department_id
                if (ticket_i.department_id is not None and 
                    ticket_j.department_id is not None and 
                    ticket_i.department_id != ticket_j.department_id):
                    continue
                
                sim = sim_matrix[i, j]
                if sim >= min_similarity:
                    duplicates.append((ticket_i, ticket_j, float(sim)))
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates

    def cluster_similar_tickets(
        self, eps: float = 0.3, min_samples: int = 2
    ) -> Dict[int, List[Ticket]]:
        if len(self.tickets) < 2:
            return {}
        dist_matrix = 1 - cosine_similarity(self.embeddings_matrix)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(dist_matrix)

        clusters: Dict[int, List[Ticket]] = {}
        for t, label in zip(self.tickets, labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(t)
        return clusters

    # -----------------------------  persistence  --------------------------- #
    def save_model(self, path: str | Path) -> None:
        """Pickle tickets + params (NOT the transformer model)."""
        state = {
            "tickets": self.tickets,
            "similarity_threshold": self.similarity_threshold,
            "ticket_index": self.ticket_index,
        }
        with Path(path).open("wb") as fh:
            pickle.dump(state, fh)
        logger.info("Detector state saved to {}", path)

    def load_model(self, path: str | Path) -> None:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        self.tickets = state["tickets"]
        self.similarity_threshold = state["similarity_threshold"]
        self.ticket_index = state["ticket_index"]
        self._update_embeddings_matrix()
        logger.info("Detector state loaded from {}", path)

    # -----------------------------  stats  --------------------------------- #
    def get_statistics(self) -> Dict:
        if not self.tickets:
            return {"total_tickets": 0}
        dups = self.find_potential_duplicates()
        clusters = self.cluster_similar_tickets()
        return {
            "total_tickets": len(self.tickets),
            "potential_duplicates": len(dups),
            "clusters_found": len(clusters),
            "tickets_in_clusters": sum(len(c) for c in clusters.values()),
            "similarity_threshold": self.similarity_threshold,
        }