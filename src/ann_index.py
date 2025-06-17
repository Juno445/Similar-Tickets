"""
Read-only FAISS + metadata loader with hot-reload capability.
Snapshot layout created by the trainer job:

/shared/
└── current -> snapshots/2024-06-01T12-00-00Z   (symlink, flipped atomically)
    ├─ vectors.faiss
    └─ meta.json    {"model": "...", "dim": 384, "similarity_threshold": 0.85}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Tuple
from dateutil.parser import isoparse

import faiss
import numpy as np
from loguru import logger

from sentence_transformers import SentenceTransformer

from .similarity.ticket_similarity import Ticket

INDEX_ROOT   = Path(os.getenv("INDEX_PATH", "/shared"))
CURRENT_LINK = INDEX_ROOT / "current"
VEC_FILE     = CURRENT_LINK / "vectors.faiss"
META_FILE    = CURRENT_LINK / "meta.json"

# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #
def _mtime() -> float:
    return max(VEC_FILE.stat().st_mtime, META_FILE.stat().st_mtime)


def _load_snapshot() -> tuple[faiss.Index, dict, List[Ticket]]:
    t0 = time.perf_counter()
    if not (VEC_FILE.exists() and META_FILE.exists()):
        raise FileNotFoundError("snapshot missing – trainer must run first")

    index = faiss.read_index(str(VEC_FILE))
    meta = json.loads(META_FILE.read_text())

    def _parse_ticket(t: dict | Ticket) -> Ticket:
        if isinstance(t, Ticket):
            if isinstance(t.created_at, str):
                try:
                    t.created_at = isoparse(t.created_at)
                except Exception:
                    t.created_at = None
            return t
        if "created_at" in t and isinstance(t["created_at"], str):
            try:
                t["created_at"] = isoparse(t["created_at"])
            except Exception:
                t["created_at"] = None
        # Handle backward compatibility for tickets without department_id
        if "department_id" not in t:
            t["department_id"] = None
        return Ticket(**t)

    tickets = [_parse_ticket(t) for t in meta["tickets"]]

    logger.success(
        "ANN snapshot loaded ({} vectors) in {:.2f}s",
        index.ntotal,
        time.perf_counter() - t0,
    )
    return index, meta, tickets


# --------------------------------------------------------------------------- #
#  Public singleton with hot-reload
# --------------------------------------------------------------------------- #
_lock           = Lock()
_instance       = None         # type: AnnIndex | None
_snapshot_mtime = None         # type: float | None


class AnnIndex:
    def __init__(self, index: faiss.Index, meta: dict, tickets: List[Ticket]):
        self.index   = index
        self.meta    = meta
        self.tickets = tickets
        self.dim     = index.d

    # read-only search
    def top_k(self, vec: np.ndarray, k: int = 5) -> List[Tuple[Ticket, float]]:
        if vec.dtype != np.float32:
            vec = vec.astype("float32")
        D, I = self.index.search(vec[None, :], k)
        sims, idxs = D[0], I[0]
        return [
            (self.tickets[idx], float(sim))
            for sim, idx in zip(sims, idxs)
            if idx >= 0
        ]


def get_ann(force_reload: bool = False) -> AnnIndex:
    global _instance, _snapshot_mtime
    with _lock:
        current = _mtime()
        if (
            _instance is None
            or force_reload
            or _snapshot_mtime is None
            or current > _snapshot_mtime
        ):
            faiss_idx, meta, tickets = _load_snapshot()
            _instance = AnnIndex(faiss_idx, meta, tickets)
            _snapshot_mtime = current
        return _instance