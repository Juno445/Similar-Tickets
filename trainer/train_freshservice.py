"""
trainer.train_freshservice
==========================
One-shot batch job that
1. Pulls tickets from Freshservice (using the same helper code as the API).
2. Generates _new_ embeddings only for unseen tickets.
3. Builds / refreshes a FAISS inner-product (cosine) index.
4. Writes two artefacts into INDEX_PATH (default /shared):
      ├─ tickets.index   (FAISS binary, float32 vectors)
      └─ meta.pkl        (pickle: List[Ticket] + threshold + model name)
The serve container mounts the same volume _read-only_ and only
performs ANN search.  Concurrency guarantees:
  • Entire snapshot is written atomically via temporary files + rename.
  • A running serve process that already memory-mapped the files
    keeps working; the next reload cycle will pick up the new snapshot.
"""
from __future__ import annotations
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import List
import faiss
import json
from dateutil.parser import isoparse
from loguru import logger
from src.config import FreshServiceConfig
from src.similarity.ticket_similarity import TicketSimilarityDetector, Ticket
from src.freshservice.freshservice_client import FreshServiceTicketAnalyzer

# --------------------------------------------------------------------------- #
#  Environment & defaults
# --------------------------------------------------------------------------- #
INDEX_PATH           = Path(os.getenv("INDEX_PATH", "/shared"))
DAYS_BACK            = int(os.getenv("DAYS_BACK", "60"))
MODEL_NAME           = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SIM_THRESHOLD        = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TMP_SUFFIX           = ".tmp"           # temp files before atomic rename

INDEX_PATH.mkdir(parents=True, exist_ok=True)
INDEX_FILE = INDEX_PATH / "tickets.index"
META_FILE  = INDEX_PATH / "meta.pkl"

def _load_existing_metadata() -> list[Ticket] | None:
    """Return the list of *Ticket* objects from the most recent snapshot.

    Compatibility notes:
    • `meta.pkl` (legacy pickle) – deprecated but still supported to avoid a full
      re-encode right after upgrading.
    • `current/meta.json` (new format) – preferred.  This file is written by
      `_save_snapshot` starting with the new implementation.
    """

    # --- New JSON format (preferred) ------------------------------------
    current_meta_json = INDEX_PATH / "current" / "meta.json"
    if current_meta_json.exists():
        try:
            data = json.loads(current_meta_json.read_text())
            tickets_raw = data.get("tickets", [])

            def _parse(t: dict) -> Ticket:
                # `Ticket` constructor expects certain kwargs – pass through
                created_at_str = t.get("created_at")
                created_at = isoparse(created_at_str) if created_at_str else None
                
                return Ticket(
                    id=str(t["id"]),
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    requester_email=t.get("requester_email"),
                    created_at=created_at,
                    category=t.get("category"),
                    priority=t.get("priority"),
                    department_id=t.get("department_id"),
                )

            logger.info("Loaded {} existing tickets from JSON snapshot", len(tickets_raw))
            return [_parse(t) for t in tickets_raw]
        except Exception as exc:
            logger.warning("Failed to parse JSON metadata: {} – falling back to pickle", exc)

    # --- Legacy pickle fallback ----------------------------------------
    if META_FILE.exists():
        logger.info("Existing legacy metadata found (pickle) – will load for incremental update")
        try:
            with META_FILE.open("rb") as fh:
                meta = pickle.load(fh)
            return meta.get("tickets", [])
        except Exception as exc:
            logger.warning("Failed to load legacy pickle metadata: {}", exc)
    return None

def _save_snapshot(index: faiss.Index, tickets: List[Ticket], threshold: float):
    now = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%SZ')
    snapshot_dir = INDEX_PATH / "snapshots" / now
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Write FAISS index and meta _in the subdir_
    vec_file = snapshot_dir / "vectors.faiss"
    meta_file = snapshot_dir / "meta.json"

    faiss.write_index(index, str(vec_file))
    def _ticket_to_dict(t: Ticket) -> dict:
        data = {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "requester_email": t.requester_email,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "category": t.category,
            "priority": t.priority,
            "department_id": t.department_id,
        }
        return {k: v for k, v in data.items() if v is not None}

    meta = {
        "tickets": [_ticket_to_dict(t) for t in tickets],
        "similarity_threshold": threshold,
        "updated_at": now,
        "model": MODEL_NAME,
        "dim": index.d,
    }
    meta_file.write_text(json.dumps(meta, indent=2))

    # Atomically symlink /shared/current → this new snapshot
    current_link = INDEX_PATH / "current"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    os.symlink(snapshot_dir.resolve(), current_link)

    logger.success(
        "Snapshot written in {} ({} vectors)", snapshot_dir, index.ntotal
    )
def build_index() -> None:
    """End-to-end orchestration of the training job."""
    cfg = FreshServiceConfig()
    logger.info("Trainer started for domain={}  groups={}", cfg.domain, cfg.group_ids)
    # ------------------------------------------------------------------
    # 1. Instantiate detector (in-memory) and analyzer (API access)
    # ------------------------------------------------------------------
    detector = TicketSimilarityDetector(model_name=MODEL_NAME,
                                        similarity_threshold=SIM_THRESHOLD)
    analyzer = FreshServiceTicketAnalyzer(cfg, detector)
    # ------------------------------------------------------------------
    # 2.  Fetch tickets (optionally incremental)
    # ------------------------------------------------------------------
    fresh_tickets_raw = analyzer.fetch_tickets_by_group(days_back=DAYS_BACK)
    fresh_tickets     = analyzer.convert_freshservice_to_tickets(fresh_tickets_raw)
    # Merge with existing snapshot so we don't re-encode unchanged tickets
    existing = _load_existing_metadata() or []
    existing_ids = {t.id for t in existing}
    new_tickets  = [t for t in fresh_tickets if t.id not in existing_ids]
    detector.add_tickets_batch(existing + new_tickets)
    logger.info("Corpus size: {} tickets  (new: {})", len(detector.tickets), len(new_tickets))
    # ------------------------------------------------------------------
    # 3.  Build FAISS inner-product index  (cosine similarity)
    # ------------------------------------------------------------------
    emb_matrix = detector.embeddings_matrix.astype("float32")   # faiss wants f32
    dimension  = emb_matrix.shape[1]
    index      = faiss.IndexFlatIP(dimension)
    index.add(emb_matrix)
    # ------------------------------------------------------------------
    # 4. Save to shared volume
    # ------------------------------------------------------------------
    _save_snapshot(index, detector.tickets, detector.similarity_threshold)
    logger.success("Trainer finished OK")

# ---------------------------------------------------------------------- #
#  CLI entry-point
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        build_index()
    except Exception as exc:
        logger.exception("Trainer failed: {}", exc)
        raise SystemExit(1)