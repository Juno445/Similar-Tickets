"""
ticket_trainer.py
=================

Human-in-the-loop active-learning module.

• Collects human labels of "duplicate / not-duplicate" pairs.
• Persists those labels in JSON; can be exported to CSV.
• Evaluates the current cosine-similarity threshold and, if a
  significantly better one is found, updates the attached
  TicketSimilarityDetector instance.

The class is meant to run **offline** (CLI or Jupyter) – never inside the
FastAPI request path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .ticket_similarity import Ticket, TicketSimilarityDetector


# --------------------------------------------------------------------------- #
#  Data structure for a human-labeled pair
# --------------------------------------------------------------------------- #
@dataclass
class LabeledPair:
    ticket1_id: str
    ticket2_id: str
    is_duplicate: bool
    confidence: float            # 0-1 number entered by labeler
    labeled_by: str
    labeled_at: str
    similarity_score: float
    notes: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Main trainer
# --------------------------------------------------------------------------- #
class TicketTrainer:
    def __init__(
        self,
        detector: TicketSimilarityDetector,
        labels_file: str = "labeled_data.json",
    ) -> None:
        self.detector = detector
        self.labels_file = Path(labels_file)
        self.labeled_pairs: List[LabeledPair] = []
        self.performance_history: List[Dict] = []
        self._load_labels()

    # -------------------------  JSON persistence  -------------------------- #
    def _load_labels(self) -> None:
        if self.labels_file.exists():
            self.labeled_pairs = [
                LabeledPair(**obj)
                for obj in json.loads(self.labels_file.read_text())
            ]
            logger.info("Loaded {} labeled pairs", len(self.labeled_pairs))

    def _save_labels(self) -> None:
        data = [asdict(lp) for lp in self.labeled_pairs]
        self.labels_file.write_text(json.dumps(data, indent=2))
        logger.info("Saved {} labeled pairs to {}", len(self.labeled_pairs), self.labels_file)

    # -------------------------  active-learning API  ----------------------- #
    def get_unlabeled_pairs(
        self, min_similarity: float = 0.5, max_pairs: int = 50
    ) -> List[Tuple[Ticket, Ticket, float]]:
        """
        Return at most *max_pairs* ticket pairs above *min_similarity*
        which have not been labeled yet.
        """
        all_pairs = self.detector.find_potential_duplicates(min_similarity)
        labeled = {
            (p.ticket1_id, p.ticket2_id) for p in self.labeled_pairs
        } | {
            (p.ticket2_id, p.ticket1_id) for p in self.labeled_pairs
        }
        result: list[tuple[Ticket, Ticket, float]] = []
        for t1, t2, sim in all_pairs:
            if (t1.id, t2.id) not in labeled:
                result.append((t1, t2, sim))
            if len(result) >= max_pairs:
                break
        return result

    def label_pair(
        self,
        ticket1: Ticket,
        ticket2: Ticket,
        is_duplicate: bool,
        confidence: float,
        labeled_by: str,
        notes: str | None = None,
    ) -> None:
        """
        Record a human judgment for a ticket pair.
        """
        sim = float(
            np.dot(ticket1.embedding, ticket2.embedding)
        )  # embeddings already unit-norm
        pair = LabeledPair(
            ticket1_id=ticket1.id,
            ticket2_id=ticket2.id,
            is_duplicate=is_duplicate,
            confidence=confidence,
            labeled_by=labeled_by,
            labeled_at=datetime.utcnow().isoformat(),
            similarity_score=sim,
            notes=notes,
        )
        self.labeled_pairs.append(pair)
        self._save_labels()

    # -------------------------  threshold tuning  -------------------------- #
    def evaluate_model(self, test_size: float = 0.2) -> Dict:
        """
        Split the labeled data, search for the best similarity threshold on
        the train split, evaluate on the test split.  If the new threshold
        improves accuracy by ≥5 %, it is written back to the detector.
        """
        if len(self.labeled_pairs) < 10:
            logger.warning("Need ≥10 labeled pairs for evaluation")
            return {}

        similarities = [p.similarity_score for p in self.labeled_pairs]
        labels = [p.is_duplicate for p in self.labeled_pairs]

        X_tr, X_te, y_tr, y_te = train_test_split(
            similarities, labels, test_size=test_size, stratify=labels, random_state=42
        )

        thresholds = np.linspace(0.3, 1.0, 50)
        best_f1 = -1.0
        best_thr = self.detector.similarity_threshold

        for thr in thresholds:
            preds = [s >= thr for s in X_tr]
            if len(set(preds)) < 2:
                continue                     # skip degenerate threshold
            _, _, f1, _ = precision_recall_fscore_support(
                y_tr, preds, average="binary"
            )
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        # evaluate on test
        test_preds = [s >= best_thr for s in X_te]
        acc = accuracy_score(y_te, test_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, test_preds, average="binary"
        )

        # compare with current threshold
        current_preds = [s >= self.detector.similarity_threshold for s in X_te]
        current_acc = accuracy_score(y_te, current_preds)

        metrics = dict(
            optimal_threshold=best_thr,
            test_accuracy=acc,
            test_precision=prec,
            test_recall=rec,
            test_f1=f1,
            current_threshold=self.detector.similarity_threshold,
            current_accuracy=current_acc,
            labelled=len(self.labeled_pairs),
            evaluated_at=datetime.utcnow().isoformat(),
        )
        self.performance_history.append(metrics)

        # update detector if improvement ≥5 %
        if acc >= current_acc + 0.05:
            logger.success(
                "New threshold {:.3f} improves accuracy from {:.3f} → {:.3f}",
                best_thr,
                current_acc,
                acc,
            )
            self.detector.similarity_threshold = best_thr
        else:
            logger.info("No significant improvement ({:.3f} → {:.3f})", current_acc, acc)

        return metrics

    # -------------------------  reporting helpers  ------------------------- #
    def export_labeled_data(self, csv_path: str) -> None:
        if not self.labeled_pairs:
            logger.warning("No labeled data to export")
            return
        pd.DataFrame([asdict(p) for p in self.labeled_pairs]).to_csv(csv_path, index=False)
        logger.success("Labeled data exported → {}", csv_path)

    def plot_performance_history(self) -> None:
        if not self.performance_history:
            print("No evaluations yet.")
            return
        df = pd.DataFrame(self.performance_history)
        fig = df[["test_accuracy", "test_f1"]].plot(marker="o").get_figure()
        fig.suptitle("Model performance history")
        plt.show()