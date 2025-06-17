from __future__ import annotations

# =========================================================================== #
#  Standard-library imports
# =========================================================================== #
import base64
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Final, List, Optional, Tuple

# =========================================================================== #
#  Third-party imports
# =========================================================================== #
import requests
from dateutil.parser import isoparse
from loguru import logger
from urllib.parse import quote
from sentence_transformers import SentenceTransformer

# =========================================================================== #
#  Project-internal imports
# =========================================================================== #
from ..config import FreshServiceConfig
from ..similarity.ticket_similarity import Ticket, TicketSimilarityDetector
from ..similarity.ticket_trainer import TicketTrainer

# =========================================================================== #
#  Patch: minimal Freshservice client (implements merge workaround)
# =========================================================================== #
FRESHSERVICE_STATUS_CLOSED: Final[int] = 5
USER_AGENT: Final[str] = "semantic-ticket-search/merge-workaround"

log = logging.getLogger(__name__)


class FreshserviceClient:
    """
    Ultra-light wrapper around the Freshservice v2 REST API.
    Only the endpoints required by *Semantic-Ticket-Search* are exposed.
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(self, domain: str, api_key: str, *, timeout: float = 15.0) -> None:
        if not domain.startswith("https://"):
            domain = f"https://{domain}"
        self._base_url: str = domain.rstrip("/") + "/api/v2"
        self._auth: tuple[str, str] = (api_key, "X")  # basic-auth, any pwd
        self._timeout: float = timeout
        log.debug(
            "Initialised FreshserviceClient: base_url=%s, timeout=%ss",
            self._base_url,
            timeout,
        )

    # --------------------------------------------------------------------- #
    #  Public high-level helper – MERGE
    # --------------------------------------------------------------------- #
    def merge_tickets(self, *, winner_id: int, duplicate_id: int) -> None:
        """
        Emulate a "merge" by:
        1. Adding a private note to the *winner* ticket.
        2. Adding a private note to the *duplicate* ticket.
        3. Closing the duplicate ticket (status = 5 by default).
        """
        log.info("Merging Freshservice ticket %s into %s", duplicate_id, winner_id)

        # 1) private note on winner
        self._add_private_note(
            ticket_id=winner_id,
            body=(
                f"Ticket #{duplicate_id} has been automatically merged into this "
                "one by Semantic-Ticket-Search. All future correspondence will "
                "happen here."
            ),
        )

        # 2) private note on duplicate
        self._add_private_note(
            ticket_id=duplicate_id,
            body=(
                f"This ticket is a duplicate of #{winner_id} and was "
                "automatically closed by Semantic-Ticket-Search. "
                "Please refer to the master ticket for all updates."
            ),
        )

        # 3) close duplicate
        self._update_ticket_status(duplicate_id, FRESHSERVICE_STATUS_CLOSED)

    # ------------------------------------------------------------------ #
    #  Internal HTTP helpers
    # ------------------------------------------------------------------ #
    def _add_private_note(self, *, ticket_id: int, body: str) -> None:
        url = f"{self._base_url}/tickets/{ticket_id}/notes"
        payload: Dict[str, Any] = {
            "body": body,
            "private": True,
            "incoming": False,
        }
        log.debug("POST %s -> %s", url, payload)
        resp = requests.post(
            url,
            json=payload,
            auth=self._auth,
            timeout=self._timeout,
            headers={"User-Agent": USER_AGENT},
        )
        _raise_for_status(resp)

    def _update_ticket_status(self, ticket_id: int, status: int) -> None:
        url = f"{self._base_url}/tickets/{ticket_id}"
        payload = {"status": status}
        log.debug("PUT %s -> %s", url, payload)
        resp = requests.put(
            url,
            json=payload,
            auth=self._auth,
            timeout=self._timeout,
            headers={"User-Agent": USER_AGENT},
        )
        _raise_for_status(resp)


# --------------------------------------------------------------------------- #
#  Utility (used by FreshserviceClient)
# --------------------------------------------------------------------------- #
def _raise_for_status(resp: requests.Response) -> None:  # pragma: no cover
    """Re-raise an HTTPError with improved context (URL & response body)."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        snippet = resp.text[:500]
        raise requests.HTTPError(
            f"{exc} – url={resp.url!r} – body={snippet!r}", response=resp
        ) from None


# =========================================================================== #
#  Shared, module-wide constants
# =========================================================================== #
IGNORE_RE: re.Pattern | None = None  # compiled at runtime


@dataclass
class TicketProbability:
    ticket_id: str
    title: str
    description: str
    duplicate_probability: float
    most_similar_ticket_id: Optional[str] = None
    most_similar_ticket_title: Optional[str] = None
    most_similar_ticket_description: Optional[str] = None
    similarity_score: float = 0.0
    potential_duplicates: List[Tuple[str, str, float]] | None = None
    requester_email: Optional[str] = None

    def __post_init__(self):
        if self.potential_duplicates is None:
            self.potential_duplicates = []


# =========================================================================== #
#  Main FreshServiceTicketAnalyzer
# =========================================================================== #
class FreshServiceTicketAnalyzer:
    """
    Wrapper around the Freshservice REST API that also embeds /
    deduplicates tickets via Sentence-Transformers + FAISS.
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        config: FreshServiceConfig,
        detector: TicketSimilarityDetector | None = None,
    ):
        self.config = config
        self.detector = detector or TicketSimilarityDetector()
        self.trainer = TicketTrainer(self.detector)

        # -------------------------------------------------------------- #
        #  Auth + base URL (for low-level list / GET calls)
        # -------------------------------------------------------------- #
        auth = f"{self.config.api_key}:X".encode()
        self.auth_header = f"Basic {base64.b64encode(auth).decode()}"
        self.base_url = f"https://{self.config.domain}/api/v2"

        # -------------------------------------------------------------- #
        #  Freshservice *patch* client for merge workaround
        # -------------------------------------------------------------- #
        self.fs_client = FreshserviceClient(
            domain=self.config.domain, api_key=self.config.api_key
        )

        # -------------------------------------------------------------- #
        #  Rate-limit bookkeeping
        # -------------------------------------------------------------- #
        self.last_request_time = 0.0
        self.min_request_interval = 0.50  # seconds

        # -------------------------------------------------------------- #
        #  Logging sinks
        # -------------------------------------------------------------- #
        self.logger = logger.bind(module=__name__)
        unmerged_path = os.getenv("UNMERGED_LOG", "/data/unmerged_tickets.log")
        if not any(
            getattr(sink, "_name", None) == unmerged_path
            for sink in logger._core.handlers.values()
        ):
            logger.add(
                unmerged_path, level="INFO", enqueue=True, format="{time} {message}"
            )
        self.unmerged_logger = logger.bind(name="unmerged")

    # ------------------------------------------------------------------ #
    #  Internal HTTP GET helper (unchanged)
    # ------------------------------------------------------------------ #
    def _make_request(
        self, endpoint: str, params: Dict | None = None, *, max_retries: int = 3
    ) -> Dict:
        """
        Blocking (requests) GET helper with naive 429 handling.
        Runs in a thread-pool when called from FastAPI.
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": self.auth_header,
            "Content-Type": "application/json",
        }
        retries = 0
        while True:
            # simple client-side rate-limit
            delta = time.time() - self.last_request_time
            if delta < self.min_request_interval:
                time.sleep(self.min_request_interval - delta)
            try:
                resp = requests.get(
                    url, headers=headers, params=params or {}, timeout=30
                )
                self.last_request_time = time.time()
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    retries += 1
                    if retries > max_retries:
                        msg = f"Rate-limit exceeded after {max_retries} retries"
                        self.logger.error(msg)
                        raise RuntimeError(msg)
                    self.logger.warning(
                        "429 received – sleeping 60 s (attempt %d/%d)",
                        retries,
                        max_retries,
                    )
                    time.sleep(60)
                    continue
                # propagate non-200 errors WITHOUT leaking headers / body
                self.logger.error("Freshservice %s returned %s", url, resp.status_code)
                resp.raise_for_status()
            except requests.RequestException as exc:
                self.logger.error("HTTP request failed: {}", exc)
                raise

    # ------------------------------------------------------------------ #
    #  Ticket harvesting
    # ------------------------------------------------------------------ #
    def fetch_tickets_by_group(
        self,
        days_back: int = 90,
        status_filter: List[int] | None = None,
        max_pages: int | None = None,
    ) -> List[Dict]:
        """
        Pull tickets via `/tickets/filter` using the Lucene-style query
        recommended by Freshservice support.
        """
        max_pages = max_pages or self.config.max_pages
        status_filter = status_filter or self.config.status

        # date window
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=days_back)

        status_part = " OR ".join(f"status:{s}" for s in status_filter)
        group_part = " OR ".join(f"group_id:{gid}" for gid in self.config.group_ids)
        raw_query = (
            f"({status_part}) AND ({group_part}) AND "
            f"created_at:>'{from_date.strftime('%Y-%m-%d')}'"
        )
        encoded = quote(raw_query, safe="")
        endpoint_base = f'tickets/filter?query="{encoded}"'

        all_tickets: List[Dict] = []
        page = 1
        per_page = self.config.per_page
        while page <= max_pages:
            endpoint = f"{endpoint_base}&page={page}&per_page={per_page}"
            data = self._make_request(endpoint)
            tickets = data.get("tickets") or data.get("results") or []
            if not tickets:
                break
            all_tickets.extend(tickets)
            self.logger.info("Fetched {} tickets on page {}", len(tickets), page)
            if len(tickets) < per_page:
                break
            page += 1
        self.logger.info("Total tickets fetched: {}", len(all_tickets))
        return all_tickets

    # ------------------------------------------------------------------ #
    #  JSON → Ticket dataclass
    # ------------------------------------------------------------------ #
    def convert_freshservice_to_tickets(self, fs_tickets: List[Dict]) -> List[Ticket]:
        global IGNORE_RE
        if IGNORE_RE is None:
            pattern = "|".join(map(re.escape, self.config.ignore_subject_phrases))
            IGNORE_RE = re.compile(pattern, re.I)
        cleaned: List[Ticket] = []
        skipped = 0
        for raw in fs_tickets:
            subject = raw.get("subject", "") or ""
            if IGNORE_RE.search(subject):
                skipped += 1
                continue
            cleaned.append(
                Ticket(
                    id=str(raw["id"]),
                    title=subject,
                    description=re.sub("<[^>]+>", "", raw.get("description", "")),
                    requester_email=raw.get("email") or None,
                    created_at=isoparse(raw["created_at"])
                    if raw.get("created_at")
                    else None,
                    category=raw.get("category"),
                    priority=raw.get("priority"),
                    department_id=str(raw.get("department_id")) if raw.get("department_id") is not None else None,
                )
            )
        if skipped:
            self.logger.info("Skipped {} noisy tickets (subject filter)", skipped)
        return cleaned

    # ------------------------------------------------------------------ #
    #  Duplicate-analysis pipeline (offline / jobs)
    # ------------------------------------------------------------------ #
    def analyze_tickets_for_duplicates(
        self,
        tickets: List[Ticket] | None = None,
        probability_threshold: float = 0.3,
    ) -> List[TicketProbability]:
        """
        Heavy-weight method used by the daily job, not by the webhook path.
        """
        if tickets is None:
            tickets = self.convert_freshservice_to_tickets(
                self.fetch_tickets_by_group()
            )
        if not tickets:
            self.logger.warning("No tickets to analyse")
            return []

        existing = {t.id for t in self.detector.tickets}
        new_tickets = [t for t in tickets if t.id not in existing]
        if new_tickets:
            self.detector.add_tickets_batch(new_tickets)

        out: List[TicketProbability] = []
        for ticket in tickets:
            similar = self.detector.find_similar_tickets(ticket, top_k=10)
            prob = self._calculate_duplicate_probability(ticket, similar)
            
            # Filter for potential duplicates (exclude current ticket and ensure same department)
            potential = []
            for st, score in similar:
                if st.id == ticket.id:
                    continue
                if score < probability_threshold:
                    continue
                # Department filtering is already handled in find_similar_tickets and _calculate_duplicate_probability
                potential.append((st.id, st.title, score))
            
            # Filter out the current ticket from similar results before selecting the most similar
            filtered_similar = [(st, score) for st, score in similar if st.id != ticket.id]
            most = filtered_similar[0] if filtered_similar else (None, 0.0)
            out.append(
                TicketProbability(
                    ticket_id=ticket.id,
                    title=ticket.title,
                    description=(
                        ticket.description[:200] + "..."
                        if len(ticket.description) > 200
                        else ticket.description
                    ),
                    duplicate_probability=prob,
                    most_similar_ticket_id=most[0].id if most and most[0] else None,
                    most_similar_ticket_title=most[0].title if most and most[0] else None,
                    most_similar_ticket_description=(
                        (most[0].description[:200] + "..." if len(most[0].description) > 200 else most[0].description)
                        if most and most[0] else None
                    ),
                    similarity_score=most[1] if most else 0.0,
                    potential_duplicates=potential,
                    requester_email=ticket.requester_email,
                )
            )
        out.sort(key=lambda tp: tp.duplicate_probability, reverse=True)
        return out

    # ------------------------------------------------------------------ #
    #  Probability model (cosine + heuristics)
    # ------------------------------------------------------------------ #
    def _calculate_duplicate_probability(
        self, ticket: Ticket, similar: List[Tuple[Ticket, float]]
    ) -> float:
        # Filter out the current ticket and tickets from different departments
        filtered_similar = []
        for t, s in similar:
            if t.id == ticket.id:
                continue
            # Only consider tickets from the same department
            if (ticket.department_id is not None and 
                t.department_id is not None and 
                ticket.department_id != t.department_id):
                continue
            filtered_similar.append((t, s))
        
        if not filtered_similar:
            return 0.0
            
        best_ticket, cos_sim = max(filtered_similar, key=lambda ts: ts[1])

        # requester email bonus
        email_bonus = (
            1.0
            if ticket.requester_email
            and best_ticket.requester_email
            and ticket.requester_email.lower() == best_ticket.requester_email.lower()
            else 0.0
        )

        # department match bonus - already guaranteed to be same department due to filtering above
        dept_bonus = 1.0 if (ticket.department_id is not None and 
                           best_ticket.department_id is not None and 
                           ticket.department_id == best_ticket.department_id) else 0.0

        sigmoid = lambda x: 1 / (1 + math.exp(-10 * (x - 0.6)))
        sim_prob = sigmoid(cos_sim)
        probability = 0.85 * sim_prob + 0.10 * email_bonus + 0.05 * dept_bonus
        return max(0.0, min(1.0, probability))

    # ------------------------------------------------------------------ #
    #  Simple merge helpers – used by webhook path
    # ------------------------------------------------------------------ #
    def fetch_ticket_by_id(self, ticket_id: str) -> Optional[Ticket]:
        try:
            data = self._make_request(f"tickets/{ticket_id}")
        except Exception:
            return None
        if not data or "ticket" not in data:
            self.logger.error("Ticket {} not found", ticket_id)
            return None
        tickets = self.convert_freshservice_to_tickets([data["ticket"]])
        return tickets[0] if tickets else None

    # =======  UPDATED implementation – now uses FreshserviceClient  =======
    def merge_ticket(self, source_ticket_id: str, target_ticket_id: str) -> bool:
        """
        Replace the old, non-existent `/merge` endpoint with the workaround
        provided by `FreshserviceClient.merge_tickets`.
        """
        try:
            self.fs_client.merge_tickets(
                winner_id=int(target_ticket_id), duplicate_id=int(source_ticket_id)
            )
            self.logger.info("Merged ticket {} → {}", source_ticket_id, target_ticket_id)
            return True
        except Exception as exc:
            self.logger.error(
                "Merge {}→{} failed: {}", source_ticket_id, target_ticket_id, exc
            )
            return False

    def process_ticket_by_id(self, ticket_id: str, threshold: float = 0.9) -> bool:
        """
        Called by the webhook service (runs in a thread-pool).
        1. Fetch ticket
        2. Compare against ANN detector already loaded by serve process
        3. Merge if probability ≥ *threshold*
        """
        ticket = self.fetch_ticket_by_id(ticket_id)
        if ticket is None:
            return False

        # detector should already contain the corpus (loaded by serve)
        if ticket.id not in {t.id for t in self.detector.tickets}:
            self.detector.add_tickets_batch([ticket])

        similar = self.detector.find_similar_tickets(ticket, top_k=5)
        # Note: find_similar_tickets and _calculate_duplicate_probability now handle department filtering internally
        probability = self._calculate_duplicate_probability(ticket, similar)

        # Filter for merging (exclude current ticket)
        filtered = [(t, s) for t, s in similar if t.id != ticket.id]
        
        if filtered and probability >= threshold:
            return self.merge_ticket(ticket.id, filtered[0][0].id)

        self.unmerged_logger.info("{} probability={:.2f}", ticket.id, probability)
        return False