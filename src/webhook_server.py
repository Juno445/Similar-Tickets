from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import hmac, hashlib, json

from .ann_index import get_ann
from .config import FreshServiceConfig
from .freshservice.freshservice_client import FreshServiceTicketAnalyzer
from .logging_setup import configure_logging
from .monitoring import MetricsMiddleware, metrics_response
from .similarity.ticket_similarity import _create_sentence_transformer

configure_logging()
logger = logging.getLogger(__name__)

RELOAD_TOKEN = os.getenv("RELOAD_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")  # optional shared secret for webhook


# --------------------------------------------------------------------------- #
#  Lazy singletons
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    model_name: str = get_ann().meta["model"]
    logger.info("Loading SentenceTransformer model {}", model_name)
    return _create_sentence_transformer(model_name)


@lru_cache(maxsize=1)
def get_analyzer() -> FreshServiceTicketAnalyzer:
    return FreshServiceTicketAnalyzer(FreshServiceConfig())


# --------------------------------------------------------------------------- #
#  Active-learning trainer singleton (re-uses analyzer's detector)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def get_trainer() -> "TicketTrainer":
    analyzer = get_analyzer()
    # lazily import to avoid circular deps / heavy libs on cold start
    from .similarity.ticket_trainer import TicketTrainer  # noqa: WPS433 – runtime import

    trainer: TicketTrainer = analyzer.trainer  # reuse instance created in analyzer

    # ensure the detector already contains all tickets from the ANN snapshot
    if not trainer.detector.tickets:
        ann = get_ann()
        trainer.detector.add_tickets_batch(list(ann.tickets))

    return trainer


# --------------------------------------------------------------------------- #
#  FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI()
app.add_middleware(MetricsMiddleware)


class WebhookPayload(BaseModel):
    ticket_id: int


def _process_sync(ticket_id: str) -> bool:
    analyzer = get_analyzer()

    # Fetch ticket
    ticket = analyzer.fetch_ticket_by_id(ticket_id)
    if ticket is None:
        return False

    # Embed (unit-norm)
    embedder = get_embedder()
    vec = embedder.encode([ticket.get_combined_text()],
                          normalize_embeddings=True)[0]

    # ANN search
    ann = get_ann()
    top = ann.top_k(vec, k=5)
    # remove the query ticket from the result list and filter by department
    filtered = []
    for t, s in top:
        if t.id == ticket.id:
            continue
        # Only consider tickets from the same department
        if (ticket.department_id is not None and 
            t.department_id is not None and 
            ticket.department_id != t.department_id):
            continue
        filtered.append((t, s))
    
    probability = analyzer._calculate_duplicate_probability(ticket, filtered)

    logger.info(
        "ticket={} prob={:.3f} best={:.3f}",
        ticket_id,
        probability,
        filtered[0][1] if filtered else 0.0,
    )

    # merge?
    if filtered and probability >= ann.meta["similarity_threshold"]:
        return analyzer.merge_ticket(ticket_id, filtered[0][0].id)

    logging.getLogger("unmerged").info("{} prob={:.2f}", ticket_id, probability)
    return False


# --------------------------------------------------------------------------- #
#  Helpers – signature validation
# --------------------------------------------------------------------------- #

def _verify_signature(body: bytes, signature: str | None) -> None:
    """Raise HTTP 401 if *WEBHOOK_SECRET* is set and signature is invalid."""
    if WEBHOOK_SECRET is None:  # auth disabled
        return
    if not signature:
        raise HTTPException(status_code=401, detail="missing signature header")
    expected = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="invalid signature")


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_signature: str | None = Header(None, alias="X-Webhook-Signature"),
):
    """Freshservice webhook endpoint with optional HMAC authentication."""
    body = await request.body()
    _verify_signature(body, x_webhook_signature)

    try:
        payload = WebhookPayload.model_validate_json(body)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid payload")

    merged = await run_in_threadpool(_process_sync, str(payload.ticket_id))
    return {"merged": merged}


@app.get("/healthz")
async def healthz():
    try:
        ann = get_ann()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="snapshot missing")
    return {"status": "ok", "vectors": ann.index.ntotal}


# --------------------------------------------------------------------------- #
#  Hot-reload endpoint
# --------------------------------------------------------------------------- #
def _auth(x_reload_token: str = Header(..., alias="X-Reload-Token")) -> None:
    if RELOAD_TOKEN is None or x_reload_token != RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")


@app.post("/reloadIndex")
async def reload_index(_: None = Depends(_auth)):
    await run_in_threadpool(get_ann, True)  # force_reload=True
    await run_in_threadpool(get_embedder.cache_clear)
    return {"reloaded": True}


@app.get("/metrics")
async def metrics():
    return metrics_response()


# --------------------------------------------------------------------------- #
#  Review UI – human-in-the-loop feedback console
# --------------------------------------------------------------------------- #

def _build_review_html() -> str:
    """Return a small Tailwind + Alpine.js single-page UI."""

    tailwind_cdn = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
    alpine_cdn = "https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
        <title>Ticket Merge Review</title>
        <link href=\"{tailwind_cdn}\" rel=\"stylesheet\">
        <script src=\"{alpine_cdn}\" defer></script>
    </head>
    <body class=\"bg-gray-100 min-h-screen flex flex-col items-center p-6\" x-data=\"ticketApp()\">
        <h1 class=\"text-3xl font-semibold mb-4\">Potential Ticket Merges</h1>

        <template x-if=\"candidates.length === 0 && !loading\">
            <p class=\"text-gray-600\">No pending candidates &mdash; 🎉</p>
        </template>

        <template x-for=\"cand in candidates\" :key=\"`${{cand.ticket1_id}}-${{cand.ticket2_id}}`\">
            <div class=\"bg-white shadow-md rounded-md p-4 mb-4 w-full max-w-4xl\">
                <div class=\"flex justify-between items-center\">
                    <div>
                        <h2 class=\"text-lg font-medium\" x-text=\"`#{{cand.ticket1_id}}: ${{cand.ticket1_title}}`\"></h2>
                        <p class=\"text-sm text-gray-500\" x-text=\"`→ Similar to #{{cand.ticket2_id}} (${{(cand.similarity*100).toFixed(1)}}%)`\"></p>
                    </div>
                    <div class=\"space-x-2\">
                        <a :href=\"cand.ticket1_url\" target=\"_blank\" class=\"text-blue-600 underline text-sm\">Open</a>
                        <button @click=\"label(cand, true)\" class=\"px-3 py-1 bg-green-500 text-white rounded-md text-sm\">Approve</button>
                        <button @click=\"label(cand, false)\" class=\"px-3 py-1 bg-red-500 text-white rounded-md text-sm\">Deny</button>
                    </div>
                </div>
            </div>
        </template>

        <template x-if=\"loading\"><p class=\"text-gray-500\">Loading…</p></template>

        <script>
        function ticketApp() {{
            return {{
                candidates: [],
                loading: true,
                init() {{
                    this.fetchCandidates();
                }},
                async fetchCandidates() {{
                    this.loading = true;
                    try {{
                        const resp = await fetch('/candidates');
                        this.candidates = await resp.json();
                    }} finally {{
                        this.loading = false;
                    }}
                }},
                async label(cand, isDup) {{
                    await fetch('/label', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            ticket1_id: cand.ticket1_id,
                            ticket2_id: cand.ticket2_id,
                            is_duplicate: isDup
                        }})
                    }});
                    this.candidates = this.candidates.filter(c => c !== cand);
                    if (this.candidates.length === 0) {{
                        await this.fetchCandidates();
                    }}
                }}
            }}
        }}
        document.addEventListener('alpine:init', () => {{ Alpine.data('ticketApp', ticketApp) }});
        </script>
    </body>
    </html>
    """


# ---------------  Routes  --------------- #

@app.get("/review", response_class=HTMLResponse)
async def review_page():
    return _build_review_html()


@app.get("/candidates", response_class=JSONResponse)
async def candidate_pairs(max_pairs: int = 25):
    trainer = get_trainer()
    pairs = trainer.get_unlabeled_pairs(min_similarity=0.65, max_pairs=max_pairs)
    domain = get_analyzer().config.domain

    filtered = [p for p in pairs if 0.65 <= p[2] < 0.9]
    out = [
        {
            "ticket1_id": t1.id,
            "ticket1_title": t1.title,
            "ticket1_url": f"https://{domain}/helpdesk/tickets/{t1.id}",
            "ticket2_id": t2.id,
            "ticket2_title": t2.title,
            "similarity": sim,
        }
        for t1, t2, sim in filtered
    ]
    return out


class LabelPayload(BaseModel):
    ticket1_id: str
    ticket2_id: str
    is_duplicate: bool
    confidence: float | None = 1.0
    labeled_by: str | None = "reviewer"


@app.post("/label")
async def label_pair(payload: LabelPayload):
    trainer = get_trainer()
    ann = get_ann()
    id_map = {t.id: t for t in ann.tickets}

    t1 = id_map.get(payload.ticket1_id) or get_analyzer().fetch_ticket_by_id(payload.ticket1_id)
    t2 = id_map.get(payload.ticket2_id) or get_analyzer().fetch_ticket_by_id(payload.ticket2_id)

    if not (t1 and t2):
        raise HTTPException(status_code=404, detail="tickets not found")

    trainer.label_pair(
        ticket1=t1,
        ticket2=t2,
        is_duplicate=payload.is_duplicate,
        confidence=payload.confidence or 1.0,
        labeled_by=payload.labeled_by or "reviewer",
    )

    return {"labelled": True}


if __name__ == "__main__":
    uvicorn.run("src.webhook_server:app", host="0.0.0.0", port=8000, reload=False)