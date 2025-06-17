import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
sys.modules.setdefault('sentence_transformers', types.SimpleNamespace(SentenceTransformer=lambda name: None))
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('pyplot'))
sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

from src.freshservice.freshservice_client import FreshServiceTicketAnalyzer
from src.config import FreshServiceConfig


def test_fetch_tickets_uses_make_request(monkeypatch):
    cfg = FreshServiceConfig()
    cfg.per_page = 2
    analyzer = FreshServiceTicketAnalyzer(cfg)
    calls = []

    def fake_request(endpoint, params=None, max_retries=3):
        calls.append(endpoint)
        if "page=1" in endpoint:
            return {"results": [{"id": 1}, {"id": 2}]}
        return {"results": []}

    monkeypatch.setattr(analyzer, "_make_request", fake_request)
    tickets = analyzer.fetch_tickets_by_group(max_pages=5)

    assert calls[0].startswith("tickets/filter")
    assert len(calls) == 2
    assert tickets == [{"id": 1}, {"id": 2}]
