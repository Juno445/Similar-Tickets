import os
import sys
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import types

sys.modules.setdefault('sentence_transformers', types.SimpleNamespace(SentenceTransformer=lambda name: None))
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('pyplot'))
sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

import src.webhook_server as server
from src.webhook_server import app, get_analyzer

analyzer = MagicMock()
analyzer.detector = MagicMock()

client = TestClient(app)

def setup_mocks(probability: float, monkeypatch):
    monkeypatch.setattr(server, "get_analyzer", lambda: analyzer)
    def fake_process(tid):
        if probability >= 0.9:
            analyzer.merge_ticket(tid, "2")
            return True
        analyzer.unmerged_logger.info(tid)
        return False

    monkeypatch.setattr(analyzer, "process_ticket_by_id", fake_process)


def test_webhook_merges(monkeypatch):
    setup_mocks(0.95, monkeypatch)
    merge = MagicMock(return_value=True)
    monkeypatch.setattr(analyzer, "merge_ticket", merge)
    resp = client.post("/webhook", json={"ticket_id": 1})
    assert resp.status_code == 200
    assert resp.json() == {"merged": True}
    merge.assert_called_once()


def test_webhook_skips(monkeypatch):
    setup_mocks(0.5, monkeypatch)
    merge = MagicMock(return_value=True)
    log = MagicMock()
    monkeypatch.setattr(analyzer, "merge_ticket", merge)
    monkeypatch.setattr(analyzer, "unmerged_logger", log)
    resp = client.post("/webhook", json={"ticket_id": 2})
    assert resp.status_code == 200
    assert resp.json() == {"merged": False}
    merge.assert_not_called()
    log.info.assert_called_once()

