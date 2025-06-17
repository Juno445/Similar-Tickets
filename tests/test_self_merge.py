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
from src.similarity.ticket_similarity import Ticket


def test_process_ticket_ignores_self_similarity(monkeypatch):
    cfg = FreshServiceConfig()
    analyzer = FreshServiceTicketAnalyzer(cfg)

    ticket = Ticket(id="1", title="A", description="Desc", department_id="dept1")
    other = Ticket(id="2", title="B", description="Other", department_id="dept1")

    analyzer.detector = MagicMock()
    analyzer.detector.tickets = [ticket, other]
    analyzer.detector.find_similar_tickets.return_value = [(ticket, 1.0), (other, 0.95)]

    monkeypatch.setattr(analyzer, "fetch_ticket_by_id", lambda tid: ticket)
    merge = MagicMock(return_value=True)
    monkeypatch.setattr(analyzer, "merge_ticket", merge)

    merged = analyzer.process_ticket_by_id("1", threshold=0.5)

    assert merged is True
    merge.assert_called_once_with("1", "2")


def test_process_ticket_ignores_different_departments(monkeypatch):
    cfg = FreshServiceConfig()
    analyzer = FreshServiceTicketAnalyzer(cfg)

    # Tickets from different departments
    ticket = Ticket(id="1", title="A", description="Desc", department_id="dept1")
    other = Ticket(id="2", title="B", description="Other", department_id="dept2")

    analyzer.detector = MagicMock()
    analyzer.detector.tickets = [ticket, other]
    # find_similar_tickets would return the other ticket, but it should be filtered out
    analyzer.detector.find_similar_tickets.return_value = [(ticket, 1.0), (other, 0.95)]

    monkeypatch.setattr(analyzer, "fetch_ticket_by_id", lambda tid: ticket)
    merge = MagicMock(return_value=True)
    monkeypatch.setattr(analyzer, "merge_ticket", merge)

    merged = analyzer.process_ticket_by_id("1", threshold=0.5)

    # Should not merge since tickets are from different departments
    assert merged is False
    merge.assert_not_called()


