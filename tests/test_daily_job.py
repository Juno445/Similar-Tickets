from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
sys.modules.setdefault('sentence_transformers', types.SimpleNamespace(SentenceTransformer=lambda name: None))
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('pyplot'))
sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

from src.freshservice.freshservice_client import FreshServiceTicketAnalyzer, FreshServiceConfig


def test_daily_duplicate_check(tmp_path, monkeypatch):
    cfg = FreshServiceConfig()
    analyzer = FreshServiceTicketAnalyzer(cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analyzer, "analyze_tickets_for_duplicates", lambda: [MagicMock()])
    monkeypatch.setattr(analyzer, "create_duplicate_report", lambda results: "REPORT")

    class FakeDT(datetime):
        @classmethod
        def now(cls):
            return datetime(2025, 1, 2)
    monkeypatch.setattr("src.freshservice.freshservice_client.datetime", FakeDT)

    fname = analyzer.daily_duplicate_check()
    assert Path(fname).name == "daily_duplicates_20250102.txt"
    assert (tmp_path / fname).read_text() == "REPORT"

