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


def test_daily_duplicate_check_creates_file(tmp_path, monkeypatch):
    os.environ['UNMERGED_LOG'] = str(tmp_path / "unmerged.log")
    analyzer = FreshServiceTicketAnalyzer(FreshServiceConfig())
    monkeypatch.setattr(analyzer, "analyze_tickets_for_duplicates", MagicMock(return_value=[object()]))
    monkeypatch.setattr(analyzer, "create_duplicate_report", MagicMock(return_value="report"))

    out_file = tmp_path / "daily.txt"
    result = analyzer.daily_duplicate_check(output_path=str(out_file))

    assert out_file.exists()
    assert result == str(out_file)
    assert out_file.read_text() == "report"
