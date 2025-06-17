import os
import sys
import types
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Default environment values so src.config can import during test collection
os.environ.setdefault("FS_DOMAIN", "example.freshservice.com")
os.environ.setdefault("FS_API_KEY", "dummy")
os.environ.setdefault("GROUP_ID", "1")
os.environ.pop("GROUP_IDS", None)
os.environ.pop("FS_API_KEY_FILE", None)

# Mock heavy optional dependencies so tests run without installing them

# sentence_transformers
sys.modules.setdefault(
    'sentence_transformers',
    types.SimpleNamespace(SentenceTransformer=lambda name: None)
)

# matplotlib / seaborn / pandas already mocked in tests but replicate here
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('pyplot'))
sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

# numpy stub with minimal functionality
np_stub = types.ModuleType('numpy')
np_stub.ndarray = object
np_stub.argsort = lambda x: []
np_stub.vstack = lambda x: []
np_stub.dot = lambda a, b: 0
np_stub.array = lambda x: x
np_stub.zeros = lambda shape: []
np_stub.linalg = types.SimpleNamespace(norm=lambda x: 1)
sys.modules.setdefault('numpy', np_stub)

# scikit-learn stubs
sklearn_mod = types.ModuleType('sklearn')
metrics_mod = types.ModuleType('metrics')
metrics_mod.accuracy_score = lambda y_true, y_pred: 0
metrics_mod.precision_recall_fscore_support = (
    lambda y_true, y_pred, average=None: (0, 0, 0, 0)
)
pairwise_mod = types.ModuleType('pairwise')
pairwise_mod.cosine_similarity = lambda X, Y=None: []
cluster_mod = types.ModuleType('cluster')
cluster_mod.DBSCAN = lambda *a, **k: types.SimpleNamespace(fit_predict=lambda X: [])
model_selection_mod = types.ModuleType('model_selection')
model_selection_mod.train_test_split = lambda *a, **k: ([], [])

sys.modules.setdefault('sklearn', sklearn_mod)
sys.modules.setdefault('sklearn.metrics', metrics_mod)
sys.modules.setdefault('sklearn.metrics.pairwise', pairwise_mod)
sys.modules.setdefault('sklearn.cluster', cluster_mod)
sys.modules.setdefault('sklearn.model_selection', model_selection_mod)

# python-dateutil stub
dateutil_mod = types.ModuleType('dateutil')
parser_mod = types.ModuleType('parser')
parser_mod.isoparse = lambda s: datetime.now()
sys.modules.setdefault('dateutil', dateutil_mod)
sys.modules.setdefault('dateutil.parser', parser_mod)


import pytest


@pytest.fixture(autouse=True)
def freshservice_env(monkeypatch):
    """Provide default Freshservice configuration values for tests."""
    monkeypatch.setenv("FS_DOMAIN", "example.freshservice.com")
    monkeypatch.setenv("FS_API_KEY", "dummy")
    monkeypatch.setenv("GROUP_ID", "1")
    monkeypatch.delenv("GROUP_IDS", raising=False)
    monkeypatch.delenv("FS_API_KEY_FILE", raising=False)
