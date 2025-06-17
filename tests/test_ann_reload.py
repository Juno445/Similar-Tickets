import time
from pathlib import Path

import numpy as np
import pytest

from src.ann_index import CURRENT_LINK, META_FILE, get_ann

pytestmark = pytest.mark.order("last")


def test_hot_reload(tmp_path, monkeypatch):
    # prime singleton
    ann1 = get_ann()
    vec = np.random.rand(ann1.dim).astype("float32")
    _ = ann1.top_k(vec, 1)

    # copy meta, touch to force newer mtime
    meta_copy = tmp_path / "meta.json"
    meta_copy.write_bytes(META_FILE.read_bytes())
    time.sleep(0.01)
    meta_copy.touch()

    monkeypatch.setattr("src.ann_index.META_FILE", meta_copy)
    ann2 = get_ann(force_reload=True)

    assert ann1 is not ann2
    assert ann1.top_k(vec, 1) == ann2.top_k(vec, 1)