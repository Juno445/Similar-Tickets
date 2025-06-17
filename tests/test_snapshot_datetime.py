import json
from datetime import datetime

import types

import pytest

import src.ann_index as ann_index

class DummyIndex:
    def __init__(self):
        self.d = 1
        self.ntotal = 0
    def search(self, vec, k):
        return [[0.0]], [[-1]]


def test_load_snapshot_parses_datetime(tmp_path, monkeypatch):
    meta = {
        "model": "m",
        "dim": 1,
        "similarity_threshold": 0.5,
        "tickets": [
            {
                "id": "1",
                "title": "t",
                "description": "d",
                "created_at": "2024-01-01T00:00:00Z",
            }
        ],
    }
    meta_file = tmp_path / "meta.json"
    vec_file = tmp_path / "vectors.faiss"
    meta_file.write_text(json.dumps(meta))
    vec_file.write_text("fake")

    monkeypatch.setattr(ann_index, "META_FILE", meta_file)
    monkeypatch.setattr(ann_index, "VEC_FILE", vec_file)
    monkeypatch.setattr(ann_index.faiss, "read_index", lambda path: DummyIndex())

    ann = ann_index.get_ann(force_reload=True)
    assert isinstance(ann.tickets[0].created_at, datetime)
