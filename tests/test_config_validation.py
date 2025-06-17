import pytest

from src.config import FreshServiceConfig


def test_missing_domain(monkeypatch):
    with pytest.raises(ValueError):
        FreshServiceConfig(domain="", api_key="dummy", group_ids=[1])


def test_missing_group_ids(monkeypatch):
    with pytest.raises(ValueError):
        FreshServiceConfig(domain="example", api_key="dummy", group_ids=[])


def test_api_key_from_file(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("secret")
    monkeypatch.setenv("FS_API_KEY_FILE", str(key_file))
    cfg = FreshServiceConfig(domain="example", api_key="", group_ids=[1])
    assert cfg.api_key == "secret"


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("FS_API_KEY_FILE", raising=False)
    with pytest.raises(ValueError):
        FreshServiceConfig(domain="example", api_key="", group_ids=[1])
