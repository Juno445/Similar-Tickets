import importlib
import os

from src import config


def test_api_key_loaded_from_file(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("filekey\n")
    monkeypatch.setenv("FS_API_KEY_FILE", str(key_file))
    monkeypatch.delenv("FS_API_KEY", raising=False)
    importlib.reload(config)
    cfg = config.FreshServiceConfig()
    assert cfg.api_key == "filekey"


def test_env_api_key_overrides_file(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("filekey\n")
    monkeypatch.setenv("FS_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("FS_API_KEY", "envkey")
    importlib.reload(config)
    cfg = config.FreshServiceConfig()
    assert cfg.api_key == "envkey"

