"""Tests for HuggingFace model shortcuts, the 'default' env override bug fix,
and setup_helper HuggingFace env handling."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp.chroma_client import (
    HUGGINGFACE_MODEL_SHORTCUTS,
    ChromaClient,
    HuggingFaceEmbeddingFunction,
    create_chroma_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config_file(tmp_path: Path, semantic_search: dict) -> Path:
    """Write a minimal config.json and return its path."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"semantic_search": semantic_search}))
    return cfg


def _build_client_obj(embedding_model: str, embedding_config: dict | None = None):
    """Build a bare ChromaClient-like object with just enough state for
    ``_create_embedding_function`` without touching ChromaDB/disk."""
    obj = object.__new__(ChromaClient)
    obj.embedding_model = embedding_model
    obj.embedding_config = embedding_config or {}
    return obj


# ---------------------------------------------------------------------------
# 1. Bug fix: ZOTERO_EMBEDDING_MODEL="default" must NOT override config.json
# ---------------------------------------------------------------------------

class TestDefaultEnvOverrideBug:
    """Regression tests for the 'default' env var override bug."""

    def test_env_default_does_not_override_config_file(self, monkeypatch, tmp_path):
        """ZOTERO_EMBEDDING_MODEL=default + config.json embedding_model=bge-zh
        → config.json value should win."""
        monkeypatch.setenv("ZOTERO_EMBEDDING_MODEL", "default")

        cfg_path = _make_config_file(tmp_path, {"embedding_model": "bge-zh"})

        # Intercept ChromaClient.__init__ to avoid real ChromaDB I/O
        captured = {}
        original_init = ChromaClient.__init__

        def fake_init(self, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(ChromaClient, "__init__", fake_init)

        create_chroma_client(config_path=str(cfg_path))
        assert captured["embedding_model"] == "bge-zh"

    def test_env_empty_does_not_override_config_file(self, monkeypatch, tmp_path):
        """Empty ZOTERO_EMBEDDING_MODEL should not override config."""
        monkeypatch.setenv("ZOTERO_EMBEDDING_MODEL", "")

        cfg_path = _make_config_file(tmp_path, {"embedding_model": "bge-en"})

        captured = {}

        def fake_init(self, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(ChromaClient, "__init__", fake_init)

        create_chroma_client(config_path=str(cfg_path))
        assert captured["embedding_model"] == "bge-en"

    def test_env_real_value_overrides_config_file(self, monkeypatch, tmp_path):
        """A non-default env value should still override config.json."""
        monkeypatch.setenv("ZOTERO_EMBEDDING_MODEL", "qwen")

        cfg_path = _make_config_file(tmp_path, {"embedding_model": "bge-zh"})

        captured = {}

        def fake_init(self, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(ChromaClient, "__init__", fake_init)

        create_chroma_client(config_path=str(cfg_path))
        assert captured["embedding_model"] == "qwen"

    def test_no_env_uses_config_file(self, monkeypatch, tmp_path):
        """When ZOTERO_EMBEDDING_MODEL is unset, config.json should be used."""
        monkeypatch.delenv("ZOTERO_EMBEDDING_MODEL", raising=False)

        cfg_path = _make_config_file(tmp_path, {"embedding_model": "bge-large-zh"})

        captured = {}

        def fake_init(self, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(ChromaClient, "__init__", fake_init)

        create_chroma_client(config_path=str(cfg_path))
        assert captured["embedding_model"] == "bge-large-zh"


# ---------------------------------------------------------------------------
# 2. HUGGINGFACE_MODEL_SHORTCUTS registry
# ---------------------------------------------------------------------------

class TestHuggingFaceShortcuts:
    """Verify the shortcut registry is correct and complete."""

    def test_registry_keys_exist(self):
        expected_keys = {"qwen", "bge", "bge-zh", "bge-en", "bge-large-zh", "embeddinggemma"}
        assert expected_keys == set(HUGGINGFACE_MODEL_SHORTCUTS.keys())

    def test_registry_values_are_full_model_names(self):
        for key, value in HUGGINGFACE_MODEL_SHORTCUTS.items():
            # Full HF model names contain a slash (org/model)
            assert "/" in value, f"Shortcut '{key}' → '{value}' doesn't look like a HF model name"

    @pytest.mark.parametrize("shortcut,expected_model", [
        ("qwen", "Qwen/Qwen3-Embedding-0.6B"),
        ("bge", "BAAI/bge-small-zh-v1.5"),
        ("bge-zh", "BAAI/bge-small-zh-v1.5"),
        ("bge-en", "BAAI/bge-small-en-v1.5"),
        ("bge-large-zh", "BAAI/bge-large-zh-v1.5"),
        ("embeddinggemma", "google/embeddinggemma-300m"),
    ])
    def test_shortcut_values(self, shortcut, expected_model):
        assert HUGGINGFACE_MODEL_SHORTCUTS[shortcut] == expected_model


# ---------------------------------------------------------------------------
# 3. _create_embedding_function dispatching
# ---------------------------------------------------------------------------

class TestCreateEmbeddingFunction:
    """Test that _create_embedding_function routes to the right class."""

    def test_default_model(self, monkeypatch):
        obj = _build_client_obj("default")
        ef = obj._create_embedding_function()
        # Default uses ChromaDB's built-in DefaultEmbeddingFunction
        assert type(ef).__name__ == "DefaultEmbeddingFunction"

    @pytest.mark.parametrize("shortcut", list(HUGGINGFACE_MODEL_SHORTCUTS.keys()))
    def test_shortcut_dispatches_to_huggingface(self, monkeypatch, shortcut):
        """Each shortcut key should create a HuggingFaceEmbeddingFunction
        with the full model name from the registry."""
        # Mock HuggingFaceEmbeddingFunction to avoid downloading models
        calls = []
        import zotero_mcp.chroma_client as cc_module

        class FakeHF:
            def __init__(self, model_name):
                calls.append(model_name)

        monkeypatch.setattr(cc_module, "HuggingFaceEmbeddingFunction", FakeHF)

        obj = _build_client_obj(shortcut)
        obj._create_embedding_function()

        assert len(calls) == 1
        assert calls[0] == HUGGINGFACE_MODEL_SHORTCUTS[shortcut]

    def test_arbitrary_hf_model_name(self, monkeypatch):
        """A non-reserved, non-shortcut string should be treated as a raw
        HuggingFace model name."""
        calls = []
        import zotero_mcp.chroma_client as cc_module

        class FakeHF:
            def __init__(self, model_name):
                calls.append(model_name)

        monkeypatch.setattr(cc_module, "HuggingFaceEmbeddingFunction", FakeHF)

        obj = _build_client_obj("BAAI/bge-m3")
        obj._create_embedding_function()

        assert calls == ["BAAI/bge-m3"]

    def test_shortcut_respects_embedding_config_override(self, monkeypatch):
        """If embedding_config.model_name is set, it should override the
        shortcut's default."""
        calls = []
        import zotero_mcp.chroma_client as cc_module

        class FakeHF:
            def __init__(self, model_name):
                calls.append(model_name)

        monkeypatch.setattr(cc_module, "HuggingFaceEmbeddingFunction", FakeHF)

        obj = _build_client_obj("bge-zh", {"model_name": "BAAI/bge-base-zh-v1.5"})
        obj._create_embedding_function()

        assert calls == ["BAAI/bge-base-zh-v1.5"]


# ---------------------------------------------------------------------------
# 4. HF_ENDPOINT logging in HuggingFaceEmbeddingFunction
# ---------------------------------------------------------------------------

class TestHFEndpointLogging:
    def test_hf_endpoint_logged_when_set(self, monkeypatch, caplog):
        """When HF_ENDPOINT is set, __init__ should log it."""
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")

        # Mock sentence_transformers to avoid real model loading
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = MagicMock()
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

        import logging
        with caplog.at_level(logging.INFO, logger="zotero_mcp.chroma_client"):
            ef = HuggingFaceEmbeddingFunction(model_name="test/model")

        assert ef.model_name == "test/model"
        assert any("hf-mirror.com" in r.message for r in caplog.records)

    def test_hf_endpoint_not_logged_when_unset(self, monkeypatch, caplog):
        """When HF_ENDPOINT is not set, no mirror log line."""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = MagicMock()
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

        import logging
        with caplog.at_level(logging.INFO, logger="zotero_mcp.chroma_client"):
            ef = HuggingFaceEmbeddingFunction(model_name="test/model")

        assert ef.model_name == "test/model"
        assert not any("mirror" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# 5. setup_helper: update_claude_config env writing
# ---------------------------------------------------------------------------

class TestSetupHelperEnvWriting:
    """Test that update_claude_config writes correct env vars."""

    def test_default_model_not_written_to_env(self, tmp_path):
        from zotero_mcp.setup_helper import update_claude_config

        config_path = tmp_path / "claude_desktop_config.json"

        update_claude_config(
            config_path,
            "/usr/bin/zotero-mcp",
            local=True,
            semantic_config={"embedding_model": "default"},
        )

        written = json.loads(config_path.read_text())
        env = written["mcpServers"]["zotero"]["env"]
        assert "ZOTERO_EMBEDDING_MODEL" not in env

    def test_nondefault_model_written_to_env(self, tmp_path):
        from zotero_mcp.setup_helper import update_claude_config

        config_path = tmp_path / "claude_desktop_config.json"

        update_claude_config(
            config_path,
            "/usr/bin/zotero-mcp",
            local=True,
            semantic_config={"embedding_model": "bge-zh"},
        )

        written = json.loads(config_path.read_text())
        env = written["mcpServers"]["zotero"]["env"]
        assert env["ZOTERO_EMBEDDING_MODEL"] == "bge-zh"

    def test_hf_endpoint_written_for_huggingface(self, tmp_path):
        from zotero_mcp.setup_helper import update_claude_config

        config_path = tmp_path / "claude_desktop_config.json"

        update_claude_config(
            config_path,
            "/usr/bin/zotero-mcp",
            local=True,
            semantic_config={
                "embedding_model": "bge-zh",
                "embedding_config": {"hf_endpoint": "https://hf-mirror.com"},
            },
        )

        written = json.loads(config_path.read_text())
        env = written["mcpServers"]["zotero"]["env"]
        assert env["HF_ENDPOINT"] == "https://hf-mirror.com"

    def test_openai_env_still_works(self, tmp_path):
        from zotero_mcp.setup_helper import update_claude_config

        config_path = tmp_path / "claude_desktop_config.json"

        update_claude_config(
            config_path,
            "/usr/bin/zotero-mcp",
            local=True,
            semantic_config={
                "embedding_model": "openai",
                "embedding_config": {
                    "api_key": "sk-test",
                    "model_name": "text-embedding-3-small",
                },
            },
        )

        written = json.loads(config_path.read_text())
        env = written["mcpServers"]["zotero"]["env"]
        assert env["ZOTERO_EMBEDDING_MODEL"] == "openai"
        assert env["OPENAI_API_KEY"] == "sk-test"
        assert env["OPENAI_EMBEDDING_MODEL"] == "text-embedding-3-small"
        assert "HF_ENDPOINT" not in env

    def test_no_semantic_config_no_embedding_env(self, tmp_path):
        from zotero_mcp.setup_helper import update_claude_config

        config_path = tmp_path / "claude_desktop_config.json"

        update_claude_config(
            config_path,
            "/usr/bin/zotero-mcp",
            local=True,
            semantic_config=None,
        )

        written = json.loads(config_path.read_text())
        env = written["mcpServers"]["zotero"]["env"]
        assert "ZOTERO_EMBEDDING_MODEL" not in env
        assert "HF_ENDPOINT" not in env
