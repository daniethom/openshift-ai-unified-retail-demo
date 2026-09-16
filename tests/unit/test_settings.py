"""Tests for central application settings."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.settings import Settings, get_public_config


def test_settings_builds_mcp_urls_from_ports(monkeypatch):
    monkeypatch.delenv("LLM_MCP_URL", raising=False)
    monkeypatch.setenv("LLM_MCP_PORT", "9001")
    monkeypatch.setenv("RAG_MCP_URL", "http://rag.example:8002")
    monkeypatch.setenv("SEARCH_MCP_URL", "http://search.example:8003")
    monkeypatch.setenv("ANALYTICS_MCP_URL", "http://analytics.example:8004")

    settings = Settings.from_env()

    assert settings.llm_mcp_url == "http://127.0.0.1:9001"
    assert settings.rag_mcp_url == "http://rag.example:8002"


def test_settings_supports_granite_endpoint_alias(monkeypatch):
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.setenv("GRANITE_ENDPOINT", "https://granite.example.com")

    settings = Settings.from_env()

    assert settings.llm_api_base == "https://granite.example.com"


def test_mcp_servers_config_shape(monkeypatch):
    monkeypatch.setenv("LLM_MCP_URL", "http://llm:8001")
    monkeypatch.setenv("RAG_MCP_URL", "http://rag:8002")
    monkeypatch.setenv("SEARCH_MCP_URL", "http://search:8003")
    monkeypatch.setenv("ANALYTICS_MCP_URL", "http://analytics:8004")

    settings = Settings.from_env()
    config = settings.mcp_servers_config()

    assert set(config.keys()) == {
        "llm_server",
        "rag_server",
        "search_server",
        "analytics_server",
    }
    assert config["llm_server"]["endpoint"] == "http://llm:8001"


def test_public_config_redacts_secrets(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://meridian:secret@127.0.0.1:5432/meridian",
    )
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret-key-value")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_MODEL_NAME", "granite3.2:8b")

    settings = Settings.from_env()
    public = settings.public_config()

    assert "secret" not in public["DATABASE_URL"]
    assert "***" in public["DATABASE_URL"]
    assert public["TAVILY_API_KEY"] == "configured (redacted)"
    assert public["LLM_API_KEY"] == "ollama"
    assert public["LLM_MODEL_NAME"] == "granite3.2:8b"
    assert public["TAVILY_API_KEY"] != "tvly-secret-key-value"


def test_get_public_config_reads_current_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "granite3.2:8b")
    assert get_public_config()["LLM_MODEL_NAME"] == "granite3.2:8b"
