"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local development and OpenShift deployments."""

    llm_mcp_url: str
    rag_mcp_url: str
    search_mcp_url: str
    analytics_mcp_url: str

    llm_api_base: str
    llm_api_key: str
    llm_model_name: str

    milvus_uri: str
    milvus_host: str
    milvus_port: int
    embedding_model: str

    tavily_api_key: str
    tavily_max_results: int
    milvus_collection_name: str
    rag_use_fallback: bool
    log_level: str
    meridian_debug: bool
    openshift_namespace: str

    @classmethod
    def from_env(cls) -> Settings:
        milvus_host = _env("MILVUS_HOST", "127.0.0.1")
        milvus_port = _env_int("MILVUS_PORT", 19530)
        milvus_uri = _env("MILVUS_URI") or f"http://{milvus_host}:{milvus_port}"

        llm_port = _env("LLM_MCP_PORT", "8001")
        rag_port = _env("RAG_MCP_PORT", "8002")
        search_port = _env("SEARCH_MCP_PORT", "8003")
        analytics_port = _env("ANALYTICS_MCP_PORT", "8004")

        llm_api_base = _env("LLM_API_BASE") or _env(
            "GRANITE_ENDPOINT", "http://127.0.0.1:8080"
        )

        return cls(
            llm_mcp_url=_env("LLM_MCP_URL", f"http://127.0.0.1:{llm_port}"),
            rag_mcp_url=_env("RAG_MCP_URL", f"http://127.0.0.1:{rag_port}"),
            search_mcp_url=_env("SEARCH_MCP_URL", f"http://127.0.0.1:{search_port}"),
            analytics_mcp_url=_env(
                "ANALYTICS_MCP_URL", f"http://127.0.0.1:{analytics_port}"
            ),
            llm_api_base=llm_api_base,
            llm_api_key=_env("LLM_API_KEY", "not-needed"),
            llm_model_name=_env("LLM_MODEL_NAME", "gpt-4o-mini"),
            milvus_uri=milvus_uri,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            embedding_model=_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            tavily_api_key=_env("TAVILY_API_KEY"),
            tavily_max_results=_env_int("TAVILY_MAX_RESULTS", 5),
            milvus_collection_name=_env("MILVUS_COLLECTION_NAME", "meridian_knowledge"),
            rag_use_fallback=_env_bool("RAG_USE_FALLBACK", False),
            log_level=_env("LOG_LEVEL", "INFO"),
            meridian_debug=_env_bool("MERIDIAN_DEBUG", False),
            openshift_namespace=_env("OPENSHIFT_NAMESPACE", "retail-ai-demo"),
        )

    def mcp_servers_config(self) -> dict[str, dict[str, str]]:
        """Endpoint map used for agent status display and wiring."""
        return {
            "llm_server": {"endpoint": self.llm_mcp_url},
            "rag_server": {"endpoint": self.rag_mcp_url},
            "search_server": {"endpoint": self.search_mcp_url},
            "analytics_server": {"endpoint": self.analytics_mcp_url},
        }


settings = Settings.from_env()
