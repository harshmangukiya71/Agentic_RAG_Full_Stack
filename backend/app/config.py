"""
config.py - centralised settings via pydantic-settings.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_max_tokens: int = 1024
    nvidia_temperature: float = 0.1

    @property
    def gemini_api_key(self) -> str:
        return self.nvidia_api_key

    @property
    def gemini_base_url(self) -> str:
        return self.nvidia_base_url

    @property
    def gemini_model(self) -> str:
        return self.nvidia_model

    @property
    def gemini_max_tokens(self) -> int:
        return self.nvidia_max_tokens

    @property
    def gemini_temperature(self) -> float:
        return self.nvidia_temperature

    summary_chunk_size_tokens: int = 1200
    summary_chunk_overlap_tokens: int = 120
    summary_max_tokens_per_chunk: int = 350
    summary_final_max_tokens: int = 700

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cpu"

    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "legal_docs"

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    min_native_chars_per_page: int = 40
    ocr_render_dpi: int = 220

    bm25_top_k: int = 20
    dense_top_k: int = 20
    graph_top_k: int = 12
    graph_boost_weight: float = 0.12
    rerank_top_k: int = 5
    final_context_k: int = 5

    min_relevance_score: float = 0.05
    min_answer_coverage: float = 0.20
    confidence_scale_factor: float = 1.6

    memory_backend: str = "auto"
    redis_url: str = "redis://localhost:6379/0"
    memory_ttl_seconds: int = 60 * 60 * 24
    memory_max_turns: int = 4

    cache_enabled: bool = True
    cache_ttl_seconds: int = 60 * 60
    cache_max_entries: int = 512
    semantic_cache_threshold: float = 0.92

    graph_backend: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # ── Agentic RAG settings ──────────────────────────────────────────────────
    query_agent_enabled: bool = True
    reasoning_agent_enabled: bool = True
    max_retrieval_iterations: int = 3
    cache_load_monitoring: bool = True
    cache_ready_percent: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
