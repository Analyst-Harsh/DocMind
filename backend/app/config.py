# create env config file
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "docmind"

    # OpenAI settings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    draft_generate_llm_model: str = "gpt-4o"

    use_reranker: bool = True

    # Langfuse settings
    enable_tracing: bool = False
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str

    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    semantic_cache_ttl_seconds: int = 86400
    semantic_cache_similarity_threshold: float = 0.75
    enable_semantic_cache: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
