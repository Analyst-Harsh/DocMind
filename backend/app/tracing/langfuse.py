# app/tracing/langfuse_client.py
from functools import lru_cache

from langfuse import Langfuse

from app.config import get_settings

settings = get_settings()


@lru_cache
def get_langfuse() -> Langfuse:
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
    )
