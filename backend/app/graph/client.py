# app/graph/client.py
from functools import lru_cache

from neo4j import Driver, GraphDatabase

from app.config import get_settings

settings = get_settings()


@lru_cache
def get_neo4j_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
