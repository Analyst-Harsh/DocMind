# app/graph/schema.py
from neo4j import Driver

from app.config import get_settings
from app.graph.client import get_neo4j_driver

settings = get_settings()

VECTOR_INDEX_NAME = "chunk_embedding_index"

CONSTRAINTS = [
    "CREATE CONSTRAINT document_doc_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
]


def ensure_schema(vector_dim: int, driver: Driver | None = None) -> None:
    """
    Create the uniqueness constraints and the Chunk vector index if they
    don't already exist. Safe to call on every ingest run, same spirit as
    app/ingestion/indexer.py's ensure_collection.
    """
    if driver is None:
        driver = get_neo4j_driver()

    for statement in CONSTRAINTS:
        driver.execute_query(statement, database_=settings.neo4j_database)

    driver.execute_query(
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {{ indexConfig: {{
          `vector.dimensions`: $vector_dim,
          `vector.similarity_function`: 'cosine'
        }}}}
        """,
        vector_dim=vector_dim,
        database_=settings.neo4j_database,
    )
