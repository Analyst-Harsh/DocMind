# app/graph/writer.py
from neo4j import Driver

from app.config import get_settings
from app.graph.client import get_neo4j_driver
from app.graph.extractor import ExtractionResult
from app.ingestion.chunker import Chunk
from app.ingestion.loader import Document

settings = get_settings()


def _entity_id(name: str, entity_type: str) -> str:
    return f"{name.strip().lower()}::{entity_type.strip().upper()}"


def write_chunk_graph(
    document: Document,
    chunk: Chunk,
    embedding: list[float],
    extraction: ExtractionResult,
    driver: Driver | None = None,
) -> None:
    """
    Write one chunk's Document/Chunk nodes plus its extracted entities and
    relations into Neo4j. Entities named only inside a relation (source/
    target not present in extraction.entities) still get a MENTIONS edge
    from this chunk, with type "UNKNOWN" - otherwise the graph traversal
    in graph_searcher.py would have no path back to them.
    """
    if driver is None:
        driver = get_neo4j_driver()

    driver.execute_query(
        """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET d.title = $doc_title, d.doc_type = $doc_type,
                      d.source_path = $source_path, d.tags = $tags
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.doc_id = $doc_id, c.doc_title = $doc_title, c.text = $text,
            c.token_count = $token_count, c.chunk_index = $chunk_index,
            c.source_path = $source_path, c.embedding = $embedding
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        doc_id=document.doc_id,
        doc_title=document.title,
        doc_type=document.doc_type,
        source_path=document.source_path,
        tags=document.tags,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        token_count=chunk.token_count,
        chunk_index=chunk.chunk_index,
        embedding=embedding,
        database_=settings.neo4j_database,
    )

    entities_by_name: dict[str, dict] = {}
    for e in extraction.entities:
        entities_by_name[e.name.strip().lower()] = {
            "entity_id": _entity_id(e.name, e.type),
            "name": e.name,
            "type": e.type,
            "description": e.description,
        }

    def _ensure_entity(name: str) -> str:
        key = name.strip().lower()
        if key not in entities_by_name:
            entities_by_name[key] = {
                "entity_id": _entity_id(name, "UNKNOWN"),
                "name": name,
                "type": "UNKNOWN",
                "description": "",
            }
        return entities_by_name[key]["entity_id"]

    relation_rows = [
        {
            "source_id": _ensure_entity(r.source),
            "target_id": _ensure_entity(r.target),
            "relation": r.relation,
        }
        for r in extraction.relations
    ]
    entity_rows = list(entities_by_name.values())

    if entity_rows:
        driver.execute_query(
            """
            UNWIND $entities AS entity
            MERGE (e:Entity {entity_id: entity.entity_id})
            ON CREATE SET e.name = entity.name, e.type = entity.type,
                          e.description = entity.description
            WITH e
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            entities=entity_rows,
            chunk_id=chunk.chunk_id,
            database_=settings.neo4j_database,
        )

    if relation_rows:
        driver.execute_query(
            """
            UNWIND $relations AS rel
            MATCH (s:Entity {entity_id: rel.source_id})
            MATCH (t:Entity {entity_id: rel.target_id})
            MERGE (s)-[:RELATED_TO {type: rel.relation}]->(t)
            """,
            relations=relation_rows,
            database_=settings.neo4j_database,
        )
