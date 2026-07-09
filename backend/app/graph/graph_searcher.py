# app/graph/graph_searcher.py
from neo4j import Driver
from structlog import get_logger

from app.config import get_settings
from app.graph.client import get_neo4j_driver
from app.graph.schema import VECTOR_INDEX_NAME
from app.ingestion.embedder import embed_query
from app.retrieval.reranker import rerank as _rerank
from app.retrieval.searcher import RetrievedChunk

settings = get_settings()
log = get_logger(__name__)


def _record_to_chunk(record, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=record["chunk_id"] or "",
        doc_id=record["doc_id"] or "",
        doc_title=record["doc_title"] or "",
        text=record["text"] or "",
        score=score,
        source_path=record["source_path"] or "",
        chunk_index=record["chunk_index"] or 0,
    )


def retrieve_graph(
    query: str,
    top_k: int = 5,
    driver: Driver | None = None,
    embedding_model: str | None = None,
    rerank: bool = False,
) -> list[RetrievedChunk]:
    """
    Vector search over Chunk.embedding via Neo4j's native vector index,
    expanded by chunks reached through a 1-hop shared-entity traversal.
    The traversal is seeded from the *whole* over-fetched candidate pool
    (top_k * 3), not just the top_k direct hits - the lower-ranked-but-
    still-relevant vector matches give the graph expansion more entry
    points to work from, without themselves being returned as direct
    hits.

    rerank=True always runs the expansion (not just to fill gaps) and
    re-scores direct hits + expansion candidates together with the
    existing cross-encoder reranker (app/retrieval/reranker.py), so the
    entity graph gets a real chance to surface multi-hop chunks a
    single vector query would rank too low to return - this is the
    mechanism that's actually supposed to make graph RAG useful. It also
    adds a 2-hop pass over the extracted RELATED_TO edges between
    entities (seed chunk -> mentioned entity -> RELATED_TO -> other
    entity -> chunk mentioning that entity), which reaches chunks that
    share no entity directly with the seed but are connected through a
    documented relationship - the actual multi-hop case single-entity
    co-mention can't cover. rerank=False only backfills the vector hits
    when they undershoot top_k, using a fabricated backfill_score floor,
    since there's no way to meaningfully rank a bigger pool without a
    reranker.

    Returns the same RetrievedChunk shape as app/retrieval/searcher.py so
    this drops straight into the existing eval harness as a retrieve_fn.
    """
    if driver is None:
        driver = get_neo4j_driver()

    query_vector = embed_query(query, model=embedding_model)

    vector_records, _, _ = driver.execute_query(
        """
        CALL db.index.vector.queryNodes($index_name, $k, $vector)
        YIELD node, score
        RETURN node.chunk_id AS chunk_id, node.doc_id AS doc_id,
               node.doc_title AS doc_title, node.text AS text,
               node.source_path AS source_path,
               node.chunk_index AS chunk_index, score
        """,
        index_name=VECTOR_INDEX_NAME,
        k=top_k * 3,
        vector=query_vector,
        database_=settings.neo4j_database,
    )

    chunks = [_record_to_chunk(r, r["score"]) for r in vector_records]

    def _expand(seed_ids: list[str], seen_ids: set[str], limit: int) -> list:
        # No vector score exists for traversal-only candidates, so rank by
        # bridging-entity weight instead: sum(1 / degree(e)) rather than a
        # plain count. A hub entity mentioned across dozens of chunks
        # (e.g. "Transformer" in a paper about Transformers) says almost
        # nothing about topical relevance, so it must count for less than
        # a rare, specific entity shared by only a couple of chunks - the
        # graph analogue of IDF. Without this, generic reference/appendix
        # chunks that happen to mention many entities outrank the chunks
        # that are actually on-topic (see 2-hop docstring below for the
        # concrete case this was caught on).
        expansion_records, _, _ = driver.execute_query(
            """
            MATCH (seed:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Chunk)
            WHERE seed.chunk_id IN $seed_ids AND NOT other.chunk_id IN $seen_ids
            WITH DISTINCT other, e
            WITH other, sum(1.0 / COUNT { (e)--() }) AS weight
            RETURN other.chunk_id AS chunk_id, other.doc_id AS doc_id,
                   other.doc_title AS doc_title, other.text AS text,
                   other.source_path AS source_path,
                   other.chunk_index AS chunk_index
            ORDER BY weight DESC
            LIMIT $limit
            """,
            seed_ids=seed_ids,
            seen_ids=list(seen_ids),
            limit=limit,
            database_=settings.neo4j_database,
        )
        return expansion_records

    def _expand_related(seed_ids: list[str], seen_ids: set[str], limit: int) -> list:
        # 2-hop: bridges chunks that don't share an entity outright but
        # are connected via a documented entity-to-entity relationship.
        # RELATED_TO is matched undirected - the source/target order the
        # extractor assigned reflects sentence structure, not which
        # direction is useful to retrieve across.
        #
        # Same inverse-degree weighting as _expand, and it matters more
        # here: hub entities (e.g. "Transformer", degree 33 in this
        # corpus) fan out through RELATED_TO to dozens of unrelated
        # entities, so a plain count(DISTINCT e2) let reference-list and
        # appendix chunks that densely mention many entities "win" purely
        # by mentioning a lot of things, not by being on-topic - caught
        # empirically when this surfaced a Ragas paper title-page chunk
        # as the top 2-hop candidate for a Transformer/RAG question.
        expansion_records, _, _ = driver.execute_query(
            """
            MATCH (seed:Chunk)-[:MENTIONS]->(:Entity)-[:RELATED_TO]-(e2:Entity)
                  <-[:MENTIONS]-(other:Chunk)
            WHERE seed.chunk_id IN $seed_ids AND NOT other.chunk_id IN $seen_ids
            WITH DISTINCT other, e2
            WITH other, sum(1.0 / COUNT { (e2)--() }) AS weight
            RETURN other.chunk_id AS chunk_id, other.doc_id AS doc_id,
                   other.doc_title AS doc_title, other.text AS text,
                   other.source_path AS source_path,
                   other.chunk_index AS chunk_index
            ORDER BY weight DESC
            LIMIT $limit
            """,
            seed_ids=seed_ids,
            seen_ids=list(seen_ids),
            limit=limit,
            database_=settings.neo4j_database,
        )
        return expansion_records

    if rerank:
        log.info("Expanding results for query", query=query)
        seed_ids = [c.chunk_id for c in chunks]
        seen_chunk_ids = {c.chunk_id for c in chunks}

        one_hop = _expand(
            seed_ids=seed_ids, seen_ids=seen_chunk_ids, limit=top_k * 2
        )
        seen_chunk_ids |= {r["chunk_id"] for r in one_hop}

        two_hop = _expand_related(
            seed_ids=seed_ids, seen_ids=seen_chunk_ids, limit=top_k * 2
        )

        candidate_pool = (
            chunks
            + [_record_to_chunk(r, 0.0) for r in one_hop]
            + [_record_to_chunk(r, 0.0) for r in two_hop]
        )
        return _rerank(query, candidate_pool, top_k)

    direct_hits = chunks[:top_k]
    results = direct_hits.copy()
    seen_chunk_ids = {r.chunk_id for r in direct_hits}

    needs_expansion = direct_hits and len(results) < top_k
    if needs_expansion:
        log.info("Expanding results for query", query=query)
        expansion_records = _expand(
            seed_ids=[r["chunk_id"] for r in vector_records],
            seen_ids=seen_chunk_ids,
            limit=top_k * 2,
        )
        backfill_score = min(r.score for r in direct_hits) * 0.5
        results.extend(
            _record_to_chunk(r, backfill_score) for r in expansion_records
        )

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]
