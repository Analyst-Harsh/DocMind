# scripts/inspect_graph.py
"""
Sanity-check the Neo4j knowledge graph built by scripts/ingest_graph.py:
node/relationship counts, entity-type distribution, top entities by degree,
and a sample of extracted (entity)-[relation]->(entity) triples.

Usage: python -m scripts.inspect_graph
"""

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.graph.client import get_neo4j_driver

NODE_LABELS = ["Document", "Chunk", "Entity"]
REL_TYPES = ["HAS_CHUNK", "MENTIONS", "RELATED_TO"]


def main() -> None:
    settings = get_settings()
    console = Console()
    driver = get_neo4j_driver()
    db = settings.neo4j_database

    console.print("[bold]Node counts[/bold]")
    node_table = Table()
    node_table.add_column("Label")
    node_table.add_column("Count", justify="right")
    for label in NODE_LABELS:
        records, _, _ = driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS count", database_=db
        )
        node_table.add_row(label, str(records[0]["count"]))
    console.print(node_table)

    console.print("\n[bold]Relationship counts[/bold]")
    rel_table = Table()
    rel_table.add_column("Type")
    rel_table.add_column("Count", justify="right")
    for rel_type in REL_TYPES:
        records, _, _ = driver.execute_query(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count",
            database_=db,
        )
        rel_table.add_row(rel_type, str(records[0]["count"]))
    console.print(rel_table)

    console.print("\n[bold]Entity type distribution[/bold]")
    type_records, _, _ = driver.execute_query(
        "MATCH (e:Entity) RETURN e.type AS type, count(*) AS count "
        "ORDER BY count DESC",
        database_=db,
    )
    type_table = Table()
    type_table.add_column("Entity type")
    type_table.add_column("Count", justify="right")
    for record in type_records:
        type_table.add_row(record["type"], str(record["count"]))
    console.print(type_table)

    console.print("\n[bold]Top entities by degree[/bold]")
    degree_records, _, _ = driver.execute_query(
        "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, "
        "COUNT { (e)--() } AS degree ORDER BY degree DESC LIMIT 15",
        database_=db,
    )
    degree_table = Table()
    degree_table.add_column("Entity")
    degree_table.add_column("Type")
    degree_table.add_column("Degree", justify="right")
    for record in degree_records:
        degree_table.add_row(
            record["name"], record["type"], str(record["degree"])
        )
    console.print(degree_table)

    console.print("\n[bold]Sample triples[/bold]")
    triple_records, _, _ = driver.execute_query(
        "MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity) "
        "RETURN s.name AS source, r.type AS relation, t.name AS target "
        "LIMIT 15",
        database_=db,
    )
    triple_table = Table()
    triple_table.add_column("Source")
    triple_table.add_column("Relation")
    triple_table.add_column("Target")
    for record in triple_records:
        triple_table.add_row(
            record["source"], record["relation"], record["target"]
        )
    console.print(triple_table)


if __name__ == "__main__":
    main()
