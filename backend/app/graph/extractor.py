# app/graph/extractor.py
from openai import OpenAI
from pydantic import BaseModel

from app.config import get_settings
from app.ingestion.chunker import Chunk

settings = get_settings()
client = OpenAI()

EXTRACTION_SYSTEM_PROMPT = """\
You are extracting a knowledge graph from one chunk of a technical/academic \
document.

Identify the key entities explicitly mentioned in the text (concepts, \
technologies, papers, metrics, organizations, people, tools) and the \
relationships between them.

Rules:
- Keep entity names short and canonical (e.g. "Transformer", not "the \
Transformer architecture introduced in this paper").
- Reuse the exact same name for the same entity every time it recurs in \
this chunk.
- Only extract entities/relations explicitly stated or clearly implied by \
the text - never invent facts.
- If the chunk has no meaningful entities, return empty lists.
"""


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str


class ExtractedRelation(BaseModel):
    source: str
    relation: str
    target: str


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


def extract_entities_and_relations(
    chunk: Chunk, model: str | None = None
) -> ExtractionResult:
    completion = client.chat.completions.parse(
        model=model or settings.llm_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": chunk.text},
        ],
        response_format=ExtractionResult,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return ExtractionResult(entities=[], relations=[])
    return parsed
