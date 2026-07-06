from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import tiktoken

from app.ingestion.loader import Document


@dataclass
class Chunk:
    chunk_id: str  # "{doc_id}_{chunk_index}"
    doc_id: str
    doc_title: str
    text: str
    token_count: int
    chunk_index: int
    doc_type: str
    source_path: str
    tags: list[str]
    chunking_strategy: str
    # Table-specific fields — None for non-table chunks
    table_markdown: str | None = None
    table_headers: list[str] | None = None
    table_index: int | None = None
    page_number: int | None = None
    row_count: int | None = None
    col_count: int | None = None
    is_table: bool = False


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class ChunkStrategy(StrEnum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    STRUCTURE_AWARE = "structure_aware"


class BaseChunker(ABC):
    strategy_name: str = "base"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def chunk_document(self, doc: Document) -> list[Chunk]: ...

    def _get_default_separators(self) -> list[str]:
        return DEFAULT_SEPARATORS

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            print(
                f"  [{self.strategy_name}] {doc.doc_id}: {len(doc_chunks)} chunks"
            )
        return all_chunks

    def _make_chunk(self, doc: Document, text: str, chunk_index: int) -> Chunk:
        token_count = len(self.encoder.encode(text))
        return Chunk(
            chunk_id=f"{doc.doc_id}_{self.strategy_name}_{chunk_index}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            text=text,
            token_count=token_count,
            chunk_index=chunk_index,
            doc_type=doc.doc_type,
            source_path=doc.source_path,
            tags=doc.tags,
            chunking_strategy=self.strategy_name,
        )

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        Tries the first separator; any resulting piece still too big gets
        recursively split with the next separator down the list, until
        pieces fit the token budget or no separators remain.
        """
        if not text.strip():
            return []

        if len(self.encoder.encode(text)) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        sep, *rest = separators
        result = []
        for piece in text.split(sep):
            if not piece.strip():
                continue
            if len(self.encoder.encode(piece)) <= self.chunk_size:
                result.append(piece)
            else:
                result.extend(self._split_recursive(piece, rest))
        return result

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split by raw token count when no separator helps."""
        tokens = self.encoder.encode(text)
        return [
            self.encoder.decode(tokens[i : i + self.chunk_size])
            for i in range(0, len(tokens), self.chunk_size)
        ]

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """
        Greedily packs small pieces into chunks up to chunk_size tokens.
        Carries trailing pieces forward into the next chunk to create
        overlap, without re-splitting any individual piece.
        """
        merged, current, current_tokens = [], [], 0

        for piece in pieces:
            piece_tokens = len(self.encoder.encode(piece))

            if current_tokens + piece_tokens > self.chunk_size and current:
                merged.append("\n\n".join(current))
                # carry trailing pieces forward as overlap
                overlap, overlap_tokens = [], 0
                for p in reversed(current):
                    p_tokens = len(self.encoder.encode(p))
                    if overlap_tokens + p_tokens > self.chunk_overlap:
                        break
                    overlap.insert(0, p)
                    overlap_tokens += p_tokens
                current, current_tokens = overlap, overlap_tokens

            current.append(piece)
            current_tokens += piece_tokens

        if current:
            merged.append("\n\n".join(current))
        return merged
