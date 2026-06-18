from backend.app.ingestion.loader import Document

from .base_chunker import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    strategy_name = "fixed_size"

    def chunk_document(self, doc: Document) -> list[Chunk]:
        return [self._make_chunk(doc, text, i) for i, text in enumerate(self._hard_split(doc.text))]
