from backend.app.ingestion.loader import Document

from .base_chunker import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def chunk_document(self, doc: Document) -> list[Chunk]:
        # first recursively split the document text into smaller chunks
        split_texts = self._split_recursive(
            doc.text, self._get_default_separators()
        )
        # merge the split texts into chunks of the specified size with overlap
        merged_chunks = self._merge_pieces(split_texts)
        return [
            self._make_chunk(doc, text, i)
            for i, text in enumerate(merged_chunks)
        ]
