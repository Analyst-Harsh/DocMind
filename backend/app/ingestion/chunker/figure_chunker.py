"""
app/ingestion/chunker/figure_chunker.py

Converts figures detected by Docling and captioned by GPT-4o Vision into
Chunk objects. Each figure becomes exactly one chunk — captions are short
so no token-based splitting is needed. Only PDF documents are processed.
"""

from pathlib import Path

from app.ingestion.chunker.base_chunker import BaseChunker, Chunk
from app.ingestion.figure_extractor import FigureData, process_pdf_for_figures
from app.ingestion.loader import Document


class FigureChunker(BaseChunker):
    strategy_name = "figure"

    def chunk_document(self, doc: Document) -> list[Chunk]:
        if doc.doc_type != "pdf":
            return []

        figures = process_pdf_for_figures(Path(doc.source_path))
        return [self._figure_to_chunk(doc, fig, i) for i, fig in enumerate(figures)]

    def _figure_to_chunk(
        self, doc: Document, fig: FigureData, chunk_index: int
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}_figure_{chunk_index}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            text=fig.caption,
            token_count=len(self.encoder.encode(fig.caption)),
            chunk_index=chunk_index,
            doc_type=doc.doc_type,
            source_path=doc.source_path,
            tags=doc.tags,
            chunking_strategy="figure",
            page_number=fig.page,
            is_figure=True,
            figure_index=fig.figure_index,
            figure_caption=fig.caption,
        )
