"""
app/ingestion/chunker/table_chunker.py

Converts tables extracted by Docling into Chunk objects with KV-formatted text
(each row as "Header1: val1 | Header2: val2 | ..."), then recursively chunks
the non-table body text using RecursiveChunker. Returns both in a single call
so one Docling pass covers all content for the PDF.

Table splitting is token-based: rows accumulate greedily until adding the next
row would exceed chunk_size, then a new sub-chunk starts — no hardcoded row limit.
Only PDF documents are processed; all other doc types are skipped silently.
"""

from dataclasses import replace
from pathlib import Path

from app.ingestion.chunker.base_chunker import BaseChunker, Chunk
from app.ingestion.chunker.recursive_chunker import RecursiveChunker
from app.ingestion.loader import Document
from app.ingestion.table_extractor import TableData, process_pdf_with_docling


class TableChunker(BaseChunker):
    strategy_name = "table"

    def chunk_document(self, doc: Document) -> list[Chunk]:
        if doc.doc_type != "pdf":
            return []

        non_table_text, tables = process_pdf_with_docling(Path(doc.source_path))

        chunk_index = 0
        table_chunks: list[Chunk] = []
        for table in tables:
            tc = self._table_to_chunks(table, doc, chunk_index)
            table_chunks.extend(tc)
            chunk_index += len(tc)

        text_chunks: list[Chunk] = []
        if non_table_text.strip():
            masked_doc = replace(doc, text=non_table_text)
            text_chunks = RecursiveChunker(
                self.chunk_size, self.chunk_overlap
            ).chunk_document(masked_doc)

        return table_chunks + text_chunks

    def _table_to_chunks(
        self, table: TableData, doc: Document, base_index: int
    ) -> list[Chunk]:
        if not table.rows:
            return []

        chunks: list[Chunk] = []
        current_rows: list[list[str]] = []
        current_tokens = 0
        sub_idx = 0

        for row in table.rows:
            row_tokens = len(
                self.encoder.encode(
                    " | ".join(
                        f"{h}: {v}"
                        for h, v in zip(table.headers, row, strict=True)
                    )
                )
            )
            if current_rows and current_tokens + row_tokens > self.chunk_size:
                chunks.append(
                    self._make_table_chunk(
                        table, doc, base_index + sub_idx, current_rows
                    )
                )
                sub_idx += 1
                current_rows, current_tokens = [], 0

            current_rows.append(row)
            current_tokens += row_tokens

        if current_rows:
            chunks.append(
                self._make_table_chunk(
                    table, doc, base_index + sub_idx, current_rows
                )
            )
        return chunks

    def _make_table_chunk(
        self,
        table: TableData,
        doc: Document,
        chunk_index: int,
        rows: list[list[str]],
    ) -> Chunk:
        kv_text = self._rows_to_kv_text(table.headers, rows)
        return Chunk(
            chunk_id=f"{doc.doc_id}_table_{chunk_index}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            text=kv_text,
            token_count=len(self.encoder.encode(kv_text)),
            chunk_index=chunk_index,
            doc_type=doc.doc_type,
            source_path=doc.source_path,
            tags=doc.tags,
            chunking_strategy="table",
            table_markdown=table.markdown,
            table_headers=table.headers,
            table_index=table.table_index,
            page_number=table.page,
            row_count=table.row_count,
            col_count=table.col_count,
            is_table=True,
        )

    @staticmethod
    def _rows_to_kv_text(headers: list[str], rows: list[list[str]]) -> str:
        return "\n".join(
            " | ".join(f"{h}: {v}" for h, v in zip(headers, row, strict=True))
            for row in rows
        )
