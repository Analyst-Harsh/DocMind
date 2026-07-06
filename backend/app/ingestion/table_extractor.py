"""
app/ingestion/table_extractor.py

Runs Docling's TableFormer on a PDF page-by-page, separating TableItem elements
from body text. Returns both the non-table text (table regions excluded) and
structured TableData objects with KV-ready cell values.

Processing one page at a time avoids the std::bad_alloc that TableFormer causes
when it batches all pages simultaneously — same fix as scripts/check_tables.py.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF — used only to get page count cheaply
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem
from structlog import get_logger

logging.getLogger("docling").setLevel(logging.WARNING)

log = get_logger(__name__)

_CONVERTER: DocumentConverter | None = None


def _build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions(do_table_structure=True, do_ocr=False)
    opts.images_scale = 1
    opts.generate_page_images = False
    opts.generate_picture_images = False
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _get_converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = _build_converter()
    return _CONVERTER


@dataclass
class TableData:
    page: int
    table_index: int   # 0-indexed count of tables seen on this page
    markdown: str
    row_count: int
    col_count: int
    headers: list[str]
    rows: list[list[str]]  # cell values as strings, ready for KV formatting


def process_pdf_with_docling(pdf_path: Path) -> tuple[str, list[TableData]]:
    """
    Run Docling on the PDF one page at a time.

    Returns:
        (non_table_text, tables) where non_table_text is all body text with
        table regions excluded, and tables is the list of extracted TableData.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as fitz_doc:
        total_pages = fitz_doc.page_count

    text_parts: list[str] = []
    tables: list[TableData] = []
    converter = _get_converter()

    for page_no in range(1, total_pages + 1):
        page_table_count = 0
        try:
            result = converter.convert(
                str(pdf_path), page_range=(page_no, page_no)
            )
        except Exception as exc:
            log.warning("docling_page_error", page=page_no, error=str(exc))
            continue

        for item, _ in result.document.iterate_items():
            if isinstance(item, TableItem):
                df = item.export_to_dataframe(doc=result.document)
                md = item.export_to_markdown(doc=result.document)
                headers = [str(c) for c in df.columns]
                rows = [[str(v) for v in row] for _, row in df.iterrows()]

                tables.append(
                    TableData(
                        page=page_no,
                        table_index=page_table_count,
                        markdown=md,
                        row_count=len(df),
                        col_count=len(df.columns),
                        headers=headers,
                        rows=rows,
                    )
                )
                page_table_count += 1
            elif hasattr(item, "text") and item.text:
                text_parts.append(item.text)

    return "\n\n".join(text_parts), tables
