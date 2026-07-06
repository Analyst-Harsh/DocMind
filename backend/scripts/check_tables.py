"""
scripts/check_tables.py

Diagnostic: scan every PDF in the corpus with Docling's DocumentConverter
and report which pages contain tables, their shape, and a Markdown preview.

Docling uses the TableFormer ML model for table structure recognition —
it handles multi-header, borderless, and merged-cell tables correctly.

We convert one page at a time to avoid std::bad_alloc from TableFormer
batching all pages in memory simultaneously.

Usage (from backend/):
    python -m scripts.check_tables
    python -m scripts.check_tables --pdf pdfs/attention_vaswani_2017.pdf
    python -m scripts.check_tables --verbose
"""

import argparse
import logging
from pathlib import Path

import fitz  # PyMuPDF — used only to get page count cheaply
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem

logging.getLogger("docling").setLevel(logging.WARNING)

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def _build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions(do_table_structure=True, do_ocr=False)
    opts.images_scale = 1
    opts.generate_page_images = False
    opts.generate_picture_images = False
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# Build once — TableFormer weights are loaded at construction time.
_CONVERTER = _build_converter()


def _tables_on_page(pdf_path: Path, page_no: int) -> list[dict]:
    """Run Docling on a single page and return any tables found."""
    try:
        result = _CONVERTER.convert(
            str(pdf_path), page_range=(page_no, page_no)
        )
    except Exception as exc:
        print(f"  [page {page_no}] conversion error: {exc}")
        return []

    tables = []
    for item, _ in result.document.iterate_items():
        if not isinstance(item, TableItem):
            continue
        df = item.export_to_dataframe(doc=result.document)
        md = item.export_to_markdown(doc=result.document)
        tables.append({"page": page_no, "df": df, "md": md})
    return tables


def check_pdf(pdf_path: Path, verbose: bool = False) -> dict:
    # Get page count without loading Docling's image pipeline.
    with fitz.open(pdf_path) as fitz_doc:
        total_pages = fitz_doc.page_count

    tables_found = []
    for page_no in range(1, total_pages + 1):
        page_tables = _tables_on_page(pdf_path, page_no)
        tables_found.extend(page_tables)

        if verbose and page_tables:
            for idx, t in enumerate(page_tables, start=1):
                df = t["df"]
                print(
                    f"  Page {page_no}, Table {idx}: {df.shape[0]}r x {df.shape[1]}c"
                )
                md_lines = t["md"].splitlines()
                for line in md_lines[:7]:
                    print(f"    {line.encode('ascii', 'replace').decode()}")
                if len(md_lines) > 7:
                    print(f"    ... ({len(md_lines) - 7} more lines)")
                print()

    pages_with_tables = sorted({t["page"] for t in tables_found})
    return {
        "path": str(pdf_path.relative_to(CORPUS_DIR)),
        "total_tables": len(tables_found),
        "pages_with_tables": pages_with_tables,
        "total_pages": total_pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check table detection in corpus PDFs using Docling."
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Relative path inside corpus/ to a single PDF "
        "(e.g. pdfs/attention_vaswani_2017.pdf). "
        "Omit to scan all PDFs in corpus/pdfs/.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-table Markdown preview.",
    )
    args = parser.parse_args()

    targets = (
        [CORPUS_DIR / args.pdf]
        if args.pdf
        else sorted((CORPUS_DIR / "pdfs").glob("*.pdf"))
    )
    if not targets:
        print("No PDF files found.")
        return

    print(f"Corpus : {CORPUS_DIR}")
    print(f"Engine : Docling TableFormer, page-by-page (images_scale=0.5)\n")

    all_results = []
    for pdf_path in targets:
        print(f"--- {pdf_path.name} ---")
        result = check_pdf(pdf_path, verbose=args.verbose)
        all_results.append(result)

        pages_str = (
            ", ".join(str(p) for p in result["pages_with_tables"]) or "none"
        )
        print(
            f"  Pages with tables : {pages_str}\n"
            f"  Total tables      : {result['total_tables']} / {result['total_pages']} pages\n"
        )

    print("=== Summary ===")
    for r in all_results:
        print(
            f"  {r['path']:45s}  "
            f"{r['total_tables']} table(s) on {len(r['pages_with_tables'])} page(s)"
        )


if __name__ == "__main__":
    main()
