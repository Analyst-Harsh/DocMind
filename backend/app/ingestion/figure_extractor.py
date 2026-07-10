"""
app/ingestion/figure_extractor.py

Detects figures in PDF pages using Docling's picture detection, then captions
each with GPT-4o Vision. Returns FigureData objects ready for FigureChunker.

Processing one page at a time (same as table_extractor.py) to avoid
std::bad_alloc from Docling when batching all pages simultaneously.
"""

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import fitz
import openai
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem
from structlog import get_logger

log = get_logger(__name__)

MIN_FIGURE_PX = 100  # skip images smaller than this on either dimension

_FIGURE_CONVERTER: DocumentConverter | None = None
_OPENAI_CLIENT: openai.OpenAI | None = None

_CAPTION_PROMPT = (
    "Describe this technical diagram in detail. Include the diagram type "
    "(e.g. flowchart, architecture diagram, bar chart, circuit diagram), "
    "key components, relationships between elements, and any visible labels "
    "or data values. Write as if explaining the diagram to someone who "
    "cannot see it but needs to answer questions about its contents."
)


def _get_figure_converter() -> DocumentConverter:
    global _FIGURE_CONVERTER
    if _FIGURE_CONVERTER is None:
        opts = PdfPipelineOptions(do_table_structure=False, do_ocr=False)
        opts.images_scale = 2
        opts.generate_page_images = False
        opts.generate_picture_images = True
        _FIGURE_CONVERTER = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
            }
        )
    return _FIGURE_CONVERTER


def _get_openai_client() -> openai.OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = openai.OpenAI()
    return _OPENAI_CLIENT


@dataclass
class FigureData:
    page: int
    figure_index: int  # 0-indexed per page
    caption: str  # GPT-4o Vision output
    width: int
    height: int


def _image_to_base64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _caption_figure(img) -> str | None:
    b64 = _image_to_base64(img)
    try:
        response = _get_openai_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            },
                        },
                        {"type": "text", "text": _CAPTION_PROMPT},
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as exc:
        log.warning("figure_caption_error", error=str(exc))
        return None


def process_pdf_for_figures(pdf_path: Path) -> list[FigureData]:
    """
    Run Docling on the PDF one page at a time, detect PictureItems, and
    caption each with GPT-4o Vision.

    Returns a list of FigureData (one per detected, captioned figure).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as fitz_doc:
        total_pages = fitz_doc.page_count

    figures: list[FigureData] = []
    converter = _get_figure_converter()

    for page_no in range(1, total_pages + 1):
        page_figure_count = 0
        try:
            result = converter.convert(
                str(pdf_path), page_range=(page_no, page_no)
            )
        except Exception as exc:
            log.warning("docling_page_error", page=page_no, error=str(exc))
            continue

        for item, _ in result.document.iterate_items():
            if not isinstance(item, PictureItem):
                continue

            img = item.get_image(doc=result.document)
            if img is None:
                continue

            w, h = img.size
            if w < MIN_FIGURE_PX or h < MIN_FIGURE_PX:
                continue

            caption = _caption_figure(img)
            if caption is None:
                continue

            figures.append(
                FigureData(
                    page=page_no,
                    figure_index=page_figure_count,
                    caption=caption,
                    width=w,
                    height=h,
                )
            )
            log.info(
                "figure_captioned",
                page=page_no,
                index=page_figure_count,
                size=f"{w}x{h}",
                caption=caption,
            )
            page_figure_count += 1

    return figures
