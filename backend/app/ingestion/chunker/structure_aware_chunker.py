import ast
import re

from app.ingestion.loader import Document

from .base_chunker import DEFAULT_SEPARATORS, BaseChunker, Chunk, ChunkStrategy


class StructureAwareChunker(BaseChunker):
    strategy_name = ChunkStrategy.STRUCTURE_AWARE

    MARKDOWN_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+.*$", re.MULTILINE)
    FENCED_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)

    def chunk_document(self, doc: Document) -> list[Chunk]:
        if doc.doc_type == "markdown":
            sections = self._split_markdown(doc.text)
        elif doc.doc_type == "code" and doc.language == "python":
            sections = self._split_python(doc.text)
        else:
            # PDFs and non-Python code have no reliable structural markers
            # yet (Week 4 adds proper PDF table/figure structure extraction)
            pieces = self._split_recursive(
                doc.text, self._get_default_separators()
            )
            merged = self._merge_pieces(pieces)
            return [
                self._make_chunk(doc, text, i) for i, text in enumerate(merged)
            ]

        merged = self._merge_sections(sections)
        return [self._make_chunk(doc, text, i) for i, text in enumerate(merged)]

    def _split_markdown(self, text: str) -> list[str]:
        # don't treat '#' inside fenced code blocks (e.g. Python comments
        # shown in a README snippet) as a markdown header
        code_spans = [
            (m.start(), m.end())
            for m in self.FENCED_CODE_BLOCK_PATTERN.finditer(text)
        ]

        def _inside_code_block(pos: int) -> bool:
            return any(start <= pos < end for start, end in code_spans)

        matches = [
            m
            for m in self.MARKDOWN_HEADER_PATTERN.finditer(text)
            if not _inside_code_block(m.start())
        ]

        if not matches:
            return [text]

        sections = []
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(preamble)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        return sections

    def _split_python(self, text: str) -> list[str]:
        """
        One section per top-level function/class, including decorators,
        docstring, and full body. Module-level code between definitions
        (imports, constants) becomes its own section.
        """
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [text]  # unparseable -> treat as plain text, don't crash

        lines = text.splitlines(keepends=True)
        sections, cursor = [], 0

        top_level_nodes = [
            node
            for node in tree.body
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        ]

        for node in top_level_nodes:
            node_start = node.lineno - 1
            if node.decorator_list:
                node_start = node.decorator_list[0].lineno - 1

            if node_start > cursor:
                preamble = "".join(lines[cursor:node_start]).strip()
                if preamble:
                    sections.append(preamble)

            # ast.parse (not compile()) always sets end_lineno on these node
            # types, so this is never actually None.
            assert node.end_lineno is not None
            section_text = "".join(lines[node_start : node.end_lineno])
            sections.append(section_text)
            cursor = node.end_lineno

        if cursor < len(lines):
            trailing = "".join(lines[cursor:]).strip()
            if trailing:
                sections.append(trailing)

        return sections

    def _merge_sections(self, sections: list[str]) -> list[str]:
        # any section too large to be its own chunk gets recursively split
        # first, so _merge_pieces never receives an oversized piece
        atomic_pieces = []
        for section in sections:
            if len(self.encoder.encode(section)) <= self.chunk_size:
                atomic_pieces.append(section)
            else:
                atomic_pieces.extend(
                    self._split_recursive(section, DEFAULT_SEPARATORS)
                )

        return self._merge_pieces(atomic_pieces)
