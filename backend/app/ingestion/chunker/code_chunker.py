import re

from app.ingestion.loader import Document

from .base_chunker import DEFAULT_SEPARATORS, BaseChunker, Chunk, ChunkStrategy

# Matches a keyword separator like "\nclass " or "\ndef " -- as opposed to
# purely-whitespace separators ("\n\n", "\n", " ") or punctuation ones
# (". "). Only keyword separators get reattached; see _split_recursive.
_KEYWORD_SEPARATOR = re.compile(r"^\n[A-Za-z_]+ $")

# Per-language separator hierarchies, most structurally significant first.
# _split_recursive tries each separator in order, recursing into the next
# one for any piece still over chunk_size -- so putting class/function
# keywords first keeps a function body intact whenever it fits the token
# budget, instead of falling straight to blank-line/word splitting like
# RecursiveChunker's prose-oriented DEFAULT_SEPARATORS would.
LANGUAGE_SEPARATORS: dict[str, list[str]] = {
    "python": ["\nclass ", "\ndef ", "\n\tdef ", "\n    def ", "\n\n", "\n", " "],
    "javascript": [
        "\nclass ",
        "\nfunction ",
        "\nexport ",
        "\nconst ",
        "\n\n",
        "\n",
        " ",
    ],
    "typescript": [
        "\nclass ",
        "\nfunction ",
        "\nexport ",
        "\nconst ",
        "\ninterface ",
        "\n\n",
        "\n",
        " ",
    ],
    "go": ["\nfunc ", "\ntype ", "\n\n", "\n", " "],
    "rust": ["\nfn ", "\nimpl ", "\npub fn ", "\npub struct ", "\n\n", "\n", " "],
    "java": ["\nclass ", "\npublic ", "\nprivate ", "\nprotected ", "\n\n", "\n", " "],
    "csharp": [
        "\nclass ",
        "\npublic ",
        "\nprivate ",
        "\nprotected ",
        "\n\n",
        "\n",
        " ",
    ],
    "kotlin": ["\nclass ", "\nfun ", "\n\n", "\n", " "],
    "scala": ["\nclass ", "\nobject ", "\ndef ", "\n\n", "\n", " "],
    "ruby": ["\nclass ", "\nmodule ", "\ndef ", "\n\n", "\n", " "],
    "php": ["\nclass ", "\nfunction ", "\n\n", "\n", " "],
    "swift": ["\nclass ", "\nstruct ", "\nfunc ", "\n\n", "\n", " "],
    "c": ["\n\n", "\n", " "],
    "cpp": ["\nclass ", "\n\n", "\n", " "],
    "shell": ["\n\n", "\n", " "],
    "sql": ["\n\n", "\n", " "],
    "markdown": ["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    "text": ["\n\n", "\n", ". ", " "],
    "yaml": ["\n\n", "\n", " "],
    "toml": ["\n\n", "\n", " "],
    "html": ["\n\n", "\n", " "],
    # No keyword hierarchy (selectors aren't a fixed vocabulary like
    # class/def) -- split after a rule's closing brace first, so a rule
    # stays intact whenever it fits the token budget, before falling back
    # to blank-line/word splitting.
    "css": ["\n}\n", "\n\n", "\n", " "],
    "scss": ["\n}\n", "\n\n", "\n", " "],
}


class CodeChunker(BaseChunker):
    """
    Recursive chunking with language-aware separator hierarchies, so
    functions/classes stay intact instead of splitting on the prose-
    oriented boundaries RecursiveChunker uses. Reuses BaseChunker's
    _split_recursive/_merge_pieces -- only separator selection differs.
    """

    strategy_name = ChunkStrategy.CODE

    def chunk_document(self, doc: Document) -> list[Chunk]:
        separators = LANGUAGE_SEPARATORS.get(
            doc.language or "", DEFAULT_SEPARATORS
        )
        pieces = self._split_recursive(doc.text, separators)
        merged = self._merge_pieces(pieces)
        return [self._make_chunk(doc, text, i) for i, text in enumerate(merged)]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        Same recursive strategy as BaseChunker (try each separator, recurse
        into the next one for oversized pieces), but reattaches keyword
        separators ("\\nclass ", "\\ndef ", ...) to the piece that follows
        instead of letting str.split() consume them. Plain str.split()
        would strip "def "/"class "/"function " off the front of every
        piece after the first -- exactly the token that makes a chunk's
        role legible to BM25 and to a human reading a citation, undermining
        the reason to have language-aware separators at all.
        """
        if not text.strip():
            return []

        if len(self.encoder.encode(text)) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        sep, *rest = separators
        reattach = bool(_KEYWORD_SEPARATOR.match(sep))
        prefix = sep.lstrip("\n") if reattach else ""

        result = []
        for i, piece in enumerate(text.split(sep)):
            if not piece.strip():
                continue
            if reattach and i > 0:
                piece = prefix + piece
            if len(self.encoder.encode(piece)) <= self.chunk_size:
                result.append(piece)
            else:
                result.extend(self._split_recursive(piece, rest))
        return result
