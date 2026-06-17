# app/ingestion/chunker.py
from dataclasses import dataclass
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


class FixedSizeChunker:
    """
    Splits text into overlapping chunks by token count.
    Uses tiktoken for accurate token counting (matches what OpenAI will bill you).
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, doc: Document) -> list[Chunk]:
        tokens = self.encoder.encode(doc.text)
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{chunk_index}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    chunk_index=chunk_index,
                    doc_type=doc.doc_type,
                    source_path=doc.source_path,
                    tags=doc.tags,
                )
            )

            # Move forward by chunk_size - overlap
            # so consecutive chunks share `chunk_overlap` tokens
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            print(f"  {doc.doc_id}: {len(doc_chunks)} chunks")
        return all_chunks
