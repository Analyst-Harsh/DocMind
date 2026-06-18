from .base_chunker import BaseChunker, ChunkStrategy
from .fixed_size_chunker import FixedSizeChunker
from .recursive_chunker import RecursiveChunker
from .structure_aware_chunker import StructureAwareChunker

CHUNKER_REGISTRY: dict[ChunkStrategy, type[BaseChunker]] = {
    ChunkStrategy.FIXED_SIZE: FixedSizeChunker,
    ChunkStrategy.RECURSIVE: RecursiveChunker,
    ChunkStrategy.STRUCTURE_AWARE: StructureAwareChunker,
}


def get_chunker(
    strategy: ChunkStrategy, chunk_size: int = 500, chunk_overlap: int = 50
) -> BaseChunker:
    return CHUNKER_REGISTRY[strategy](
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
