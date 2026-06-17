# app/generation/generator.py
from openai import OpenAI
from app.config import get_settings
from app.generation.prompts import build_qa_prompt
from app.retrieval.searcher import RetrievedChunk
from dataclasses import dataclass
from typing import Iterator

settings = get_settings()
client = OpenAI()


@dataclass
class GenerationResult:
    answer: str
    sources: list[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


# GPT-4o-mini pricing (as of 2024)
COST_PER_INPUT_TOKEN = 0.00000015  # $0.15 / 1M
COST_PER_OUTPUT_TOKEN = 0.00000060  # $0.60 / 1M


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> GenerationResult:
    prompt = build_qa_prompt(question, chunks)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic — crucial for eval reproducibility
    )

    usage = response.usage
    cost = (
        usage.prompt_tokens * COST_PER_INPUT_TOKEN + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
    )

    return GenerationResult(
        answer=response.choices[0].message.content,
        sources=chunks,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=cost,
    )


def stream_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> Iterator[str]:
    """
    Streaming version for the API endpoint.
    Yields answer tokens as they arrive.
    """
    prompt = build_qa_prompt(question, chunks)

    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
