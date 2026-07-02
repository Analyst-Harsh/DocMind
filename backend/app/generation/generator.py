# app/generation/generator.py
from collections.abc import Iterator
from dataclasses import dataclass

from openai import OpenAI

from app.config import get_settings
from app.generation.prompts import build_qa_prompt, get_registry
from app.retrieval.searcher import RetrievedChunk
from app.tracing.spans import traced_span

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

    with traced_span(
        "final-answer",
        as_type="generation",
        model=settings.llm_model,
        input=prompt,
    ) as span:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # deterministic — crucial for eval reproducibility
        )

        if response.usage is None:
            raise RuntimeError("OpenAI response missing usage data")
        usage = response.usage

        answer = response.choices[0].message.content
        if answer is None:
            raise RuntimeError("OpenAI response missing answer content")

        cost = (
            usage.prompt_tokens * COST_PER_INPUT_TOKEN
            + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
        )

        span.update(
            output=answer,
            usage_details={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
            cost_details={"total": cost},
            metadata={"prompt_version": "v1_grounded_qa"},
        )

    return GenerationResult(
        answer=answer,
        sources=chunks,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=cost,
    )


def generate_partial_answer(
    question: str,
    chunks: list[RetrievedChunk],
    missing_aspects: list[str],
) -> GenerationResult:
    prompt = get_registry().render(
        "partial_answer",
        question=question,
        chunks=chunks,
        missing_aspects=missing_aspects,
    )

    with traced_span(
        "final-answer",
        as_type="generation",
        model=settings.llm_model,
        input=prompt,
    ) as span:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        if response.usage is None:
            raise RuntimeError("OpenAI response missing usage data")
        usage = response.usage

        answer = response.choices[0].message.content
        if answer is None:
            raise RuntimeError("OpenAI response missing answer content")

        cost = (
            usage.prompt_tokens * COST_PER_INPUT_TOKEN
            + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
        )

        span.update(
            output=answer,
            usage_details={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
            cost_details={"total": cost},
            metadata={"prompt_version": "v1_partial_answer"},
        )

    return GenerationResult(
        answer=answer,
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
