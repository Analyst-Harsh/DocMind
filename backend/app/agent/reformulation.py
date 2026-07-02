from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.generation.prompts import get_registry
from app.tracing.spans import traced_span

client = OpenAI()
settings = get_settings()

COST_PER_INPUT_TOKEN = 0.00000015  # $0.15 / 1M
COST_PER_OUTPUT_TOKEN = 0.00000060  # $0.60 / 1M


def reformulate_query(
    original_question: str,
    missing_aspects: list[str],
    previous_queries: list[str],
) -> tuple[str, float]:
    prompt = get_registry().render(
        "query_reformulation",
        original_question=original_question,
        missing_aspects=missing_aspects,
        previous_queries=previous_queries,
    )

    with traced_span(
        "query-reformulation",
        as_type="generation",
        model=settings.llm_model,
        input=prompt,
    ) as span:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                "OpenAI response missing content in query reformulation"
            )
        new_query = content.strip()
        usage = response.usage
        if usage is None:
            raise RuntimeError(
                "OpenAI response missing usage in query reformulation"
            )
        cost_usd = (
            usage.prompt_tokens * COST_PER_INPUT_TOKEN
            + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
        )
        span.update(output=new_query)

    return new_query, cost_usd
