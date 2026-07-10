from __future__ import annotations

import json

from openai import OpenAI

from app.agent.state import SufficiencyResult
from app.config import get_settings
from app.generation.prompts import get_registry
from app.retrieval.searcher import RetrievedChunk
from app.tracing.spans import traced_span

client = OpenAI()
settings = get_settings()

COST_PER_INPUT_TOKEN = 0.00000015  # $0.15 / 1M
COST_PER_OUTPUT_TOKEN = 0.00000060  # $0.60 / 1M


def assess_sufficiency(
    question: str,
    chunks: list[RetrievedChunk],
) -> SufficiencyResult:
    prompt = get_registry().render(
        "sufficiency_assessment",
        question=question,
        chunks=chunks,
    )

    with traced_span(
        "sufficiency-assessment",
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
            raise RuntimeError("OpenAI response missing content in sufficiency assessment")
        try:
            data: dict = json.loads(content)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"Sufficiency assessment returned non-JSON: {content[:200]!r}",
                exc.doc,
                exc.pos,
            ) from exc

        usage = response.usage
        if usage is None:
            raise RuntimeError("OpenAI response missing usage in sufficiency assessment")
        cost = (
            usage.prompt_tokens * COST_PER_INPUT_TOKEN
            + usage.completion_tokens * COST_PER_OUTPUT_TOKEN
        )
        result = SufficiencyResult(
            is_sufficient=bool(data["is_sufficient"]),
            reasoning=str(data["reasoning"]),
            missing_aspects=list(data["missing_aspects"]),
            confidence=str(data["confidence"]),
            cost_usd=cost,
        )
        span.update(output=data)

    return result
