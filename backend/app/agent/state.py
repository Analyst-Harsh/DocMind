from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.searcher import RetrievedChunk


@dataclass
class SufficiencyResult:
    is_sufficient: bool
    reasoning: str
    missing_aspects: list[str]
    confidence: str  # "high" | "medium" | "low"
    cost_usd: float = 0.0


@dataclass
class AgentLoopState:
    original_question: str
    current_query: str
    accumulated_chunks: list[RetrievedChunk] = field(default_factory=list)
    iteration: int = 0
    sufficiency_history: list[SufficiencyResult] = field(default_factory=list)
    loop_cost: float = 0.0
    query_history: list[str] = field(default_factory=list)
    loop_terminated_by: str = "cap_reached"
