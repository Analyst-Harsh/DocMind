from app.agent.state import AgentLoopState, SufficiencyResult


def test_sufficiency_result_defaults():
    r = SufficiencyResult(
        is_sufficient=False,
        reasoning="Missing the definition.",
        missing_aspects=["definition of chunking"],
        confidence="high",
    )
    assert r.is_sufficient is False
    assert r.missing_aspects == ["definition of chunking"]


def test_agent_loop_state_defaults():
    state = AgentLoopState(
        original_question="What is RAG?",
        current_query="What is RAG?",
    )
    assert state.iteration == 0
    assert state.accumulated_chunks == []
    assert state.sufficiency_history == []
    assert state.current_query == "What is RAG?"


def test_agent_loop_state_current_query_can_diverge():
    state = AgentLoopState(
        original_question="What is RAG?",
        current_query="retrieval augmented generation definition",
    )
    assert state.original_question != state.current_query


def test_sufficiency_history_is_independent_between_instances():
    s1 = AgentLoopState(original_question="q1", current_query="q1")
    s2 = AgentLoopState(original_question="q2", current_query="q2")
    s1.sufficiency_history.append(
        SufficiencyResult(is_sufficient=True, reasoning="ok", missing_aspects=[], confidence="high")
    )
    assert s2.sufficiency_history == [], "mutable default must not be shared"
