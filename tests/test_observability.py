"""Tests for AgentLens core: wrapping, attribution, redundancy, outcomes.

Uses a fake OpenAI-compatible client so tests run offline and free.
"""

import threading

import pytest

from observability import Lens, agent_attribution, cost_per_outcome, redundancy_report
from observability.pricing import estimate_cost, register_pricing


# -- fake OpenAI/Groq-compatible client -----------------------------------

class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(self, model, prompt_tokens=100, completion_tokens=50):
        self.model = model
        self.usage = _Usage(prompt_tokens, completion_tokens)
        self.choices = [{"message": {"content": "ok"}}]


class _Completions:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return _Response(kwargs.get("model", "llama-3.1-8b-instant"))


class _Chat:
    def __init__(self, fail=False):
        self.completions = _Completions(fail=fail)


class FakeClient:
    def __init__(self, fail=False):
        self.chat = _Chat(fail=fail)
        self.api_key = "fake"  # arbitrary passthrough attribute


@pytest.fixture
def lens():
    return Lens(app="test", db_path=":memory:")


def _ask(client, model="llama-3.1-8b-instant", content="What was Apple's FY2024 revenue?"):
    return client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}]
    )


def test_wrap_passes_through_and_records(lens):
    client = lens.wrap(FakeClient())
    assert client.api_key == "fake"  # proxy transparency
    with lens.trace(name="q1"):
        with lens.agent("planner"):
            resp = _ask(client)
    assert resp.usage.prompt_tokens == 100
    rows = agent_attribution(lens.storage, app="test")
    assert len(rows) == 1
    assert rows[0]["agent"] == "planner"
    assert rows[0]["calls"] == 1
    assert rows[0]["prompt_tokens"] == 100
    assert rows[0]["cost_usd"] > 0


def test_attribution_across_agents(lens):
    client = lens.wrap(FakeClient())
    with lens.trace(name="q1"):
        with lens.agent("planner"):
            _ask(client, content="decompose this question please")
        with lens.agent("synthesizer"):
            _ask(client, model="llama-3.3-70b-versatile", content="write the final answer now")
    rows = {r["agent"]: r for r in agent_attribution(lens.storage, app="test")}
    assert set(rows) == {"planner", "synthesizer"}
    # 70B model costs more than 8B at equal tokens
    assert rows["synthesizer"]["cost_usd"] > rows["planner"]["cost_usd"]
    assert abs(sum(r["cost_share"] for r in rows.values()) - 1.0) < 1e-9


def test_unattributed_and_orphan_trace(lens):
    client = lens.wrap(FakeClient())
    _ask(client)  # no trace, no agent
    rows = agent_attribution(lens.storage, app="test")
    assert rows[0]["agent"] == "unattributed"
    traces = lens.storage.query("SELECT * FROM traces")
    assert len(traces) == 1 and traces[0]["name"] == "(orphan)"


def test_exact_duplicate_detection(lens):
    client = lens.wrap(FakeClient())
    with lens.trace(name="q1"):
        with lens.agent("retriever"):
            _ask(client, content="find revenue for Apple fiscal 2024")
            _ask(client, content="find revenue for Apple fiscal 2024")  # identical
    red = redundancy_report(lens.storage, app="test")
    assert len(red["exact_duplicates"]) == 1
    assert red["estimated_wasted_cost_usd"] > 0


def test_near_duplicate_detection(lens):
    client = lens.wrap(FakeClient())
    with lens.trace(name="q1"):
        with lens.agent("retriever"):
            _ask(client, content="what was the total net revenue reported by Apple for fiscal year 2024 in the annual filing")
            _ask(client, content="what was the total net revenue reported by Apple for the fiscal year 2024 annual filing")
    red = redundancy_report(lens.storage, app="test")
    assert len(red["near_duplicates"]) >= 1


def test_no_false_dup_across_traces(lens):
    client = lens.wrap(FakeClient())
    for name in ("q1", "q2"):
        with lens.trace(name=name):
            with lens.agent("planner"):
                _ask(client, content="same question both times")
    red = redundancy_report(lens.storage, app="test")
    # same prompt in *different* traces is legitimate reuse, not waste
    assert len(red["exact_duplicates"]) == 0


def test_cost_per_outcome(lens):
    client = lens.wrap(FakeClient())
    ids = []
    for name in ("q1", "q2", "q3"):
        with lens.trace(name=name) as tid:
            with lens.agent("synthesizer"):
                _ask(client, content=f"answer {name}")
            ids.append(tid)
    lens.record_outcome(ids[0], success=True)
    lens.record_outcome(ids[1], success=True)
    lens.record_outcome(ids[2], success=False, meta={"reason": "verifier_failed"})
    cpo = cost_per_outcome(lens.storage, app="test")
    assert cpo["traces_total"] == 3
    assert cpo["successes"] == 2 and cpo["failures"] == 1
    assert cpo["success_rate"] == pytest.approx(2 / 3)
    # failed trace's spend is included in numerator: cost/success > avg cost/trace
    assert cpo["cost_per_successful_outcome_usd"] > cpo["avg_cost_per_trace_usd"]


def test_error_recorded_and_reraised(lens):
    client = lens.wrap(FakeClient(fail=True))
    with pytest.raises(RuntimeError):
        with lens.trace(name="q1"):
            with lens.agent("planner"):
                _ask(client)
    rows = agent_attribution(lens.storage, app="test")
    assert rows[0]["errors"] == 1


def test_pricing_fallbacks():
    assert estimate_cost("totally-unknown-model", 1000, 1000) == 0.0
    register_pricing("totally-unknown-model", 1.0, 2.0)
    assert estimate_cost("totally-unknown-model", 1_000_000, 0) == pytest.approx(1.0)
    # prefix match for versioned model names
    assert estimate_cost("gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(2.50)


def test_thread_isolation(lens):
    """Two threads with different agents must not cross-attribute."""
    client = lens.wrap(FakeClient())
    errors = []

    def run(agent_name):
        try:
            with lens.trace(name=agent_name):
                with lens.agent(agent_name):
                    for _ in range(5):
                        _ask(client, content=f"work for {agent_name}")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run, args=(f"agent{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    rows = {r["agent"]: r["calls"] for r in agent_attribution(lens.storage, app="test")}
    assert rows == {f"agent{i}": 5 for i in range(4)}


def test_near_dup_ignores_cross_agent(lens):
    """Different agents referencing the same question is normal, not waste."""
    client = lens.wrap(FakeClient())
    q = "what was the total net revenue reported by Apple for fiscal year 2024"
    with lens.trace(name="q1"):
        with lens.agent("planner"):
            _ask(client, content="decompose this: " + q)
        with lens.agent("critic"):
            _ask(client, content="rate the evidence for: " + q)
    red = redundancy_report(lens.storage, app="test")
    assert len(red["near_duplicates"]) == 0
