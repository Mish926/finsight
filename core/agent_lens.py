"""
AgentLens — lightweight per-agent observability for FinSight.

Tracks token usage, latency, and estimated cost for every LLM call made
during a query, broken down by agent. This is what turns "I built a
4-agent pipeline" into "I built a 4-agent pipeline and I can tell you
exactly what each agent costs and where the latency goes" — the
production question interviewers actually ask.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

# Groq published pricing for Llama 3.1 8B Instant, per 1M tokens (USD).
# The free tier itself costs $0 — this constant lets FinSight report what
# the pipeline WOULD cost on paid infra, which is the number that matters
# in a production conversation.
PRICE_PER_1M_INPUT_TOKENS = 0.05
PRICE_PER_1M_OUTPUT_TOKENS = 0.08


@dataclass
class AgentCall:
    agent: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
            + self.completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
        )


class AgentLens:
    """Collects AgentCall records for a single query and summarizes them."""

    def __init__(self):
        self.calls: List[AgentCall] = []

    def reset(self) -> None:
        self.calls = []

    def record(
        self,
        agent: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
    ) -> None:
        self.calls.append(
            AgentCall(agent, prompt_tokens, completion_tokens, latency_seconds)
        )

    def summary(self) -> Dict:
        """Per-agent + total breakdown for the most recent query."""
        by_agent: Dict[str, Dict] = {}
        for call in self.calls:
            entry = by_agent.setdefault(
                call.agent,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": 0.0,
                    "cost_usd": 0.0,
                },
            )
            entry["calls"] += 1
            entry["prompt_tokens"] += call.prompt_tokens
            entry["completion_tokens"] += call.completion_tokens
            entry["total_tokens"] += call.total_tokens
            entry["latency_seconds"] += call.latency_seconds
            entry["cost_usd"] += call.cost_usd

        for entry in by_agent.values():
            entry["latency_seconds"] = round(entry["latency_seconds"], 3)
            entry["cost_usd"] = round(entry["cost_usd"], 6)

        totals = {
            "total_tokens": sum(c.total_tokens for c in self.calls),
            "total_cost_usd": round(sum(c.cost_usd for c in self.calls), 6),
            "total_latency_seconds": round(sum(c.latency_seconds for c in self.calls), 3),
        }

        return {"by_agent": by_agent, "totals": totals}
