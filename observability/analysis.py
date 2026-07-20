"""Analysis over recorded traces.

Three reports, each answering a question generic tracing tools don't:

1. agent_attribution  -- which agent is spending the money?
2. redundancy_report  -- which calls are duplicated or near-duplicated
                         within a single trace (wasted spend)?
3. cost_per_outcome   -- what does one *successful* task cost, not one
                         call? Requires outcomes via Lens.record_outcome.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

from .storage import Storage

_WORD_RE = re.compile(r"[a-z0-9]+")


def _shingles(text: str, k: int = 2) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def agent_attribution(storage: Storage, app: Optional[str] = None) -> list[dict[str, Any]]:
    """Per-agent rollup: calls, tokens, cost, latency, share of total cost."""
    where = "WHERE t.app = ?" if app else ""
    params = (app,) if app else ()
    rows = storage.query(
        f"""
        SELECT s.agent,
               COUNT(*)                  AS calls,
               SUM(s.prompt_tokens)      AS prompt_tokens,
               SUM(s.completion_tokens)  AS completion_tokens,
               SUM(s.cost_usd)           AS cost_usd,
               AVG(s.latency_ms)         AS avg_latency_ms,
               SUM(CASE WHEN s.error IS NOT NULL THEN 1 ELSE 0 END) AS errors
        FROM spans s JOIN traces t ON t.trace_id = s.trace_id
        {where}
        GROUP BY s.agent
        ORDER BY cost_usd DESC
        """,
        params,
    )
    result = [dict(r) for r in rows]
    total_cost = sum(r["cost_usd"] or 0 for r in result) or 1e-12
    total_tokens = sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in result) or 1
    for r in result:
        r["cost_share"] = (r["cost_usd"] or 0) / total_cost
        r["token_share"] = ((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)) / total_tokens
    return result


def redundancy_report(
    storage: Storage,
    app: Optional[str] = None,
    near_dup_threshold: float = 0.6,
    same_agent_only: bool = True,
) -> dict[str, Any]:
    """Find exact and near-duplicate LLM calls within each trace.

    Exact duplicates: identical prompt_hash within one trace (the same
    messages sent twice -- pure waste unless intentional retry).
    Near-duplicates: word-shingle Jaccard similarity above threshold
    between prompt previews within one trace (e.g. a Planner emitting two
    sub-questions so similar the Retriever+Critic run twice for one
    answer). By default only same-agent pairs are compared: every agent
    in a pipeline legitimately references the same user question, so
    cross-agent similarity is expected, not waste. Pass
    same_agent_only=False to see cross-agent pairs anyway. Deliberately
    dependency-free; an embedding backend can be swapped in later without
    changing the report shape.
    """
    where = "WHERE t.app = ?" if app else ""
    params = (app,) if app else ()
    rows = storage.query(
        f"""
        SELECT s.trace_id, s.span_id, s.agent, s.prompt_hash, s.prompt_preview,
               s.cost_usd, s.prompt_tokens, s.completion_tokens
        FROM spans s JOIN traces t ON t.trace_id = s.trace_id
        {where}
        ORDER BY s.trace_id, s.started_at
        """,
        params,
    )

    by_trace: dict[str, list] = defaultdict(list)
    for r in rows:
        by_trace[r["trace_id"]].append(r)

    exact_dups: list[dict[str, Any]] = []
    near_dups: list[dict[str, Any]] = []
    wasted_cost = 0.0

    for trace_id, spans in by_trace.items():
        seen_hashes: dict[str, Any] = {}
        for s in spans:
            h = s["prompt_hash"]
            if h and h in seen_hashes:
                exact_dups.append(
                    {
                        "trace_id": trace_id,
                        "agent": s["agent"],
                        "span_id": s["span_id"],
                        "duplicate_of": seen_hashes[h]["span_id"],
                        "cost_usd": s["cost_usd"],
                    }
                )
                wasted_cost += s["cost_usd"] or 0
            elif h:
                seen_hashes[h] = s

        # near-duplicates (skip pairs already flagged as exact)
        shingled = [(s, _shingles(s["prompt_preview"] or "")) for s in spans]
        for i in range(len(shingled)):
            for j in range(i + 1, len(shingled)):
                s1, sh1 = shingled[i]
                s2, sh2 = shingled[j]
                if same_agent_only and s1["agent"] != s2["agent"]:
                    continue
                if s1["prompt_hash"] and s1["prompt_hash"] == s2["prompt_hash"]:
                    continue
                sim = _jaccard(sh1, sh2)
                if sim >= near_dup_threshold:
                    near_dups.append(
                        {
                            "trace_id": trace_id,
                            "agents": (s1["agent"], s2["agent"]),
                            "span_ids": (s1["span_id"], s2["span_id"]),
                            "similarity": round(sim, 3),
                            "cost_usd": s2["cost_usd"],
                        }
                    )

    return {
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "estimated_wasted_cost_usd": wasted_cost,
    }


def cost_per_outcome(storage: Storage, app: Optional[str] = None) -> dict[str, Any]:
    """Cost per successful task -- the number that actually matters.

    Total spend is divided by *successful* traces only: failed traces'
    spend is real money that produced nothing, so it belongs in the
    numerator, not filtered out."""
    where = "WHERE t.app = ?" if app else ""
    params = (app,) if app else ()
    rows = storage.query(
        f"""
        SELECT t.trace_id, t.outcome_success,
               COALESCE(SUM(s.cost_usd), 0)  AS cost_usd,
               COALESCE(SUM(s.prompt_tokens + s.completion_tokens), 0) AS tokens,
               COUNT(s.span_id) AS calls
        FROM traces t LEFT JOIN spans s ON s.trace_id = t.trace_id
        {where}
        GROUP BY t.trace_id
        """,
        params,
    )
    total = len(rows)
    with_outcome = [r for r in rows if r["outcome_success"] is not None]
    successes = [r for r in with_outcome if r["outcome_success"] == 1]
    failures = [r for r in with_outcome if r["outcome_success"] == 0]
    total_cost = sum(r["cost_usd"] for r in rows)
    outcome_cost = sum(r["cost_usd"] for r in with_outcome)

    return {
        "traces_total": total,
        "traces_with_outcome": len(with_outcome),
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": (len(successes) / len(with_outcome)) if with_outcome else None,
        "total_cost_usd": total_cost,
        "avg_cost_per_trace_usd": (total_cost / total) if total else 0.0,
        "cost_per_successful_outcome_usd": (outcome_cost / len(successes)) if successes else None,
        "avg_calls_per_trace": (sum(r["calls"] for r in rows) / total) if total else 0.0,
    }
