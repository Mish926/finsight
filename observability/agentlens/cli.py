"""AgentLens CLI.

    agentlens report [--db agentlens.db] [--app finsight]

Prints the three core reports as plain text. Dashboard comes later; the
CLI exists first because a report you can run in CI or paste into a case
study is more useful than a chart you can only screenshot.
"""

from __future__ import annotations

import argparse

from .analysis import agent_attribution, cost_per_outcome, redundancy_report
from .pricing import unpriced_models
from .storage import Storage


def _fmt_usd(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.6f}" if x < 0.01 else f"${x:,.4f}"


def report(db_path: str, app: str | None) -> str:
    storage = Storage(db_path)
    lines: list[str] = []
    scope = f" (app={app})" if app else ""
    lines.append(f"AgentLens report{scope}")
    lines.append("=" * 60)

    lines.append("\nPER-AGENT COST ATTRIBUTION")
    lines.append("-" * 60)
    rows = agent_attribution(storage, app)
    if not rows:
        lines.append("  no spans recorded")
    else:
        header = f"  {'agent':<16}{'calls':>6}{'tokens':>10}{'cost':>14}{'share':>8}{'avg ms':>9}{'errs':>6}"
        lines.append(header)
        for r in rows:
            tokens = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
            lines.append(
                f"  {r['agent']:<16}{r['calls']:>6}{tokens:>10,}"
                f"{_fmt_usd(r['cost_usd']):>14}{r['cost_share']:>7.1%}"
                f"{r['avg_latency_ms']:>9.0f}{r['errors']:>6}"
            )

    lines.append("\nREDUNDANCY")
    lines.append("-" * 60)
    red = redundancy_report(storage, app)
    lines.append(f"  exact duplicate calls : {len(red['exact_duplicates'])}")
    lines.append(f"  near-duplicate pairs  : {len(red['near_duplicates'])}")
    lines.append(f"  est. wasted cost      : {_fmt_usd(red['estimated_wasted_cost_usd'])}")
    for d in red["exact_duplicates"][:5]:
        lines.append(f"    [exact] {d['agent']} in {d['trace_id']}  cost {_fmt_usd(d['cost_usd'])}")
    for d in red["near_duplicates"][:5]:
        lines.append(
            f"    [near {d['similarity']:.0%}] {d['agents'][0]} vs {d['agents'][1]} in {d['trace_id']}"
        )

    lines.append("\nCOST PER OUTCOME")
    lines.append("-" * 60)
    cpo = cost_per_outcome(storage, app)
    lines.append(f"  traces                : {cpo['traces_total']} "
                 f"({cpo['traces_with_outcome']} with recorded outcome)")
    sr = cpo["success_rate"]
    lines.append(f"  success rate          : {sr:.1%}" if sr is not None else "  success rate          : n/a (no outcomes recorded)")
    lines.append(f"  total cost            : {_fmt_usd(cpo['total_cost_usd'])}")
    lines.append(f"  avg cost / trace      : {_fmt_usd(cpo['avg_cost_per_trace_usd'])}")
    lines.append(f"  cost / SUCCESSFUL task: {_fmt_usd(cpo['cost_per_successful_outcome_usd'])}")

    missing = unpriced_models()
    if missing:
        lines.append(f"\n  note: no pricing for models {sorted(missing)}; their cost shows as $0")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentlens")
    sub = parser.add_subparsers(dest="command", required=True)
    p_report = sub.add_parser("report", help="print cost/redundancy/outcome report")
    p_report.add_argument("--db", default="agentlens.db")
    p_report.add_argument("--app", default=None)
    args = parser.parse_args()
    if args.command == "report":
        print(report(args.db, args.app))


if __name__ == "__main__":
    main()
