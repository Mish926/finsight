# Case study: profiling FinSight with AgentLens

FinSight is a multi-agent financial RAG system (QueryPlanner, Retriever, Critic, Synthesizer) that answers questions about SEC filings, built on the Groq API with no agent framework. This case study instruments it with AgentLens and profiles a 12-question run over three real filings (Apple, Amazon, JPMorgan, fiscal year 2025). FinSight's source was not modified: the harness in `casestudy/instrument_finsight.py` wraps the pipeline's Groq client and rebinds its agent methods from the outside, demonstrating that AgentLens can attach to a codebase you cannot edit.

Setup: 12 questions (7 single-company, 2 cross-company, 3 year-over-year), run sequentially with a 4 second pause between queries, on the Groq free tier. Planner and Critic run llama-3.1-8b-instant; the Synthesizer runs llama-3.3-70b-versatile. The success signal wired into `record_outcome` is FinSight's own per-answer output: citations present and self-reported confidence not LOW.

## The report

```
PER-AGENT COST ATTRIBUTION
  agent            calls    tokens          cost   share   avg ms  errs
  synthesizer         12    47,344       $0.0282  93.5%      957     0
  critic              12    33,281     $0.001724   5.7%    25615     0
  planner             12     4,021     $0.000225   0.7%     1640     0

REDUNDANCY
  exact duplicate calls : 0
  near-duplicate pairs  : 0
  est. wasted cost      : $0.000000

COST PER OUTCOME
  traces                : 12 (12 with recorded outcome)
  success rate          : 75.0%
  total cost            : $0.0302
  avg cost / trace      : $0.002514
  cost / SUCCESSFUL task: $0.003352
```

## Finding 1: 90 percent of latency was hiding inside the cheapest agent

Summing span latencies against total elapsed time, the Critic accounted for 307 of 342 seconds of wall-clock time (90 percent), averaging 25.6 seconds per call on the 8B model, while the 70B Synthesizer averaged 0.96 seconds. An 8B model being 27x slower than a 70B model cannot be inference time. The per-call breakdown shows what it is: the first Critic call completed in 1.1 seconds (normal 8B latency), and every subsequent call took 20 to 37 seconds.

The stalls are consistent with Groq's rolling tokens-per-minute budget. Planner and Critic share the 8B model and together send roughly 3,100 tokens per query; at the observed pace of about one query per 30 seconds, that is roughly 6,200 tokens per minute against the free tier's 6,000 TPM budget for that model, so sustained throughput sat at the ceiling and every call after the first waited for budget. The Synthesizer's 70B traffic stayed under its separate budget, which is why it never stalled despite larger prompts.

The notable part: FinSight has its own rate-limit retry logging, and it never fired during this run. The Groq SDK absorbs 429 responses with internal retries before application-level retry logic ever sees them, so the stalls were invisible to the application's own instrumentation. They only became visible through client-level interception, which measures the full duration of each call including whatever the SDK does inside it.

## Finding 2: cost and latency point at different agents

93.5 percent of spend goes to the Synthesizer; 90 percent of time goes to the Critic. "Optimize the expensive agent" and "optimize the slow agent" are different projects here, and only measurement reveals which one users actually experience. The Critic is also the reason the 8B budget is exhausted: 89 percent of all 8B-model tokens in the run were Critic prompts, roughly 2,600 tokens of retrieved context per call, producing a verdict of about 165 tokens. Reducing how much context the Critic reads is the highest-leverage latency change available, ahead of any model or infrastructure change.

## Finding 3: the Critic disagrees with the final confidence on most queries

The Critic judged the evidence insufficient (confidence LOW) on 10 of 12 queries, yet 9 of the 12 final answers were emitted with HIGH confidence. Whether the Critic's pessimistic verdicts still usefully shape synthesis is now a measurable question rather than a design assumption. The harness makes the ablation cheap without editing FinSight: rebind `critic.run` to a constant verdict, re-run the same 12 questions, and compare success rates and answer quality against this baseline.

## Finding 4: baseline economics

The full 12-question run cost $0.030: $0.0025 per query, $0.0034 per successful answer. The Planner is effectively free at 0.7 percent of spend. The Retriever recorded zero spans because FinSight's query expansion is a curated financial-terminology mapping rather than an LLM call, so retrieval costs no tokens by design. The redundancy report found zero exact or near-duplicate calls, confirming the pipeline does not repeat work.

## Limitations

The success signal is FinSight's self-reported confidence plus citation presence, not independently verified correctness; a manual audit of answers is the right next step before treating the 75 percent as an accuracy number. The TPM explanation for Finding 1 is inferred from the timing pattern and token arithmetic, not from provider-side confirmation; logging the SDK's retry headers would confirm it directly. Twelve questions is enough to expose structural patterns (latency attribution, token shares) but not enough for tight estimates of success rate.

## Reproducing

```
python casestudy/run_casestudy.py \
    --finsight /path/to/finsight \
    --pdf APPLE.pdf --pdf AMAZON.pdf --pdf JPM.pdf \
    --questions casestudy/questions_2025.json \
    --fresh
```
