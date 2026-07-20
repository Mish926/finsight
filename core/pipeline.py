"""
Pipeline Orchestrator - Groq version (free, fast, Llama 3)
"""

import os
import re
import time
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.agent_lens import AgentLens
from agents.planner import QueryPlannerAgent
from agents.retriever import RetrieverAgent
from agents.critic import CriticAgent
from agents.synthesizer import SynthesizerAgent

from observability import Lens as ObsLens

load_dotenv()

# Module-level so all session pipelines share one observability database.
# SQLite writes are lock-protected inside the observability package, so
# concurrent sessions are safe. Unlike the per-response agent_stats tracker
# (core/agent_lens.py, reset every query), this records persistently across
# all queries: per-agent cost attribution, redundancy detection, and
# cost-per-successful-answer. See profiling/CASESTUDY.md for findings.
obs = ObsLens(app="finsight", db_path="data/agentlens.db")


class GroqModel:
    """Wrapper so all agents work with Groq seamlessly.

    Optionally instrumented with an AgentLens: every call records token
    usage and latency, tagged with whichever agent is currently active
    (set by the pipeline via `current_agent` before each agent runs).

    Optionally has a fallback_model_name: if the primary model call fails
    (e.g. a stronger model is temporarily unavailable, deprecated, or the
    account doesn't have access), automatically retries once with the
    fallback instead of crashing the whole query.

    Groq's "tokens per minute" limit is a ROLLING budget across ALL calls
    to a model within a 60-second window, not a per-request cap -- so even
    when every individual request is well within size limits, several
    queries run back-to-back (each making a Planner + Critic call) can
    cumulatively exceed it. Rate-limit errors are retried with backoff
    (using Groq's suggested wait time when it provides one) instead of
    immediately failing the whole query.

    NOTE (measured, not assumed): the Groq SDK also absorbs some 429s with
    its own internal retries BEFORE the retry logic below ever sees them.
    Those silent stalls are invisible to this class's logging and only
    show up in the observability layer's per-span latency, because the
    wrapped client measures the full duration of each create() call
    including whatever the SDK does inside it. In the profiled 12-question
    run this accounted for 90% of end-to-end latency, concentrated in the
    Critic (see profiling/CASESTUDY.md).
    """
    MAX_RATE_LIMIT_RETRIES = 3

    def __init__(self, client, model_name="llama-3.1-8b-instant", lens: AgentLens = None,
                 fallback_model_name: str = None):
        self.client = client
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.lens = lens
        self.current_agent = "unknown"

    def generate_content(self, prompt: str):
        try:
            return self._call_with_retry(self.model_name, prompt)
        except Exception as e:
            if self.fallback_model_name:
                print(f"[GroqModel] '{self.model_name}' failed ({e}); "
                      f"falling back to '{self.fallback_model_name}'")
                return self._call_with_retry(self.fallback_model_name, prompt)
            raise

    def _call_with_retry(self, model_name: str, prompt: str):
        last_error = None
        for attempt in range(self.MAX_RATE_LIMIT_RETRIES):
            try:
                return self._call(model_name, prompt)
            except Exception as e:
                last_error = e
                wait = self._rate_limit_wait_seconds(e)
                if wait is None:
                    raise  # not a rate-limit error -- don't retry, propagate immediately
                print(f"[GroqModel] Rate limited on '{model_name}' "
                      f"(attempt {attempt + 1}/{self.MAX_RATE_LIMIT_RETRIES}), "
                      f"waiting {wait:.1f}s before retry...")
                time.sleep(wait)
        raise last_error

    def _rate_limit_wait_seconds(self, error: Exception):
        """Return seconds to wait before retrying, or None if this isn't a
        rate-limit error (in which case the caller should not retry)."""
        msg = str(error)
        if "rate_limit_exceeded" not in msg and "429" not in msg and "413" not in msg:
            return None
        # Groq's error message sometimes includes a suggested wait, e.g.
        # "...try again in 2.5s..." -- use it if present, else a safe default.
        match = re.search(r'try again in (\d+(?:\.\d+)?)s', msg)
        if match:
            return float(match.group(1)) + 0.5  # small safety margin
        return 5.0

    def _call(self, model_name: str, prompt: str):
        start = time.time()
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            # frequency_penalty discourages the model from repeating the same
            # tokens/phrases -- without it, a model that gets stuck on a hard
            # reasoning problem (e.g. a genuinely ambiguous table) can spiral
            # into repeating "the answer is X, no wait, not X..." until it
            # hits max_tokens, producing a long, useless, repetitive answer
            # instead of either a real answer or a clean "I don't know".
            frequency_penalty=0.4
        )
        latency = time.time() - start
        text = response.choices[0].message.content

        if self.lens is not None:
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self.lens.record(self.current_agent, prompt_tokens, completion_tokens, latency)

        return type('R', (), {'text': text})()


class FinSightPipeline:
    def __init__(self, index_dir: str = "data/index"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        # obs.wrap intercepts every chat.completions.create call for the
        # observability layer. The wrapper is a transparent proxy, so the
        # retry/fallback logic in GroqModel is unaffected. A bonus over the
        # internal tracker: the wrapper records response.model -- the model
        # that ACTUALLY answered -- so 70B->8B fallback events are visible
        # in the cost data.
        client = obs.wrap(Groq(api_key=api_key))
        self.lens = AgentLens()
        # Planner and Critic just need to decompose questions and judge
        # evidence quality -- the cheap, fast model handles that fine.
        self.llm = GroqModel(client, model_name="llama-3.1-8b-instant", lens=self.lens)

        # Synthesizer is where the hard reasoning actually happens: reading
        # multi-year, multi-segment tables correctly, not confusing a
        # regional subtotal for a firm-wide total, not mixing up which
        # company a number belongs to. That reliability gap showed up
        # repeatedly on the 8B model even with careful prompting, so the
        # Synthesizer gets a stronger model -- falling back to the cheap
        # one automatically if the stronger model is ever unavailable,
        # rather than failing the whole query.
        self.synthesizer_llm = GroqModel(
            client,
            model_name="llama-3.3-70b-versatile",
            lens=self.lens,
            fallback_model_name="llama-3.1-8b-instant"
        )
        self.synthesizer_llm.current_agent = "Synthesizer"

        # Financial statement tables (e.g. "2024 | Change | 2023 | Change |
        # 2022" header rows followed by "Total net sales $391,035 2% $383,285
        # (3)% $394,328") need enough chunk size + overlap that the header
        # stays attached to its data row -- 500/100 was splitting them into
        # separate chunks, leaving the LLM a bare number with no year label.
        self.processor = DocumentProcessor(chunk_size=900, overlap=300)
        self.vector_store = VectorStore(index_dir=index_dir)

        self.planner = QueryPlannerAgent(self.llm)
        # top_k=24 gives per_source_k=8 with 3 documents indexed (24 // 3).
        # This isn't a guess: direct measurement against the real index
        # (debug_rank_check.py) showed the correct JPMorgan chunk for a
        # real query ranking #6 out of 2560 candidates -- a genuinely good
        # rank, but the previous top_k=15 (per_source_k=5) excluded it by
        # exactly one position, every time, regardless of exact phrasing.
        # This explains the specific flakiness observed: whenever a
        # sub-question happened to nudge the chunk to rank 5 or better it
        # worked, and failed silently otherwise.
        self.retriever = RetrieverAgent(self.vector_store, top_k=24)
        self.critic = CriticAgent(self.llm)
        self.synthesizer = SynthesizerAgent(self.synthesizer_llm, vector_store=self.vector_store)

        self.indexed_docs = []
        if self.vector_store.load():
            self.indexed_docs = self.vector_store.stats()["documents"]

    def index_document(self, pdf_path: str) -> dict:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        print(f"\nIndexing: {path.name}")
        chunks = self.processor.process(pdf_path)
        self.vector_store.add_chunks(chunks)
        self.vector_store.save()
        if path.name not in self.indexed_docs:
            self.indexed_docs.append(path.name)
        return {
            "filename": path.name,
            "chunks": len(chunks),
            "total_indexed": len(self.vector_store.chunks)
        }

    def remove_document(self, filename: str) -> dict:
        removed_count = self.vector_store.remove_document(filename)
        if removed_count == 0:
            raise ValueError(f"'{filename}' is not in the index")
        self.vector_store.save()
        if filename in self.indexed_docs:
            self.indexed_docs.remove(filename)
        return {
            "filename": filename,
            "chunks_removed": removed_count,
            "total_remaining": len(self.vector_store.chunks)
        }

    def query(self, question: str) -> dict:
        if self.vector_store.is_empty():
            return {"error": "No documents indexed. Please upload a PDF first.", "answer": None}

        self.lens.reset()
        start = time.time()

        with obs.trace(name=question) as trace_id:
            print(f"\n[1/4] QueryPlanner running...")
            self.llm.current_agent = "QueryPlanner"
            with obs.agent("planner"):
                sub_questions = self.planner.run(question)
            print(f"  Sub-questions: {sub_questions}")

            print(f"[2/4] Retriever running...")
            with obs.agent("retriever"):
                results, context = self.retriever.run(sub_questions)
            print(f"  Retrieved {len(results)} unique chunks")

            print(f"[3/4] Critic running...")
            self.llm.current_agent = "Critic"
            with obs.agent("critic"):
                verdict = self.critic.run(question, context, results)
            print(f"  Sufficient: {verdict.sufficient} | Confidence: {verdict.confidence}")

            print(f"[4/4] Synthesizer running...")
            with obs.agent("synthesizer"):
                result = self.synthesizer.run(question, context, results, verdict, sub_questions)

            # Outcome signal for cost-per-successful-answer: citations
            # present and self-reported confidence not low. Deliberately
            # conservative -- an uncited or low-confidence answer doesn't
            # count as a success even if it happens to be correct.
            success = bool(result.citations) and str(result.confidence).lower() != "low"
            obs.record_outcome(trace_id, success=success,
                               meta={"confidence": result.confidence,
                                     "n_citations": len(result.citations or [])})

        elapsed = round(time.time() - start, 2)
        print(f"\nDone in {elapsed}s")

        return {
            "question": question,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "sub_questions": result.sub_questions,
            "verdict": result.verdict,
            "elapsed_seconds": elapsed,
            "agent_stats": self.lens.summary(),
            "trace_id": trace_id
        }

    def get_stats(self) -> dict:
        return self.vector_store.stats()
