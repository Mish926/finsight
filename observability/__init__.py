"""AgentLens: framework-agnostic cost observability for multi-agent LLM pipelines.

Per-agent cost attribution, redundant-call detection, and cost-per-outcome
reporting for any pipeline built on an OpenAI-compatible client (OpenAI,
Groq, or anything matching that API shape).
"""

from .analysis import agent_attribution, cost_per_outcome, redundancy_report
from .lens import Lens
from .pricing import register_pricing
from .storage import Storage

__version__ = "0.1.0"

__all__ = [
    "Lens",
    "Storage",
    "register_pricing",
    "agent_attribution",
    "redundancy_report",
    "cost_per_outcome",
]
