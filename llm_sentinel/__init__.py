"""llm_sentinel - A Python middleware library for detecting prompt injection and jailbreak attempts.

This package provides a drop-in proxy wrapper compatible with any Python LLM client
(OpenAI, Anthropic, LiteLLM) that intercepts prompts, scores them against a configurable
rule engine, logs detected threats, and optionally dispatches real-time alerts.

Public API
----------
- SentinelProxy   : The main proxy class that wraps any LLM callable.
- SentinelConfig  : Pydantic configuration model for the sentinel.
- ThreatLevel     : Enum representing LOW / MEDIUM / HIGH / CRITICAL severity.
- ThreatReport    : Data model representing the result of a detection run.
- DetectionResult : Lightweight result returned from individual rule checks.

Quick start
-----------
    from llm_sentinel import SentinelProxy, SentinelConfig

    config = SentinelConfig(block_threshold="HIGH")
    client = SentinelProxy(llm_callable=openai_chat_fn, config=config)
    response = client(messages=[{"role": "user", "content": "Hello!"}])
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("llm_sentinel")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# ---------------------------------------------------------------------------
# Deferred imports — populated in later phases.  These are declared here so
# that the public namespace is stable from phase 1 onward and IDEs can resolve
# the symbols.  The actual implementations live in the sub-modules.
# ---------------------------------------------------------------------------

__all__ = [
    "__version__",
    "SentinelProxy",
    "SentinelConfig",
    "ThreatLevel",
    "ThreatReport",
    "DetectionResult",
]


def __getattr__(name: str) -> object:
    """Lazy-load public symbols so early imports do not fail before all phases
    are implemented, while still raising an informative AttributeError for
    genuinely unknown names."""
    _public_map = {
        "SentinelProxy": ("llm_sentinel.proxy", "SentinelProxy"),
        "SentinelConfig": ("llm_sentinel.config", "SentinelConfig"),
        "ThreatLevel": ("llm_sentinel.config", "ThreatLevel"),
        "ThreatReport": ("llm_sentinel.detector", "ThreatReport"),
        "DetectionResult": ("llm_sentinel.detector", "DetectionResult"),
    }
    if name not in _public_map:
        raise AttributeError(f"module 'llm_sentinel' has no attribute {name!r}")

    module_path, attr = _public_map[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr)
