"""Core data models shared across the llm_sentinel library.

This module defines:

- :class:`DetectionMatch`  – a single rule that fired during analysis.
- :class:`DetectionResult` – the lightweight per-rule check result.
- :class:`ThreatReport`    – the full, serialisable output of a detection run.
- :class:`ThreatDetectedError` – raised by SentinelProxy when a call is blocked.

All models are Pydantic v2 ``BaseModel`` subclasses to ensure consistent
validation, serialisation, and schema generation throughout the library.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from llm_sentinel.config import ThreatLevel


# ---------------------------------------------------------------------------
# Helper to generate a sortable, URL-safe unique ID
# ---------------------------------------------------------------------------

def _new_report_id() -> str:
    """Generate a random UUID4 string used as a ThreatReport identifier."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# DetectionMatch – one rule that fired
# ---------------------------------------------------------------------------

class DetectionMatch(BaseModel):
    """Represents a single rule that matched during prompt analysis.

    Attributes
    ----------
    rule_id:
        Identifier of the rule that fired (e.g. ``"BUILTIN-DAN-001"`` or a
        user-defined id).
    description:
        Human-readable explanation of what the rule detects.
    matched_text:
        The substring of the prompt that triggered the rule, if available.
        For heuristic checks this may be ``None``.
    score_contribution:
        How much this match added to the cumulative threat score (0.0 – 1.0).
    threat_level:
        The severity level assigned to this individual rule match.
    rule_type:
        One of ``"regex"``, ``"keyword"``, ``"heuristic"``, or ``"custom"``
        indicating the detection mechanism used.
    """

    rule_id: str = Field(..., description="Identifier of the matched rule")
    description: str = Field(default="", description="Rule description")
    matched_text: str | None = Field(
        default=None,
        description="Prompt substring that triggered the rule",
    )
    score_contribution: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score added by this match (0.0 – 1.0)",
    )
    threat_level: ThreatLevel = Field(
        default=ThreatLevel.LOW,
        description="Severity assigned to this individual rule",
    )
    rule_type: str = Field(
        default="regex",
        description="Detection mechanism: regex | keyword | heuristic | custom",
    )


# ---------------------------------------------------------------------------
# DetectionResult – lightweight per-rule evaluation outcome
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    """Lightweight outcome from evaluating a single rule against a prompt.

    This is the low-level building block returned by individual rule checkers
    inside :mod:`llm_sentinel.detector`.  Multiple ``DetectionResult`` objects
    are aggregated into a :class:`ThreatReport`.

    Attributes
    ----------
    fired:
        Whether the rule matched the prompt.
    match:
        The :class:`DetectionMatch` detail if *fired* is True; else ``None``.
    """

    fired: bool = Field(default=False, description="Whether this rule matched")
    match: DetectionMatch | None = Field(
        default=None,
        description="Match detail when the rule fired",
    )

    @classmethod
    def hit(
        cls,
        rule_id: str,
        description: str,
        matched_text: str | None,
        score_contribution: float,
        threat_level: ThreatLevel,
        rule_type: str = "regex",
    ) -> "DetectionResult":
        """Convenience factory for a *fired* result.

        Parameters
        ----------
        rule_id:
            Identifier of the rule that fired.
        description:
            Rule description.
        matched_text:
            Substring of the prompt that triggered the rule.
        score_contribution:
            Score delta contributed by this match.
        threat_level:
            Severity level for this match.
        rule_type:
            Detection mechanism string.

        Returns
        -------
        DetectionResult
            A result with ``fired=True`` and a populated ``match``.
        """
        return cls(
            fired=True,
            match=DetectionMatch(
                rule_id=rule_id,
                description=description,
                matched_text=matched_text,
                score_contribution=score_contribution,
                threat_level=threat_level,
                rule_type=rule_type,
            ),
        )

    @classmethod
    def miss(cls) -> "DetectionResult":
        """Convenience factory for a *non-firing* result."""
        return cls(fired=False, match=None)


# ---------------------------------------------------------------------------
# ThreatReport – the full output of a detection run
# ---------------------------------------------------------------------------

class ThreatReport(BaseModel):
    """Complete, serialisable report produced by the detection engine.

    A ``ThreatReport`` is generated for every prompt processed by the sentinel,
    regardless of whether a threat was detected.  Reports with
    ``threat_detected=False`` represent benign prompts and are logged at DEBUG
    level only when the log threshold permits.

    Attributes
    ----------
    report_id:
        UUID4 unique identifier for this report, useful for correlation.
    timestamp:
        UTC datetime at which the detection run completed.
    prompt:
        The original prompt text that was analysed.  May be ``None`` if the
        sentinel is configured with ``include_prompt_in_log=False``.
    prompt_length:
        Character count of the original (un-truncated) prompt.
    threat_detected:
        True when at least one rule fired or the cumulative score exceeds the
        configured score threshold.
    threat_level:
        Highest severity level among all matched rules.  Defaults to ``LOW``
        even for benign prompts (the proxy uses ``threat_detected`` to gate
        actions, not the level alone).
    cumulative_score:
        Sum of ``score_contribution`` values from all fired rules, capped at
        1.0.  Provides a continuous measure of threat intensity.
    matches:
        Ordered list of :class:`DetectionMatch` objects for every rule that
        fired, sorted by descending ``score_contribution``.
    rules_evaluated:
        Total number of rules that were tested against the prompt.
    caller_metadata:
        Copy of :attr:`SentinelConfig.caller_metadata` at the time of the run,
        plus any per-call metadata supplied by the proxy.
    extra:
        Arbitrary additional context (e.g. model name, request id) attached
        at call time.
    """

    report_id: str = Field(
        default_factory=_new_report_id,
        description="Unique UUID4 identifier for this detection report",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of when this report was produced",
    )
    prompt: str | None = Field(
        default=None,
        description="The analysed prompt text (may be absent for privacy)",
    )
    prompt_length: int = Field(
        default=0,
        ge=0,
        description="Character length of the original prompt",
    )
    threat_detected: bool = Field(
        default=False,
        description="Whether a threat was detected in the prompt",
    )
    threat_level: ThreatLevel = Field(
        default=ThreatLevel.LOW,
        description="Highest severity level among all matched rules",
    )
    cumulative_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated threat score across all fired rules (capped at 1.0)",
    )
    matches: list[DetectionMatch] = Field(
        default_factory=list,
        description="All rule matches sorted by descending score contribution",
    )
    rules_evaluated: int = Field(
        default=0,
        ge=0,
        description="Number of rules evaluated against the prompt",
    )
    caller_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from SentinelConfig plus per-call overrides",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional context attached at call time",
    )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @field_serializer("timestamp")
    def _serialize_timestamp(self, dt: datetime, _info: Any) -> str:
        """Serialise datetime to ISO-8601 string with UTC offset."""
        return dt.isoformat()

    @field_serializer("threat_level")
    def _serialize_threat_level(self, level: ThreatLevel, _info: Any) -> str:
        """Serialise ThreatLevel to its string value."""
        return level.value

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def matched_rule_ids(self) -> list[str]:
        """Return a list of rule IDs for all matches in this report."""
        return [m.rule_id for m in self.matches]

    @property
    def top_match(self) -> DetectionMatch | None:
        """Return the highest-scoring match, or ``None`` if no rules fired."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.score_contribution)

    def to_log_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for structured logging.

        This method serialises the report to a dict that structlog can render
        as JSON without any additional processing.
        """
        data = self.model_dump(mode="json")
        # Ensure threat_level is the string value
        data["threat_level"] = self.threat_level.value
        return data


# ---------------------------------------------------------------------------
# ThreatDetectedError – raised when a call is blocked
# ---------------------------------------------------------------------------

class ThreatDetectedError(Exception):
    """Raised by :class:`~llm_sentinel.proxy.SentinelProxy` when a prompt is
    classified at or above the configured block threshold.

    Attributes
    ----------
    report:
        The :class:`ThreatReport` that triggered the block.
    message:
        Human-readable description of why the call was blocked.
    """

    def __init__(self, report: ThreatReport, message: str | None = None) -> None:
        self.report = report
        if message is None:
            level = report.threat_level.value
            top = report.top_match
            rule_hint = f" (top rule: {top.rule_id})" if top else ""
            message = (
                f"LLM call blocked: prompt classified as {level}{rule_hint}. "
                f"Score: {report.cumulative_score:.3f}, "
                f"Matches: {len(report.matches)}"
            )
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"ThreatDetectedError(level={self.report.threat_level.value!r}, "
            f"score={self.report.cumulative_score:.3f}, "
            f"matches={len(self.report.matches)})"
        )
