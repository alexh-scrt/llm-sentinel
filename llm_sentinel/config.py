"""Configuration model for llm_sentinel.

Defines the SentinelConfig pydantic model that governs all behavioural aspects
of the sentinel: detection thresholds, blocking policy, alert destinations,
audit log settings, and custom rule injection.

Also exports the ThreatLevel enum which is used across the library to
categorise the severity of a detected threat.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator


class ThreatLevel(str, Enum):
    """Ordered severity levels for detected threats.

    Levels are ranked from least to most severe:
    LOW < MEDIUM < HIGH < CRITICAL.

    Each level is a string so that it serialises cleanly to JSON / logs.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    # ------------------------------------------------------------------
    # Comparison helpers so that levels can be ordered (e.g. >= HIGH).
    # ------------------------------------------------------------------

    _ORDER: dict[str, int]  # populated below after class body

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return self._rank() < other._rank()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return self._rank() <= other._rank()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return self._rank() > other._rank()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return self._rank() >= other._rank()

    def _rank(self) -> int:
        """Return numeric rank for ordering comparisons."""
        _ranks = {
            ThreatLevel.LOW: 0,
            ThreatLevel.MEDIUM: 1,
            ThreatLevel.HIGH: 2,
            ThreatLevel.CRITICAL: 3,
        }
        return _ranks[self]


class CustomRule(BaseModel):
    """A user-defined detection rule injected via SentinelConfig.

    Attributes
    ----------
    rule_id:
        Unique identifier for the rule (e.g. ``"CORP-001"``).
    description:
        Human-readable description of what the rule detects.
    pattern:
        A regular-expression string that is matched against the raw prompt text
        (case-insensitive by default unless *case_sensitive* is True).
    keywords:
        Optional list of plain-text keywords; a match on *any* keyword
        contributes to the score independently of the regex pattern.
    score:
        The numeric score (0.0 – 1.0) added to the prompt's cumulative threat
        score when this rule fires.
    threat_level:
        The minimum ThreatLevel this rule is considered to represent.  The
        final ThreatLevel of a ThreatReport is the maximum across all fired rules.
    case_sensitive:
        When True the regex pattern is compiled without re.IGNORECASE.
    enabled:
        Set to False to temporarily disable a rule without removing it.
    """

    rule_id: str = Field(..., min_length=1, description="Unique rule identifier")
    description: str = Field(default="", description="Human-readable rule description")
    pattern: str | None = Field(
        default=None,
        description="Regex pattern matched against the prompt text",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Plain-text keywords that trigger this rule",
    )
    score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Contribution to cumulative threat score (0.0 – 1.0)",
    )
    threat_level: ThreatLevel = Field(
        default=ThreatLevel.MEDIUM,
        description="Severity level represented by this rule",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the regex pattern match is case-sensitive",
    )
    enabled: bool = Field(default=True, description="Whether this rule is active")

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str | None) -> str | None:
        """Ensure the regex pattern compiles without errors."""
        if v is None:
            return v
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc
        return v

    @model_validator(mode="after")
    def _require_pattern_or_keywords(self) -> "CustomRule":
        """Ensure at least one of *pattern* or *keywords* is provided."""
        if not self.pattern and not self.keywords:
            raise ValueError(
                f"Rule '{self.rule_id}' must define at least one of 'pattern' or 'keywords'."
            )
        return self


class AlertConfig(BaseModel):
    """Configuration for a single alert destination.

    Attributes
    ----------
    url:
        The HTTP/HTTPS endpoint URL (Slack incoming webhook or arbitrary webhook).
    is_slack:
        When True the payload is formatted as a Slack ``blocks`` message.  When
        False a generic JSON body containing the serialised ThreatReport is sent.
    headers:
        Optional additional HTTP headers to include in the request (e.g. for
        bearer-token authentication on private webhooks).
    timeout:
        HTTP request timeout in seconds.  Defaults to 10 seconds.
    enabled:
        Set to False to suppress alerts to this destination without removing
        its configuration.
    """

    url: AnyHttpUrl = Field(..., description="Webhook or Slack incoming-webhook URL")
    is_slack: bool = Field(
        default=False,
        description="Format payload as a Slack blocks message",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers for authenticated webhooks",
    )
    timeout: Annotated[float, Field(gt=0.0)] = Field(
        default=10.0,
        description="Request timeout in seconds",
    )
    enabled: bool = Field(default=True, description="Whether this alert destination is active")


class SentinelConfig(BaseModel):
    """Top-level configuration model for llm_sentinel.

    Attributes
    ----------
    block_threshold:
        The minimum ThreatLevel at which the proxy *blocks* the LLM call and
        raises a ``ThreatDetectedError``.  Calls with a threat level below
        this threshold are still logged (and optionally alerted on) but are
        passed through to the underlying LLM.  Defaults to ``HIGH``.
    alert_threshold:
        The minimum ThreatLevel that triggers an alert dispatch.  Must be
        less than or equal to *block_threshold*.  Defaults to ``MEDIUM``.
    log_threshold:
        The minimum ThreatLevel that is written to the structured audit log.
        Defaults to ``LOW`` (i.e. everything is logged).
    score_threshold:
        Raw cumulative score (0.0 – 1.0) above which a prompt is considered
        at least LOW-level even if no individual rule fired at that level.
        Provides a catch-all heuristic floor.  Defaults to 0.2.
    enable_built_in_rules:
        When True (default) the 30+ built-in jailbreak / injection rules are
        loaded automatically alongside any *custom_rules*.
    custom_rules:
        List of user-defined :class:`CustomRule` objects appended to the
        active rule set.
    alert_destinations:
        List of :class:`AlertConfig` objects describing where alerts are sent.
    log_file:
        Optional path to a rotating JSON log file.  When ``None`` log events
        are emitted only to stdout.
    log_max_bytes:
        Maximum size (in bytes) of a single log file before rotation.
        Defaults to 10 MB.
    log_backup_count:
        Number of rotated log file backups to retain.  Defaults to 5.
    include_prompt_in_log:
        When True (default) the full prompt text is included in log records.
        Set to False for privacy-sensitive deployments.
    include_prompt_in_alert:
        When True (default) the full prompt text is included in alert payloads.
        Set to False to avoid transmitting user data to external webhooks.
    max_prompt_length:
        Prompts longer than this value (in characters) are truncated *for
        logging/alerting only* — the original text is still analysed in full.
        Defaults to 2000.  Set to 0 to disable truncation.
    caller_metadata:
        Arbitrary key/value pairs attached to every ThreatReport produced
        under this config (e.g. service name, environment, version).
    """

    # ------------------------------------------------------------------
    # Threshold / action policy
    # ------------------------------------------------------------------
    block_threshold: ThreatLevel = Field(
        default=ThreatLevel.HIGH,
        description="Minimum threat level that causes the proxy to block the LLM call",
    )
    alert_threshold: ThreatLevel = Field(
        default=ThreatLevel.MEDIUM,
        description="Minimum threat level that triggers an alert dispatch",
    )
    log_threshold: ThreatLevel = Field(
        default=ThreatLevel.LOW,
        description="Minimum threat level written to the audit log",
    )
    score_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.2,
        description="Cumulative score floor that qualifies a prompt as LOW-level",
    )

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------
    enable_built_in_rules: bool = Field(
        default=True,
        description="Load the built-in jailbreak / injection rule set",
    )
    custom_rules: list[CustomRule] = Field(
        default_factory=list,
        description="User-defined rules appended to the active rule set",
    )

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    alert_destinations: list[AlertConfig] = Field(
        default_factory=list,
        description="Webhook / Slack destinations for alert dispatch",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_file: Path | None = Field(
        default=None,
        description="Path to rotating JSON audit log file; None = stdout only",
    )
    log_max_bytes: Annotated[int, Field(gt=0)] = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum log file size before rotation (bytes)",
    )
    log_backup_count: Annotated[int, Field(ge=0)] = Field(
        default=5,
        description="Number of rotated backup log files to keep",
    )

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------
    include_prompt_in_log: bool = Field(
        default=True,
        description="Include full prompt text in audit log records",
    )
    include_prompt_in_alert: bool = Field(
        default=True,
        description="Include full prompt text in alert payloads",
    )
    max_prompt_length: Annotated[int, Field(ge=0)] = Field(
        default=2000,
        description="Truncate prompt in logs/alerts to this many characters (0 = no truncation)",
    )

    # ------------------------------------------------------------------
    # Caller metadata
    # ------------------------------------------------------------------
    caller_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to every ThreatReport",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_threshold_ordering(self) -> "SentinelConfig":
        """Ensure log_threshold <= alert_threshold <= block_threshold."""
        if self.log_threshold > self.alert_threshold:
            raise ValueError(
                f"log_threshold ({self.log_threshold}) must be <= "
                f"alert_threshold ({self.alert_threshold})"
            )
        if self.alert_threshold > self.block_threshold:
            raise ValueError(
                f"alert_threshold ({self.alert_threshold}) must be <= "
                f"block_threshold ({self.block_threshold})"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def should_block(self, level: ThreatLevel) -> bool:
        """Return True if *level* meets or exceeds the block threshold."""
        return level >= self.block_threshold

    def should_alert(self, level: ThreatLevel) -> bool:
        """Return True if *level* meets or exceeds the alert threshold."""
        return level >= self.alert_threshold

    def should_log(self, level: ThreatLevel) -> bool:
        """Return True if *level* meets or exceeds the log threshold."""
        return level >= self.log_threshold

    def truncate_prompt(self, prompt: str) -> str:
        """Return *prompt* truncated to *max_prompt_length* characters.

        If *max_prompt_length* is 0 the original string is returned as-is.
        Truncated strings are suffixed with ``'…'`` to signal truncation.
        """
        if self.max_prompt_length == 0 or len(prompt) <= self.max_prompt_length:
            return prompt
        return prompt[: self.max_prompt_length] + "…"

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "populate_by_name": True,
    }
