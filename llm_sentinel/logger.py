"""Structured JSON audit logger for llm_sentinel.

This module provides :class:`SentinelLogger`, a structlog-based audit logger
that persists :class:`~llm_sentinel.models.ThreatReport` events as structured
JSON to both stdout and (optionally) a rotating log file.

Key features:

- **Structured JSON output** via structlog processors for machine-readable logs.
- **Rotating file handler** using :class:`logging.handlers.RotatingFileHandler`
  to bound disk usage with configurable max size and backup count.
- **Privacy-aware** – prompts are included or excluded from log records
  according to :attr:`~llm_sentinel.config.SentinelConfig.include_prompt_in_log`.
- **Threshold filtering** – events below
  :attr:`~llm_sentinel.config.SentinelConfig.log_threshold` are silently
  dropped without incurring I/O overhead.
- **Caller metadata** attached to every log record for correlation.

Typical usage::

    from llm_sentinel.config import SentinelConfig
    from llm_sentinel.logger import SentinelLogger
    from llm_sentinel.models import ThreatReport

    config = SentinelConfig(log_file="/var/log/sentinel/audit.json")
    logger = SentinelLogger(config)
    logger.log_report(report)  # logs if report.threat_level >= log_threshold
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from llm_sentinel.config import SentinelConfig, ThreatLevel
from llm_sentinel.models import ThreatReport

# ---------------------------------------------------------------------------
# Module-level structlog logger (for internal diagnostics only)
# ---------------------------------------------------------------------------

_internal_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Log level names used for threat levels
# ---------------------------------------------------------------------------

_THREAT_TO_LOG_LEVEL: dict[ThreatLevel, int] = {
    ThreatLevel.LOW: logging.INFO,
    ThreatLevel.MEDIUM: logging.WARNING,
    ThreatLevel.HIGH: logging.ERROR,
    ThreatLevel.CRITICAL: logging.CRITICAL,
}

# Name used for the Python stdlib logger that backs the rotating file handler.
_SENTINEL_LOGGER_NAME = "llm_sentinel.audit"


# ---------------------------------------------------------------------------
# Custom structlog processor: add threat_event marker
# ---------------------------------------------------------------------------

def _add_sentinel_marker(
    logger: WrappedLogger,
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """Structlog processor that tags every record with a sentinel marker.

    Parameters
    ----------
    logger:
        The wrapped logger instance (unused).
    method:
        The logging method name (unused).
    event_dict:
        The mutable event dictionary being processed.

    Returns
    -------
    EventDict
        The event_dict with an added ``sentinel`` key.
    """
    event_dict.setdefault("sentinel", True)
    return event_dict


# ---------------------------------------------------------------------------
# SentinelLogger
# ---------------------------------------------------------------------------


class SentinelLogger:
    """Structured JSON audit logger for llm_sentinel threat events.

    Creates a dedicated structlog/stdlib logging pipeline that writes
    JSON-formatted threat records to stdout and optionally to a rotating
    file.  The logger is configured once at construction time and is
    safe to reuse across many calls.

    Parameters
    ----------
    config:
        The :class:`~llm_sentinel.config.SentinelConfig` that controls
        thresholds, file paths, privacy flags, and rotation settings.

    Attributes
    ----------
    config:
        Reference to the configuration supplied at construction.
    """

    def __init__(self, config: SentinelConfig) -> None:
        self.config: SentinelConfig = config
        self._stdlib_logger: logging.Logger = self._build_stdlib_logger(config)
        self._logger: structlog.BoundLogger = self._build_structlog_logger(
            self._stdlib_logger
        )
        _internal_log.debug(
            "sentinel_logger_initialised",
            log_file=str(config.log_file) if config.log_file else None,
            log_threshold=config.log_threshold.value,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_report(self, report: ThreatReport) -> None:
        """Write *report* to the audit log if it meets the log threshold.

        The log level is mapped from :attr:`~llm_sentinel.models.ThreatReport.threat_level`:

        - ``LOW``      → ``INFO``
        - ``MEDIUM``   → ``WARNING``
        - ``HIGH``     → ``ERROR``
        - ``CRITICAL`` → ``CRITICAL``

        Parameters
        ----------
        report:
            The :class:`~llm_sentinel.models.ThreatReport` to log.
        """
        if not self.config.should_log(report.threat_level):
            return

        log_dict = self._build_log_dict(report)
        level = _THREAT_TO_LOG_LEVEL.get(report.threat_level, logging.INFO)

        # Route to the correct structlog method based on level
        if level >= logging.CRITICAL:
            self._logger.critical(**log_dict)
        elif level >= logging.ERROR:
            self._logger.error(**log_dict)
        elif level >= logging.WARNING:
            self._logger.warning(**log_dict)
        else:
            self._logger.info(**log_dict)

    def log_event(
        self,
        event: str,
        level: str = "info",
        **kwargs: Any,
    ) -> None:
        """Log an arbitrary sentinel event (not tied to a ThreatReport).

        Useful for logging proxy lifecycle events (startup, shutdown, errors)
        with the same structured JSON format.

        Parameters
        ----------
        event:
            Human-readable event description.
        level:
            One of ``"debug"``, ``"info"``, ``"warning"``, ``"error"``,
            ``"critical"``.
        **kwargs:
            Additional key-value pairs to include in the log record.
        """
        log_fn = getattr(self._logger, level, self._logger.info)
        log_fn(event=event, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_log_dict(self, report: ThreatReport) -> dict[str, Any]:
        """Convert a :class:`~llm_sentinel.models.ThreatReport` into a flat
        dictionary suitable for structured logging.

        Applies privacy settings (prompt inclusion, truncation) and flattens
        the match list to a compact representation.

        Parameters
        ----------
        report:
            The threat report to serialise.

        Returns
        -------
        dict[str, Any]
            A flat, JSON-serialisable dictionary.
        """
        # Resolve the prompt text for logging
        if self.config.include_prompt_in_log and report.prompt is not None:
            prompt_text: str | None = self.config.truncate_prompt(report.prompt)
        elif self.config.include_prompt_in_log and report.prompt is None:
            # Detector stored the prompt; use it directly if present
            prompt_text = None
        else:
            prompt_text = None

        matches_summary = [
            {
                "rule_id": m.rule_id,
                "description": m.description,
                "score": m.score_contribution,
                "threat_level": m.threat_level.value,
                "rule_type": m.rule_type,
                "matched_text": (
                    m.matched_text[:100] if m.matched_text else None
                ),
            }
            for m in report.matches
        ]

        return {
            "event": "threat_report",
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "threat_detected": report.threat_detected,
            "threat_level": report.threat_level.value,
            "cumulative_score": round(report.cumulative_score, 4),
            "rules_evaluated": report.rules_evaluated,
            "match_count": len(report.matches),
            "matches": matches_summary,
            "prompt_length": report.prompt_length,
            "prompt": prompt_text,
            "caller_metadata": report.caller_metadata,
            "extra": report.extra,
        }

    @staticmethod
    def _build_stdlib_logger(config: SentinelConfig) -> logging.Logger:
        """Create and configure a stdlib :class:`logging.Logger` with the
        appropriate handlers.

        A :class:`~logging.StreamHandler` writing to stdout is always added.
        When :attr:`~llm_sentinel.config.SentinelConfig.log_file` is set, a
        :class:`~logging.handlers.RotatingFileHandler` is also added.

        Parameters
        ----------
        config:
            Sentinel configuration providing file path and rotation settings.

        Returns
        -------
        logging.Logger
            Configured stdlib logger.
        """
        logger = logging.getLogger(_SENTINEL_LOGGER_NAME)
        # Avoid adding duplicate handlers if logger already configured
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Stdout handler – structlog handles formatting so just emit the message
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        # Use a minimal formatter; structlog produces the full JSON string
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stdout_handler)

        # Rotating file handler
        if config.log_file is not None:
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=config.log_max_bytes,
                backupCount=config.log_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def _build_structlog_logger(
        stdlib_logger: logging.Logger,
    ) -> structlog.BoundLogger:
        """Build a structlog bound logger that wraps *stdlib_logger*.

        The processor chain is:

        1. :func:`structlog.contextvars.merge_contextvars` – merge any
           context-local variables.
        2. :func:`structlog.stdlib.add_log_level` – add ``level`` key.
        3. :func:`structlog.stdlib.add_logger_name` – add ``logger`` key.
        4. :func:`structlog.processors.TimeStamper` – add ISO-8601 timestamp.
        5. :func:`_add_sentinel_marker` – add ``sentinel: true``.
        6. :func:`structlog.processors.StackInfoRenderer` – render stack info.
        7. :func:`structlog.processors.format_exc_info` – render exceptions.
        8. :func:`structlog.processors.UnicodeDecoder` – decode byte strings.
        9. :class:`structlog.processors.JSONRenderer` – serialize to JSON.

        Parameters
        ----------
        stdlib_logger:
            The stdlib logger to route final output through.

        Returns
        -------
        structlog.BoundLogger
            A configured bound logger.
        """
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_sentinel_marker,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]

        # Build a factory that wraps our stdlib logger
        bound_logger = structlog.wrap_logger(
            stdlib_logger,
            processors=processors,
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
        )
        return bound_logger
