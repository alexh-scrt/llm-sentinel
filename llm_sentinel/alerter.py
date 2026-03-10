"""Async alert dispatcher for llm_sentinel.

This module provides :class:`SentinelAlerter`, which dispatches
:class:`~llm_sentinel.models.ThreatReport` payloads asynchronously to one or
more configured alert destinations using :mod:`httpx`.

Two payload formats are supported:

- **Slack Incoming Webhook** – a rich Slack ``blocks`` message is constructed
  containing the threat level, score, matched rules, and (optionally) the
  prompt text.
- **Generic HTTP Webhook** – the full serialised
  :class:`~llm_sentinel.models.ThreatReport` is POST-ed as a JSON body.

All HTTP calls are made with :class:`httpx.AsyncClient` and errors are caught
and logged without propagating exceptions to the caller, ensuring that alert
failures never disrupt LLM request processing.

Typical usage::

    import asyncio
    from llm_sentinel.config import SentinelConfig, AlertConfig
    from llm_sentinel.alerter import SentinelAlerter

    config = SentinelConfig(
        alert_destinations=[
            AlertConfig(url="https://hooks.slack.com/…", is_slack=True),
        ]
    )
    alerter = SentinelAlerter(config)
    asyncio.run(alerter.dispatch(report))
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from llm_sentinel.config import AlertConfig, SentinelConfig, ThreatLevel
from llm_sentinel.models import ThreatReport

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Emoji / colour mappings for Slack messages
# ---------------------------------------------------------------------------

_LEVEL_EMOJI: dict[ThreatLevel, str] = {
    ThreatLevel.LOW: ":large_yellow_circle:",
    ThreatLevel.MEDIUM: ":large_orange_circle:",
    ThreatLevel.HIGH: ":red_circle:",
    ThreatLevel.CRITICAL: ":skull:",
}

_LEVEL_COLOUR: dict[ThreatLevel, str] = {
    ThreatLevel.LOW: "#FFCC00",
    ThreatLevel.MEDIUM: "#FF8800",
    ThreatLevel.HIGH: "#FF2200",
    ThreatLevel.CRITICAL: "#990000",
}


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _build_slack_payload(
    report: ThreatReport,
    config: SentinelConfig,
) -> dict[str, Any]:
    """Build a Slack Incoming Webhook payload for *report*.

    The message uses Slack Block Kit with:

    - A header block with the threat level and emoji.
    - A section block with key metrics (score, matches, report ID, timestamp).
    - An optional section block with truncated prompt text.
    - A context block listing matched rule IDs.

    Parameters
    ----------
    report:
        The threat report to format.
    config:
        The sentinel configuration (used for privacy settings).

    Returns
    -------
    dict[str, Any]
        A Slack-compatible ``blocks`` message payload.
    """
    level = report.threat_level
    emoji = _LEVEL_EMOJI.get(level, ":warning:")
    colour = _LEVEL_COLOUR.get(level, "#888888")

    header_text = f"{emoji} LLM Sentinel Alert — {level.value}"

    # Metrics fields
    fields: list[dict[str, str]] = [
        {
            "type": "mrkdwn",
            "text": f"*Threat Level*\n{level.value}",
        },
        {
            "type": "mrkdwn",
            "text": f"*Cumulative Score*\n{report.cumulative_score:.3f}",
        },
        {
            "type": "mrkdwn",
            "text": f"*Matches*\n{len(report.matches)}",
        },
        {
            "type": "mrkdwn",
            "text": f"*Rules Evaluated*\n{report.rules_evaluated}",
        },
        {
            "type": "mrkdwn",
            "text": f"*Report ID*\n`{report.report_id}`",
        },
        {
            "type": "mrkdwn",
            "text": f"*Timestamp*\n{report.timestamp.isoformat()}",
        },
    ]

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": header_text,
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": fields,
        },
    ]

    # Optional prompt block
    if config.include_prompt_in_alert and report.prompt:
        truncated = config.truncate_prompt(report.prompt)
        # Slack text fields have a 3000-char limit; stay well within it
        if len(truncated) > 800:
            truncated = truncated[:800] + "…"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Prompt (truncated)*\n```{truncated}```",
                },
            }
        )

    # Matched rules context block
    if report.matches:
        top_matches = report.matches[:5]  # limit to top-5 to avoid block overflow
        rule_lines = [
            f"• `{m.rule_id}` — {m.description or 'no description'} (score: {m.score_contribution:.2f})"
            for m in top_matches
        ]
        if len(report.matches) > 5:
            rule_lines.append(f"_… and {len(report.matches) - 5} more match(es)_")
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "\n".join(rule_lines),
                    }
                ],
            }
        )

    # Caller metadata context block
    if report.caller_metadata:
        meta_parts = [
            f"`{k}`: {v}" for k, v in list(report.caller_metadata.items())[:5]
        ]
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Metadata: " + " | ".join(meta_parts),
                    }
                ],
            }
        )

    return {
        "attachments": [
            {
                "color": colour,
                "blocks": blocks,
            }
        ]
    }


def _build_generic_payload(
    report: ThreatReport,
    config: SentinelConfig,
) -> dict[str, Any]:
    """Build a generic JSON webhook payload for *report*.

    The payload is a flat dictionary derived from the report's serialised
    form, with the prompt conditionally included based on privacy settings.

    Parameters
    ----------
    report:
        The threat report to serialise.
    config:
        The sentinel configuration (used for privacy settings).

    Returns
    -------
    dict[str, Any]
        A JSON-serialisable dictionary.
    """
    data = report.model_dump(mode="json")

    # Apply privacy setting for alert payload (may differ from log setting)
    if not config.include_prompt_in_alert:
        data["prompt"] = None
    elif data.get("prompt"):
        data["prompt"] = config.truncate_prompt(data["prompt"])

    # Ensure threat_level is serialised as string value
    data["threat_level"] = report.threat_level.value

    return {
        "sentinel_alert": True,
        "report": data,
    }


# ---------------------------------------------------------------------------
# SentinelAlerter
# ---------------------------------------------------------------------------


class SentinelAlerter:
    """Async alert dispatcher for :class:`~llm_sentinel.models.ThreatReport` events.

    Sends alert payloads to all enabled
    :class:`~llm_sentinel.config.AlertConfig` destinations configured in
    :class:`~llm_sentinel.config.SentinelConfig`.  Each destination is
    contacted concurrently using :func:`asyncio.gather`.

    Errors from individual destinations are caught and logged; a failure to
    reach one destination does not prevent alerts to others.

    Parameters
    ----------
    config:
        The sentinel configuration providing alert destinations and
        privacy settings.

    Attributes
    ----------
    config:
        Reference to the configuration supplied at construction.
    """

    def __init__(self, config: SentinelConfig) -> None:
        self.config: SentinelConfig = config
        _active = [
            d for d in config.alert_destinations if d.enabled
        ]
        log.debug(
            "alerter_initialised",
            active_destinations=len(_active),
            total_destinations=len(config.alert_destinations),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(self, report: ThreatReport) -> None:
        """Dispatch *report* to all enabled alert destinations concurrently.

        Only dispatches when :attr:`~llm_sentinel.config.SentinelConfig.should_alert`
        returns ``True`` for the report's threat level.  Silently returns
        early if no destinations are configured or the threshold is not met.

        Parameters
        ----------
        report:
            The threat report to dispatch.
        """
        if not self.config.should_alert(report.threat_level):
            log.debug(
                "alert_skipped_below_threshold",
                threat_level=report.threat_level.value,
                alert_threshold=self.config.alert_threshold.value,
                report_id=report.report_id,
            )
            return

        active_destinations = [
            dest for dest in self.config.alert_destinations if dest.enabled
        ]
        if not active_destinations:
            log.debug(
                "alert_skipped_no_destinations",
                report_id=report.report_id,
            )
            return

        tasks = [
            self._send_to_destination(dest, report)
            for dest in active_destinations
        ]
        # Gather all tasks; errors are handled inside _send_to_destination
        await asyncio.gather(*tasks, return_exceptions=False)

    async def send_to_slack(
        self,
        url: str,
        report: ThreatReport,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> bool:
        """Send a Slack-formatted alert to *url* for *report*.

        Parameters
        ----------
        url:
            Slack Incoming Webhook URL.
        report:
            The threat report to format and send.
        headers:
            Optional additional HTTP headers.
        timeout:
            Request timeout in seconds.

        Returns
        -------
        bool
            ``True`` if the request succeeded (HTTP 2xx), ``False`` otherwise.
        """
        payload = _build_slack_payload(report, self.config)
        return await self._post_json(
            url=url,
            payload=payload,
            headers=headers or {},
            timeout=timeout,
            destination_type="slack",
            report_id=report.report_id,
        )

    async def send_to_webhook(
        self,
        url: str,
        report: ThreatReport,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> bool:
        """Send a generic JSON webhook alert to *url* for *report*.

        Parameters
        ----------
        url:
            Generic webhook endpoint URL.
        report:
            The threat report to serialise and send.
        headers:
            Optional additional HTTP headers.
        timeout:
            Request timeout in seconds.

        Returns
        -------
        bool
            ``True`` if the request succeeded (HTTP 2xx), ``False`` otherwise.
        """
        payload = _build_generic_payload(report, self.config)
        return await self._post_json(
            url=url,
            payload=payload,
            headers=headers or {},
            timeout=timeout,
            destination_type="webhook",
            report_id=report.report_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _send_to_destination(
        self,
        dest: AlertConfig,
        report: ThreatReport,
    ) -> None:
        """Route *report* to the correct sender based on *dest* configuration.

        Parameters
        ----------
        dest:
            The alert destination configuration.
        report:
            The threat report to send.
        """
        url = str(dest.url)
        if dest.is_slack:
            await self.send_to_slack(
                url=url,
                report=report,
                headers=dest.headers,
                timeout=dest.timeout,
            )
        else:
            await self.send_to_webhook(
                url=url,
                report=report,
                headers=dest.headers,
                timeout=dest.timeout,
            )

    async def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
        destination_type: str,
        report_id: str,
    ) -> bool:
        """POST *payload* as JSON to *url* using an ephemeral httpx client.

        Catches all httpx and network-level exceptions and logs them without
        re-raising, so alert failures are non-fatal.

        Parameters
        ----------
        url:
            Target URL.
        payload:
            JSON-serialisable dictionary to POST.
        headers:
            Additional HTTP headers (merged with default content-type).
        timeout:
            Request timeout in seconds.
        destination_type:
            String label for logging (``"slack"`` or ``"webhook"``)
        report_id:
            The associated report ID for log correlation.

        Returns
        -------
        bool
            ``True`` on HTTP 2xx, ``False`` on error or non-2xx response.
        """
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
            ) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=request_headers,
                )

            if response.is_success:
                log.info(
                    "alert_sent",
                    destination_type=destination_type,
                    url=url,
                    status_code=response.status_code,
                    report_id=report_id,
                )
                return True
            else:
                log.warning(
                    "alert_non_2xx_response",
                    destination_type=destination_type,
                    url=url,
                    status_code=response.status_code,
                    response_body=response.text[:500],
                    report_id=report_id,
                )
                return False

        except httpx.TimeoutException as exc:
            log.error(
                "alert_timeout",
                destination_type=destination_type,
                url=url,
                error=str(exc),
                report_id=report_id,
            )
            return False

        except httpx.ConnectError as exc:
            log.error(
                "alert_connection_error",
                destination_type=destination_type,
                url=url,
                error=str(exc),
                report_id=report_id,
            )
            return False

        except httpx.HTTPError as exc:
            log.error(
                "alert_http_error",
                destination_type=destination_type,
                url=url,
                error=str(exc),
                report_id=report_id,
            )
            return False

        except Exception as exc:  # noqa: BLE001
            log.exception(
                "alert_unexpected_error",
                destination_type=destination_type,
                url=url,
                error=str(exc),
                report_id=report_id,
            )
            return False
