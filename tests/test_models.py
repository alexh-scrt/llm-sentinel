"""Tests for the core data models and configuration defined in
``llm_sentinel.config`` and ``llm_sentinel.models``.

Covers ThreatLevel ordering, SentinelConfig validation, CustomRule validation,
DetectionResult factories, ThreatReport helpers, and ThreatDetectedError.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llm_sentinel.config import AlertConfig, CustomRule, SentinelConfig, ThreatLevel
from llm_sentinel.models import (
    DetectionMatch,
    DetectionResult,
    ThreatDetectedError,
    ThreatReport,
)


# ===========================================================================
# ThreatLevel
# ===========================================================================


class TestThreatLevel:
    """Test ordering and enum semantics for ThreatLevel."""

    def test_ordering_low_lt_medium(self) -> None:
        assert ThreatLevel.LOW < ThreatLevel.MEDIUM

    def test_ordering_medium_lt_high(self) -> None:
        assert ThreatLevel.MEDIUM < ThreatLevel.HIGH

    def test_ordering_high_lt_critical(self) -> None:
        assert ThreatLevel.HIGH < ThreatLevel.CRITICAL

    def test_ordering_ge_self(self) -> None:
        assert ThreatLevel.HIGH >= ThreatLevel.HIGH

    def test_ordering_le_self(self) -> None:
        assert ThreatLevel.MEDIUM <= ThreatLevel.MEDIUM

    def test_ordering_gt(self) -> None:
        assert ThreatLevel.CRITICAL > ThreatLevel.LOW

    def test_ordering_not_lt_self(self) -> None:
        assert not (ThreatLevel.HIGH < ThreatLevel.HIGH)

    def test_string_values(self) -> None:
        assert ThreatLevel.LOW.value == "LOW"
        assert ThreatLevel.MEDIUM.value == "MEDIUM"
        assert ThreatLevel.HIGH.value == "HIGH"
        assert ThreatLevel.CRITICAL.value == "CRITICAL"

    def test_is_str_subclass(self) -> None:
        assert isinstance(ThreatLevel.LOW, str)

    def test_comparison_with_non_threat_level_returns_not_implemented(self) -> None:
        result = ThreatLevel.LOW.__lt__("something")
        assert result is NotImplemented


# ===========================================================================
# CustomRule
# ===========================================================================


class TestCustomRule:
    """Validation tests for CustomRule."""

    def test_valid_rule_with_pattern(self) -> None:
        rule = CustomRule(
            rule_id="TEST-001",
            description="Test rule",
            pattern=r"ignore\s+previous",
            score=0.8,
            threat_level=ThreatLevel.HIGH,
        )
        assert rule.rule_id == "TEST-001"
        assert rule.score == 0.8
        assert rule.enabled is True

    def test_valid_rule_with_keywords_only(self) -> None:
        rule = CustomRule(
            rule_id="TEST-002",
            keywords=["jailbreak", "bypass"],
            score=0.5,
        )
        assert rule.keywords == ["jailbreak", "bypass"]

    def test_valid_rule_with_both_pattern_and_keywords(self) -> None:
        rule = CustomRule(
            rule_id="TEST-003",
            pattern=r"dan mode",
            keywords=["DAN"],
            score=0.9,
            threat_level=ThreatLevel.CRITICAL,
        )
        assert rule.pattern == r"dan mode"
        assert "DAN" in rule.keywords

    def test_missing_pattern_and_keywords_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one of"):
            CustomRule(rule_id="BAD-001", score=0.5)

    def test_invalid_regex_pattern_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid regex"):
            CustomRule(rule_id="BAD-002", pattern="[unclosed", score=0.5)

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            CustomRule(rule_id="BAD-003", pattern=r"test", score=1.5)

    def test_score_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            CustomRule(rule_id="BAD-004", pattern=r"test", score=-0.1)

    def test_disabled_rule(self) -> None:
        rule = CustomRule(
            rule_id="TEST-DIS",
            pattern=r"test",
            enabled=False,
        )
        assert rule.enabled is False

    def test_case_sensitive_flag(self) -> None:
        rule = CustomRule(
            rule_id="TEST-CASE",
            pattern=r"Secret",
            case_sensitive=True,
        )
        assert rule.case_sensitive is True


# ===========================================================================
# AlertConfig
# ===========================================================================


class TestAlertConfig:
    """Tests for AlertConfig validation."""

    def test_valid_slack_config(self) -> None:
        cfg = AlertConfig(
            url="https://hooks.slack.com/services/T000/B000/xxxx",
            is_slack=True,
        )
        assert cfg.is_slack is True
        assert cfg.timeout == 10.0

    def test_valid_generic_webhook(self) -> None:
        cfg = AlertConfig(
            url="https://example.com/webhook",
            headers={"Authorization": "Bearer token123"},
        )
        assert cfg.headers["Authorization"] == "Bearer token123"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            AlertConfig(url="not-a-url")

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            AlertConfig(url="https://example.com/hook", timeout=0.0)


# ===========================================================================
# SentinelConfig
# ===========================================================================


class TestSentinelConfig:
    """Tests for SentinelConfig validation and helper methods."""

    def test_default_config(self) -> None:
        cfg = SentinelConfig()
        assert cfg.block_threshold == ThreatLevel.HIGH
        assert cfg.alert_threshold == ThreatLevel.MEDIUM
        assert cfg.log_threshold == ThreatLevel.LOW
        assert cfg.score_threshold == 0.2
        assert cfg.enable_built_in_rules is True
        assert cfg.custom_rules == []
        assert cfg.alert_destinations == []
        assert cfg.log_file is None
        assert cfg.include_prompt_in_log is True
        assert cfg.include_prompt_in_alert is True
        assert cfg.max_prompt_length == 2000

    def test_custom_thresholds(self) -> None:
        cfg = SentinelConfig(
            block_threshold=ThreatLevel.CRITICAL,
            alert_threshold=ThreatLevel.HIGH,
            log_threshold=ThreatLevel.MEDIUM,
        )
        assert cfg.block_threshold == ThreatLevel.CRITICAL

    def test_alert_threshold_gt_block_threshold_raises(self) -> None:
        with pytest.raises(ValidationError, match="alert_threshold"):
            SentinelConfig(
                block_threshold=ThreatLevel.MEDIUM,
                alert_threshold=ThreatLevel.HIGH,
            )

    def test_log_threshold_gt_alert_threshold_raises(self) -> None:
        with pytest.raises(ValidationError, match="log_threshold"):
            SentinelConfig(
                log_threshold=ThreatLevel.HIGH,
                alert_threshold=ThreatLevel.MEDIUM,
                block_threshold=ThreatLevel.CRITICAL,
            )

    def test_should_block_at_threshold(self) -> None:
        cfg = SentinelConfig(block_threshold=ThreatLevel.HIGH)
        assert cfg.should_block(ThreatLevel.HIGH) is True
        assert cfg.should_block(ThreatLevel.CRITICAL) is True
        assert cfg.should_block(ThreatLevel.MEDIUM) is False
        assert cfg.should_block(ThreatLevel.LOW) is False

    def test_should_alert_at_threshold(self) -> None:
        cfg = SentinelConfig(alert_threshold=ThreatLevel.MEDIUM)
        assert cfg.should_alert(ThreatLevel.MEDIUM) is True
        assert cfg.should_alert(ThreatLevel.HIGH) is True
        assert cfg.should_alert(ThreatLevel.LOW) is False

    def test_should_log_at_threshold(self) -> None:
        cfg = SentinelConfig(log_threshold=ThreatLevel.LOW)
        assert cfg.should_log(ThreatLevel.LOW) is True
        assert cfg.should_log(ThreatLevel.CRITICAL) is True

    def test_truncate_prompt_short(self) -> None:
        cfg = SentinelConfig(max_prompt_length=100)
        short = "Hello world"
        assert cfg.truncate_prompt(short) == short

    def test_truncate_prompt_long(self) -> None:
        cfg = SentinelConfig(max_prompt_length=10)
        long_prompt = "A" * 20
        result = cfg.truncate_prompt(long_prompt)
        assert result == "A" * 10 + "…"
        assert len(result) == 11  # 10 chars + ellipsis

    def test_truncate_prompt_zero_no_truncation(self) -> None:
        cfg = SentinelConfig(max_prompt_length=0)
        long_prompt = "B" * 5000
        assert cfg.truncate_prompt(long_prompt) == long_prompt

    def test_caller_metadata(self) -> None:
        cfg = SentinelConfig(caller_metadata={"service": "test-svc", "env": "prod"})
        assert cfg.caller_metadata["service"] == "test-svc"

    def test_with_custom_rules(self) -> None:
        rule = CustomRule(rule_id="USR-001", pattern=r"override", score=0.7)
        cfg = SentinelConfig(custom_rules=[rule])
        assert len(cfg.custom_rules) == 1
        assert cfg.custom_rules[0].rule_id == "USR-001"

    def test_with_alert_destination(self) -> None:
        dest = AlertConfig(url="https://hooks.slack.com/x", is_slack=True)
        cfg = SentinelConfig(alert_destinations=[dest])
        assert len(cfg.alert_destinations) == 1


# ===========================================================================
# DetectionMatch
# ===========================================================================


class TestDetectionMatch:
    """Tests for DetectionMatch model."""

    def test_basic_creation(self) -> None:
        match = DetectionMatch(
            rule_id="BUILTIN-001",
            description="Test",
            matched_text="ignore previous",
            score_contribution=0.8,
            threat_level=ThreatLevel.HIGH,
            rule_type="regex",
        )
        assert match.rule_id == "BUILTIN-001"
        assert match.matched_text == "ignore previous"
        assert match.score_contribution == 0.8

    def test_no_matched_text(self) -> None:
        match = DetectionMatch(
            rule_id="HEURISTIC-001",
            score_contribution=0.3,
            threat_level=ThreatLevel.LOW,
            rule_type="heuristic",
        )
        assert match.matched_text is None


# ===========================================================================
# DetectionResult
# ===========================================================================


class TestDetectionResult:
    """Tests for DetectionResult factory methods."""

    def test_miss_factory(self) -> None:
        result = DetectionResult.miss()
        assert result.fired is False
        assert result.match is None

    def test_hit_factory(self) -> None:
        result = DetectionResult.hit(
            rule_id="TEST-HIT",
            description="A test hit",
            matched_text="jailbreak",
            score_contribution=0.9,
            threat_level=ThreatLevel.CRITICAL,
            rule_type="keyword",
        )
        assert result.fired is True
        assert result.match is not None
        assert result.match.rule_id == "TEST-HIT"
        assert result.match.matched_text == "jailbreak"
        assert result.match.threat_level == ThreatLevel.CRITICAL
        assert result.match.rule_type == "keyword"

    def test_hit_without_matched_text(self) -> None:
        result = DetectionResult.hit(
            rule_id="HEURISTIC",
            description="Length heuristic",
            matched_text=None,
            score_contribution=0.2,
            threat_level=ThreatLevel.LOW,
            rule_type="heuristic",
        )
        assert result.fired is True
        assert result.match is not None
        assert result.match.matched_text is None


# ===========================================================================
# ThreatReport
# ===========================================================================


class TestThreatReport:
    """Tests for ThreatReport model."""

    def test_default_report(self) -> None:
        report = ThreatReport()
        assert report.threat_detected is False
        assert report.threat_level == ThreatLevel.LOW
        assert report.cumulative_score == 0.0
        assert report.matches == []
        assert isinstance(report.report_id, str)
        assert len(report.report_id) == 36  # UUID4 length
        assert isinstance(report.timestamp, datetime)
        assert report.timestamp.tzinfo is not None

    def test_unique_report_ids(self) -> None:
        r1 = ThreatReport()
        r2 = ThreatReport()
        assert r1.report_id != r2.report_id

    def test_matched_rule_ids_empty(self) -> None:
        report = ThreatReport()
        assert report.matched_rule_ids == []

    def test_matched_rule_ids_with_matches(self) -> None:
        match1 = DetectionMatch(
            rule_id="RULE-A", score_contribution=0.5, threat_level=ThreatLevel.HIGH
        )
        match2 = DetectionMatch(
            rule_id="RULE-B", score_contribution=0.3, threat_level=ThreatLevel.MEDIUM
        )
        report = ThreatReport(matches=[match1, match2])
        assert "RULE-A" in report.matched_rule_ids
        assert "RULE-B" in report.matched_rule_ids

    def test_top_match_none_when_no_matches(self) -> None:
        report = ThreatReport()
        assert report.top_match is None

    def test_top_match_returns_highest_score(self) -> None:
        match_low = DetectionMatch(
            rule_id="LOW", score_contribution=0.2, threat_level=ThreatLevel.LOW
        )
        match_high = DetectionMatch(
            rule_id="HIGH", score_contribution=0.9, threat_level=ThreatLevel.HIGH
        )
        report = ThreatReport(matches=[match_low, match_high])
        assert report.top_match is not None
        assert report.top_match.rule_id == "HIGH"

    def test_to_log_dict_has_expected_keys(self) -> None:
        report = ThreatReport(
            prompt="test prompt",
            prompt_length=11,
            threat_detected=True,
            threat_level=ThreatLevel.HIGH,
            cumulative_score=0.8,
        )
        log_dict = report.to_log_dict()
        assert "report_id" in log_dict
        assert "timestamp" in log_dict
        assert "threat_level" in log_dict
        assert "cumulative_score" in log_dict
        assert log_dict["threat_level"] == "HIGH"
        assert isinstance(log_dict["timestamp"], str)  # serialised to ISO string

    def test_serialisation_round_trip(self) -> None:
        original = ThreatReport(
            prompt="hello",
            prompt_length=5,
            threat_detected=True,
            threat_level=ThreatLevel.MEDIUM,
            cumulative_score=0.5,
            rules_evaluated=10,
        )
        data = original.model_dump(mode="json")
        restored = ThreatReport.model_validate(data)
        assert restored.report_id == original.report_id
        assert restored.threat_level == original.threat_level
        assert restored.cumulative_score == original.cumulative_score

    def test_extra_and_caller_metadata(self) -> None:
        report = ThreatReport(
            caller_metadata={"service": "api"},
            extra={"request_id": "abc123"},
        )
        assert report.caller_metadata["service"] == "api"
        assert report.extra["request_id"] == "abc123"


# ===========================================================================
# ThreatDetectedError
# ===========================================================================


class TestThreatDetectedError:
    """Tests for ThreatDetectedError."""

    def _make_report(self, level: ThreatLevel, score: float = 0.9) -> ThreatReport:
        match = DetectionMatch(
            rule_id="RULE-X",
            description="Test rule",
            matched_text="jailbreak",
            score_contribution=score,
            threat_level=level,
        )
        return ThreatReport(
            threat_detected=True,
            threat_level=level,
            cumulative_score=score,
            matches=[match],
        )

    def test_default_message_contains_level(self) -> None:
        report = self._make_report(ThreatLevel.HIGH)
        error = ThreatDetectedError(report)
        assert "HIGH" in str(error)

    def test_default_message_contains_rule_id(self) -> None:
        report = self._make_report(ThreatLevel.CRITICAL)
        error = ThreatDetectedError(report)
        assert "RULE-X" in str(error)

    def test_custom_message(self) -> None:
        report = self._make_report(ThreatLevel.HIGH)
        error = ThreatDetectedError(report, message="Custom block message")
        assert str(error) == "Custom block message"

    def test_report_attribute(self) -> None:
        report = self._make_report(ThreatLevel.HIGH)
        error = ThreatDetectedError(report)
        assert error.report is report

    def test_repr(self) -> None:
        report = self._make_report(ThreatLevel.HIGH, score=0.85)
        error = ThreatDetectedError(report)
        r = repr(error)
        assert "ThreatDetectedError" in r
        assert "HIGH" in r

    def test_is_exception(self) -> None:
        report = self._make_report(ThreatLevel.MEDIUM)
        error = ThreatDetectedError(report)
        assert isinstance(error, Exception)

    def test_raise_and_catch(self) -> None:
        report = self._make_report(ThreatLevel.CRITICAL)
        with pytest.raises(ThreatDetectedError) as exc_info:
            raise ThreatDetectedError(report)
        assert exc_info.value.report.threat_level == ThreatLevel.CRITICAL
