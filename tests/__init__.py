"""Test suite for llm_sentinel.

This package contains unit and integration tests organised by module:

- test_detector.py  : Tests for the core detection engine (rules + scoring).
- test_proxy.py     : Integration tests for SentinelProxy (pass-through, blocking,
                      alert triggering).
- test_alerter.py   : Tests for Slack and generic webhook alert dispatch using
                      mocked HTTP responses via respx.
- conftest.py       : Shared pytest fixtures (sample reports, configs, mock LLM).
"""
