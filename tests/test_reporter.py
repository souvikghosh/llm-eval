"""Tests for the results reporter."""

import json
import tempfile
from pathlib import Path

import pytest
from llm_eval.types import EvalResult, EvalSuiteResult, Severity
from llm_eval.runners.reporter import save_results


class TestSaveResults:
    def test_saves_json_file(self):
        suite_result = EvalSuiteResult(suite_name="test")
        suite_result.results = [
            EvalResult(
                case_name="case1",
                metric_name="relevance",
                score=0.85,
                passed=True,
                details="Good answer",
                severity=Severity.MEDIUM,
                latency_ms=150.5,
            ),
        ]
        suite_result.compute_summary()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_results(suite_result, tmpdir)
            assert Path(filepath).exists()

            data = json.loads(Path(filepath).read_text())
            assert data["suite_name"] == "test"
            assert data["total_cases"] == 1
            assert data["passed"] == 1
            assert len(data["results"]) == 1
            assert data["results"][0]["score"] == 0.85
