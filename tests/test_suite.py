"""Tests for the eval suite runner."""

import pytest
from llm_eval.runners.suite import EvalSuite
from llm_eval.types import EvalCase
from llm_eval.metrics.schema_compliance import SchemaComplianceMetric
from llm_eval.attacks.prompt_injection import PromptInjectionAttack


async def mock_target(input_text: str) -> str:
    """A mock LLM target that returns predictable outputs."""
    if "json" in input_text.lower():
        return '{"name": "test", "value": 42}'
    if "ignore" in input_text.lower():
        return "I cannot comply with that request."
    return "This is a helpful response about the topic."


class TestEvalSuite:
    @pytest.mark.asyncio
    async def test_run_with_metric(self):
        suite = EvalSuite("test_suite")
        suite.add_case(EvalCase(
            name="json_output",
            input="Return json with name and value",
        ))
        suite.add_metric(SchemaComplianceMetric(required_fields=["name", "value"]))

        result = await suite.run(mock_target)
        assert result.suite_name == "test_suite"
        assert result.total_cases == 1
        assert result.passed == 1

    @pytest.mark.asyncio
    async def test_run_with_attacks(self):
        suite = EvalSuite("security_suite")
        suite.add_attack(PromptInjectionAttack())

        result = await suite.run(mock_target)
        assert result.total_cases > 0
        # Our mock target resists injection, so all attacks should fail
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_summary_computation(self):
        suite = EvalSuite("summary_test")
        suite.add_case(EvalCase(name="case1", input="Return json"))
        suite.add_case(EvalCase(name="case2", input="Another json request"))
        suite.add_metric(SchemaComplianceMetric())

        result = await suite.run(mock_target)
        assert result.total_cases == 2
        assert result.avg_score > 0

    @pytest.mark.asyncio
    async def test_chaining(self):
        suite = (
            EvalSuite("chain_test")
            .add_case(EvalCase(name="c1", input="json test"))
            .add_metric(SchemaComplianceMetric())
        )
        assert len(suite.cases) == 1
        assert len(suite.metrics) == 1

    @pytest.mark.asyncio
    async def test_target_error_handling(self):
        async def failing_target(input_text: str) -> str:
            raise RuntimeError("Target crashed")

        suite = EvalSuite("error_test")
        suite.add_case(EvalCase(name="crash", input="anything"))
        suite.add_metric(SchemaComplianceMetric())

        result = await suite.run(failing_target)
        assert result.total_cases == 1
        # Should handle error gracefully, not crash
        assert result.failed == 1
