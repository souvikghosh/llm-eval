"""Tests for the schema compliance metric (deterministic — no LLM calls)."""

import pytest
from llm_eval.metrics.schema_compliance import SchemaComplianceMetric


class TestSchemaCompliance:
    @pytest.mark.asyncio
    async def test_valid_json_no_schema(self):
        metric = SchemaComplianceMetric()
        result = await metric.score("input", '{"key": "value"}')
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        metric = SchemaComplianceMetric()
        result = await metric.score("input", "not json at all")
        assert result.score == 0.0
        assert result.passed is False
        assert "Invalid JSON" in result.details

    @pytest.mark.asyncio
    async def test_required_fields_present(self):
        metric = SchemaComplianceMetric(required_fields=["name", "age"])
        result = await metric.score("input", '{"name": "Alice", "age": 30}')
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_required_fields_missing(self):
        metric = SchemaComplianceMetric(required_fields=["name", "age", "email"])
        result = await metric.score("input", '{"name": "Alice"}')
        assert result.score < 1.0
        assert result.passed is False
        assert "Missing required field" in result.details

    @pytest.mark.asyncio
    async def test_type_checking(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            }
        }
        metric = SchemaComplianceMetric(schema=schema)
        result = await metric.score("input", '{"name": "Alice", "count": 5}')
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_type_mismatch(self):
        schema = {
            "properties": {
                "count": {"type": "integer"},
            }
        }
        metric = SchemaComplianceMetric(schema=schema)
        result = await metric.score("input", '{"count": "not a number"}')
        assert result.score == 0.0
        assert "expected type" in result.details

    @pytest.mark.asyncio
    async def test_partial_compliance(self):
        metric = SchemaComplianceMetric(required_fields=["a", "b", "c", "d"])
        result = await metric.score("input", '{"a": 1, "b": 2}')
        assert result.score == 0.5  # 2 out of 4 fields
