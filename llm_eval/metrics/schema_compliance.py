"""Schema compliance metric — validates LLM output against a JSON schema."""

import json
from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult, Severity


class SchemaComplianceMetric(BaseMetric):
    """Validates that LLM output conforms to an expected JSON schema.

    This is a deterministic metric — no LLM judge needed.
    Checks: valid JSON, required fields present, correct types.
    """

    def __init__(
        self,
        schema: dict | None = None,
        required_fields: list[str] | None = None,
        threshold: float = 1.0,
        severity: Severity = Severity.HIGH,
    ):
        super().__init__(threshold=threshold, severity=severity)
        self.schema = schema
        self.required_fields = required_fields or []

    @property
    def name(self) -> str:
        return "schema_compliance"

    @property
    def description(self) -> str:
        return "Validates LLM output against expected JSON schema and required fields"

    async def score(self, input_text: str, output_text: str, expected: str | None = None) -> EvalResult:
        errors = []

        # Try to parse as JSON
        try:
            data = json.loads(output_text)
        except json.JSONDecodeError as e:
            return EvalResult(
                case_name="",
                metric_name=self.name,
                score=0.0,
                passed=False,
                details=f"Invalid JSON: {str(e)}",
                severity=self.severity,
            )

        total_checks = 0
        passed_checks = 0

        # Check required fields
        for field in self.required_fields:
            total_checks += 1
            if field in data:
                passed_checks += 1
            else:
                errors.append(f"Missing required field: '{field}'")

        # Check schema types if provided
        if self.schema and "properties" in self.schema:
            for field_name, field_spec in self.schema["properties"].items():
                if field_name in data:
                    total_checks += 1
                    expected_type = field_spec.get("type")
                    actual_value = data[field_name]
                    if self._check_type(actual_value, expected_type):
                        passed_checks += 1
                    else:
                        errors.append(
                            f"Field '{field_name}': expected type '{expected_type}', "
                            f"got '{type(actual_value).__name__}'"
                        )

        # Calculate score
        if total_checks == 0:
            # JSON is valid but no schema checks to perform
            score = 1.0
        else:
            score = passed_checks / total_checks

        details = "; ".join(errors) if errors else "All schema checks passed"

        return EvalResult(
            case_name="",
            metric_name=self.name,
            score=round(score, 4),
            passed=score >= self.threshold,
            details=details,
            severity=self.severity,
        )

    @staticmethod
    def _check_type(value, expected_type: str | None) -> bool:
        """Check if a value matches the expected JSON schema type."""
        if expected_type is None:
            return True
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)
