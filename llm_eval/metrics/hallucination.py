"""Hallucination detection metric — identifies fabricated claims."""

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult, Severity

HALLUCINATION_PROMPT = """Analyze the following LLM output for hallucinations — claims that are fabricated, incorrect, or cannot be verified from the provided context.

Context (ground truth): {context}

LLM Output: {output}

Identify any hallucinated claims. Rate the hallucination level from 0.0 to 1.0 where:
- 0.0 = No hallucinations, all claims are verifiable from context
- 0.5 = Some minor hallucinations or unverifiable claims
- 1.0 = Major hallucinations, fabricated facts or contradictions

Respond with ONLY a JSON object: {{"score": <float>, "hallucinated_claims": ["claim1", ...], "reasoning": "<explanation>"}}"""


class HallucinationMetric(BaseMetric):
    """Detects hallucinated or fabricated claims in LLM output.

    Returns a score where LOWER is BETTER (inverse of other metrics).
    The pass threshold is inverted: passes when score < (1 - threshold).
    """

    def __init__(self, threshold: float = 0.7, severity: Severity = Severity.HIGH):
        super().__init__(threshold=threshold, severity=severity)

    @property
    def name(self) -> str:
        return "hallucination"

    @property
    def description(self) -> str:
        return "Detects fabricated or unverifiable claims in LLM output"

    async def score(self, input_text: str, output_text: str, expected: str | None = None) -> EvalResult:
        context = expected or input_text
        prompt = HALLUCINATION_PROMPT.format(context=context, output=output_text)
        result = await self._llm_judge(prompt)

        hallucination_score = float(result.get("score", 1.0))
        # Invert: faithfulness_equivalent = 1 - hallucination_score
        clean_score = 1.0 - hallucination_score
        hallucinated_claims = result.get("hallucinated_claims", [])

        details = result.get("reasoning", "")
        if hallucinated_claims:
            details += f" | Hallucinated claims: {hallucinated_claims}"

        return EvalResult(
            case_name="",
            metric_name=self.name,
            score=clean_score,
            passed=clean_score >= self.threshold,
            details=details,
            severity=self.severity,
        )
