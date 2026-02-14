"""Faithfulness metric — measures if the output is grounded in provided context."""

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult, Severity

FAITHFULNESS_PROMPT = """Evaluate how faithfully the following answer is grounded in the provided context.

Context/Source: {context}

Answer: {output}

Rate faithfulness from 0.0 to 1.0 where:
- 1.0 = Every claim is directly supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = The answer contradicts or has no basis in the context

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}"""


class FaithfulnessMetric(BaseMetric):
    """Measures whether the LLM output is grounded in the provided source context.

    Useful for RAG evaluation — checks that the model doesn't hallucinate
    beyond what the retrieved documents support.
    """

    @property
    def name(self) -> str:
        return "faithfulness"

    @property
    def description(self) -> str:
        return "Measures if the output is faithfully grounded in the provided context"

    async def score(self, input_text: str, output_text: str, expected: str | None = None) -> EvalResult:
        """Score faithfulness. Uses `expected` as the source context."""
        context = expected or input_text
        prompt = FAITHFULNESS_PROMPT.format(context=context, output=output_text)
        result = await self._llm_judge(prompt)

        score = float(result.get("score", 0.0))
        return EvalResult(
            case_name="",  # set by runner
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            details=result.get("reasoning", ""),
            severity=self.severity,
        )
