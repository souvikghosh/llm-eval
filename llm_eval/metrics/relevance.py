"""Relevance metric — measures if the output answers the question."""

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult

RELEVANCE_PROMPT = """Evaluate how relevant and complete the following answer is to the question.

Question: {input}

Answer: {output}

Rate relevance from 0.0 to 1.0 where:
- 1.0 = Directly and completely answers the question
- 0.5 = Partially answers or includes unnecessary information
- 0.0 = Does not address the question at all

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}"""


class RelevanceMetric(BaseMetric):
    """Measures whether the output is relevant to the input question."""

    @property
    def name(self) -> str:
        return "relevance"

    @property
    def description(self) -> str:
        return "Measures if the output directly answers the input question"

    async def score(self, input_text: str, output_text: str, expected: str | None = None) -> EvalResult:
        prompt = RELEVANCE_PROMPT.format(input=input_text, output=output_text)
        result = await self._llm_judge(prompt)

        score = float(result.get("score", 0.0))
        return EvalResult(
            case_name="",
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            details=result.get("reasoning", ""),
            severity=self.severity,
        )
