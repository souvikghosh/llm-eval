"""Base metric interface."""

from abc import ABC, abstractmethod

import anthropic

from llm_eval.config import settings
from llm_eval.types import EvalResult, Severity


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.

    Each metric scores an LLM output on a specific quality dimension.
    Metrics can use LLM-as-judge or deterministic scoring.
    """

    def __init__(self, threshold: float = 0.7, severity: Severity = Severity.MEDIUM):
        self.threshold = threshold
        self.severity = severity
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @abstractmethod
    async def score(self, input_text: str, output_text: str, expected: str | None = None) -> EvalResult:
        """Score an LLM output. Returns EvalResult with score in [0, 1]."""
        ...

    async def _llm_judge(self, prompt: str) -> dict:
        """Use Claude as a judge to evaluate output quality.

        Expects the LLM to return a JSON object with 'score' and 'reasoning' keys.
        """
        import json

        response = self.client.messages.create(
            model=settings.eval_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Try to parse JSON from the response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```" in text:
                json_str = text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                return json.loads(json_str.strip())
            return {"score": 0.0, "reasoning": f"Failed to parse judge response: {text[:200]}"}
