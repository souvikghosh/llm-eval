"""Base attack interface for red-teaming."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_eval.types import EvalCase


@dataclass
class AttackResult:
    """Result from running an attack against a target."""
    attack_name: str
    original_input: str
    adversarial_input: str
    target_output: str
    attack_succeeded: bool
    details: str = ""


class BaseAttack(ABC):
    """Abstract base class for red-teaming attacks.

    Attacks generate adversarial inputs designed to make LLMs
    produce undesirable outputs (leak system prompts, ignore
    instructions, produce harmful content, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def generate_cases(self, base_input: str | None = None) -> list[EvalCase]:
        """Generate adversarial eval cases for this attack type."""
        ...

    @abstractmethod
    async def evaluate_output(self, adversarial_input: str, output: str) -> AttackResult:
        """Check if the attack succeeded based on the LLM's output."""
        ...
