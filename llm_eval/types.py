"""Core types for the eval framework."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class Severity(Enum):
    """Severity level for eval failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvalCase:
    """A single evaluation test case."""
    name: str
    input: str
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result from running a single eval case."""
    case_name: str
    metric_name: str
    score: float  # 0.0 to 1.0
    passed: bool
    details: str = ""
    severity: Severity = Severity.MEDIUM
    latency_ms: float = 0.0
    token_count: int = 0


@dataclass
class EvalSuiteResult:
    """Aggregated results from an eval suite run."""
    suite_name: str
    results: list[EvalResult] = field(default_factory=list)
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    def compute_summary(self) -> None:
        """Calculate summary statistics from individual results."""
        self.total_cases = len(self.results)
        self.passed = sum(1 for r in self.results if r.passed)
        self.failed = self.total_cases - self.passed
        self.avg_score = (
            sum(r.score for r in self.results) / self.total_cases
            if self.total_cases > 0 else 0.0
        )
        self.total_latency_ms = sum(r.latency_ms for r in self.results)
        self.total_tokens = sum(r.token_count for r in self.results)


# Type alias for the function under test
TargetFunction = Callable[[str], Awaitable[str]]
