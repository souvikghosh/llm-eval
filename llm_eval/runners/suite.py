"""Eval suite — define and run evaluation suites against a target function."""

import time
from dataclasses import dataclass, field

from llm_eval.types import EvalCase, EvalResult, EvalSuiteResult, TargetFunction
from llm_eval.metrics.base import BaseMetric
from llm_eval.attacks.base import BaseAttack


class EvalSuite:
    """A collection of eval cases, metrics, and optional attacks to run against a target.

    Usage:
        suite = EvalSuite("my_suite")
        suite.add_case(EvalCase(name="basic", input="What is Python?", expected="programming language"))
        suite.add_metric(RelevanceMetric())
        suite.add_metric(FaithfulnessMetric())
        result = await suite.run(my_llm_function)
    """

    def __init__(self, name: str):
        self.name = name
        self.cases: list[EvalCase] = []
        self.metrics: list[BaseMetric] = []
        self.attacks: list[BaseAttack] = []

    def add_case(self, case: EvalCase) -> "EvalSuite":
        """Add an eval case. Returns self for chaining."""
        self.cases.append(case)
        return self

    def add_cases(self, cases: list[EvalCase]) -> "EvalSuite":
        """Add multiple eval cases."""
        self.cases.extend(cases)
        return self

    def add_metric(self, metric: BaseMetric) -> "EvalSuite":
        """Add a metric to evaluate outputs against."""
        self.metrics.append(metric)
        return self

    def add_attack(self, attack: BaseAttack) -> "EvalSuite":
        """Add a red-teaming attack to the suite."""
        self.attacks.append(attack)
        return self

    async def run(self, target: TargetFunction) -> EvalSuiteResult:
        """Run all eval cases and attacks against the target function.

        For each case:
        1. Call the target function with the input
        2. Score the output with all metrics
        3. Collect results

        For each attack:
        1. Generate adversarial cases
        2. Run them against the target
        3. Evaluate if the attack succeeded
        """
        suite_result = EvalSuiteResult(suite_name=self.name)

        # Run standard eval cases
        for case in self.cases:
            output = await self._call_target(target, case.input)
            for metric in self.metrics:
                start = time.perf_counter()
                result = await metric.score(case.input, output, expected=case.expected)
                elapsed_ms = (time.perf_counter() - start) * 1000

                result.case_name = case.name
                result.latency_ms = round(elapsed_ms, 2)
                suite_result.results.append(result)

        # Run red-teaming attacks
        for attack in self.attacks:
            attack_cases = attack.generate_cases()
            for case in attack_cases:
                output = await self._call_target(target, case.input)
                attack_result = await attack.evaluate_output(case.input, output)

                # Convert attack result to EvalResult
                eval_result = EvalResult(
                    case_name=case.name,
                    metric_name=f"attack:{attack.name}",
                    score=0.0 if attack_result.attack_succeeded else 1.0,
                    passed=not attack_result.attack_succeeded,
                    details=attack_result.details,
                )
                suite_result.results.append(eval_result)

        suite_result.compute_summary()
        return suite_result

    async def _call_target(self, target: TargetFunction, input_text: str) -> str:
        """Call the target function and return its output."""
        try:
            return await target(input_text)
        except Exception as e:
            return f"[ERROR] Target function raised: {str(e)}"
