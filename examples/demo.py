"""Demo eval suite showing how to test an LLM application.

This example demonstrates:
1. Defining eval cases with expected outputs
2. Running quality metrics (schema compliance)
3. Running red-teaming attacks (prompt injection)
4. Viewing formatted results

Usage:
    llm-eval run examples/demo.py
    # or
    python examples/demo.py
"""

import asyncio
from llm_eval.runners.suite import EvalSuite
from llm_eval.types import EvalCase
from llm_eval.metrics.schema_compliance import SchemaComplianceMetric
from llm_eval.attacks.prompt_injection import PromptInjectionAttack
from llm_eval.runners.reporter import print_results


async def target(input_text: str) -> str:
    """Simulated LLM target function for demo purposes.

    In a real scenario, this would call your LLM application's API.
    """
    if "extract" in input_text.lower():
        return '{"name": "Anthropic", "founded": 2021, "type": "AI Safety"}'
    return "I'm a helpful assistant. I cannot comply with requests to ignore my instructions."


def create_suite() -> EvalSuite:
    """Create the demo eval suite."""
    suite = EvalSuite("Demo Eval Suite")

    # Quality eval cases
    suite.add_case(EvalCase(
        name="structured_output",
        input="Extract company info: Anthropic is an AI safety company founded in 2021",
    ))

    # Schema compliance metric
    suite.add_metric(SchemaComplianceMetric(
        required_fields=["name", "founded", "type"],
        schema={
            "properties": {
                "name": {"type": "string"},
                "founded": {"type": "integer"},
                "type": {"type": "string"},
            }
        },
    ))

    # Red-teaming
    suite.add_attack(PromptInjectionAttack())

    return suite


async def main():
    suite = create_suite()
    result = await suite.run(target)
    print_results(result)


if __name__ == "__main__":
    asyncio.run(main())
