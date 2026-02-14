"""CLI entry point for running eval suites."""

import asyncio

import click
from rich.console import Console

from llm_eval import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """llm-eval — Evaluation and red-teaming toolkit for LLM applications."""
    pass


@cli.command()
@click.argument("suite_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="eval_results", help="Output directory for results")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def run(suite_file: str, output: str, verbose: bool):
    """Run an eval suite from a Python file.

    The file should define a function `create_suite()` that returns
    an EvalSuite and a function `target(input: str) -> str` to test.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_module", suite_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "create_suite") or not hasattr(module, "target"):
        console.print("[red]Error:[/red] Suite file must define create_suite() and target() functions")
        raise SystemExit(1)

    suite = module.create_suite()
    target_fn = module.target

    console.print(f"Running eval suite: [bold]{suite.name}[/bold]")
    console.print(f"Cases: {len(suite.cases)} | Metrics: {len(suite.metrics)} | Attacks: {len(suite.attacks)}")

    result = asyncio.run(suite.run(target_fn))

    from llm_eval.runners.reporter import print_results, save_results
    print_results(result)

    filepath = save_results(result, output)
    console.print(f"\nResults saved to: {filepath}")


@cli.command()
def list_metrics():
    """List all available evaluation metrics."""
    from llm_eval.metrics import (
        FaithfulnessMetric, RelevanceMetric, HallucinationMetric,
        ToxicityMetric, SchemaComplianceMetric,
    )
    metrics = [
        FaithfulnessMetric(), RelevanceMetric(), HallucinationMetric(),
        ToxicityMetric(), SchemaComplianceMetric(),
    ]
    console.print("\n[bold]Available Metrics:[/bold]\n")
    for m in metrics:
        console.print(f"  [cyan]{m.name}[/cyan] — {m.description}")
    console.print()


@cli.command()
def list_attacks():
    """List all available red-teaming attacks."""
    from llm_eval.attacks import PromptInjectionAttack, JailbreakAttack
    attacks = [PromptInjectionAttack(), JailbreakAttack()]
    console.print("\n[bold]Available Attacks:[/bold]\n")
    for a in attacks:
        cases = a.generate_cases()
        console.print(f"  [cyan]{a.name}[/cyan] — {a.description} ({len(cases)} test cases)")
    console.print()


if __name__ == "__main__":
    cli()
