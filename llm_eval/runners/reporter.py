"""Rich console reporter for eval results."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from llm_eval.types import EvalSuiteResult, EvalResult, Severity

console = Console()

SEVERITY_COLORS = {
    Severity.LOW: "dim",
    Severity.MEDIUM: "yellow",
    Severity.HIGH: "red",
    Severity.CRITICAL: "bold red",
}


def print_results(suite_result: EvalSuiteResult) -> None:
    """Print eval results as a formatted table to the console."""
    # Header
    status = "[green]PASSED[/green]" if suite_result.failed == 0 else "[red]FAILED[/red]"
    console.print(Panel(
        f"[bold]{suite_result.suite_name}[/bold]  {status}\n"
        f"Cases: {suite_result.total_cases}  |  "
        f"Passed: [green]{suite_result.passed}[/green]  |  "
        f"Failed: [red]{suite_result.failed}[/red]  |  "
        f"Avg Score: {suite_result.avg_score:.2f}",
        title="Eval Results",
    ))

    # Results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Metric", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Pass", justify="center")
    table.add_column("Details", max_width=60)

    for result in suite_result.results:
        score_color = "green" if result.score >= 0.7 else ("yellow" if result.score >= 0.4 else "red")
        pass_icon = "[green]OK[/green]" if result.passed else "[red]FAIL[/red]"

        table.add_row(
            result.case_name,
            result.metric_name,
            f"[{score_color}]{result.score:.2f}[/{score_color}]",
            pass_icon,
            result.details[:80] if result.details else "",
        )

    console.print(table)


def save_results(suite_result: EvalSuiteResult, output_path: str = "eval_results") -> str:
    """Save eval results as JSON to a file."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    filename = f"{suite_result.suite_name.replace(' ', '_')}_results.json"
    filepath = path / filename

    data = {
        "suite_name": suite_result.suite_name,
        "total_cases": suite_result.total_cases,
        "passed": suite_result.passed,
        "failed": suite_result.failed,
        "avg_score": round(suite_result.avg_score, 4),
        "total_latency_ms": round(suite_result.total_latency_ms, 2),
        "total_tokens": suite_result.total_tokens,
        "results": [
            {
                "case_name": r.case_name,
                "metric_name": r.metric_name,
                "score": round(r.score, 4),
                "passed": r.passed,
                "details": r.details,
                "severity": r.severity.value,
                "latency_ms": round(r.latency_ms, 2),
            }
            for r in suite_result.results
        ],
    }

    filepath.write_text(json.dumps(data, indent=2))
    return str(filepath)
