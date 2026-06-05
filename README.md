# llm-eval

**pytest for your LLM** — evaluate faithfulness, detect hallucinations, and red-team prompt injection in one Python toolkit.

Built for AI engineers who want to ship LLM features with confidence, not guesswork.

```bash
pip install llm-eval
llm-eval run examples/demo.py
```

---

## What it does

Most LLM testing is manual: paste a prompt, eyeball the output, ship it. `llm-eval` brings structured quality gates and adversarial probing to your AI pipeline — the same rigor you'd apply to any production system, built for language models.

| Capability | What it checks |
|---|---|
| **Faithfulness** | Does the output stick to the source context? |
| **Hallucination detection** | Is the model fabricating facts? |
| **Relevance scoring** | Is the response actually on-topic? |
| **Toxicity** | Does the output contain harmful content? |
| **Schema compliance** | Does structured output match the expected format? |
| **Prompt injection** | Can an attacker hijack your system prompt? |
| **Jailbreak probing** | Does the model hold its safety boundaries under adversarial input? |

---

## Quick start

```bash
pip install llm-eval
export ANTHROPIC_API_KEY=your_key_here
llm-eval run examples/demo.py
```

Or with Docker:

```bash
docker build -t llm-eval .
docker run -e ANTHROPIC_API_KEY=your_key llm-eval run examples/demo.py
```

---

## Writing an eval suite

```python
from llm_eval.runners.suite import EvalSuite
from llm_eval.types import EvalCase
from llm_eval.metrics.faithfulness import FaithfulnessMetric
from llm_eval.metrics.hallucination import HallucinationMetric
from llm_eval.attacks.prompt_injection import PromptInjectionAttack

def create_suite() -> EvalSuite:
    suite = EvalSuite("My RAG pipeline evals")

    suite.add_case(EvalCase(
        name="company_extraction",
        input="Extract company info: Anthropic is an AI safety company founded in 2021",
    ))

    suite.add_metric(FaithfulnessMetric())
    suite.add_metric(HallucinationMetric())
    suite.add_attack(PromptInjectionAttack())

    return suite

async def target(input_text: str) -> str:
    # your LLM application call goes here
    ...
```

Two functions, one file. `create_suite()` defines what to test; `target()` wraps your application.

---

## CLI reference

```bash
llm-eval run <suite_file>        # run an eval suite
llm-eval list-metrics            # show available quality metrics
llm-eval list-attacks            # show available red-team attacks
```

---

## Architecture

```
llm_eval/
├── metrics/        # quality evaluators (faithfulness, hallucination, relevance, toxicity, schema)
├── attacks/        # adversarial probes (prompt injection, jailbreak)
├── runners/        # suite orchestration and result reporting
├── types.py        # EvalCase and result types
├── config.py       # settings via env / .env file
└── cli.py          # Click-based CLI
```

LLM-as-judge: quality metrics use Claude as the evaluator via the Anthropic API. Attacks generate adversarial test cases and assert model behavior holds.

---

## Requirements

- Python 3.11+
- Anthropic API key (`ANTHROPIC_API_KEY`)

Copy `.env.example` to `.env` and add your key.

---

## Stack

Python · Anthropic Claude API · Pydantic · Click · Rich · Docker · pytest
