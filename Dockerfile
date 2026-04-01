FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# Run the eval CLI by default
# Pass your eval config: docker run llm-eval llm-eval run --config evals/my_eval.yaml
ENTRYPOINT ["llm-eval"]
CMD ["--help"]
