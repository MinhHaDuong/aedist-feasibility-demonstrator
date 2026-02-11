"""Query LLMs via OpenRouter and save results.

Usage:
    python src/query.py --prompt src/prompts/prompt1.txt \
                        --models src/models.yaml \
                        --output Results/1_simply_ask/
    python src/query.py --prompt src/prompts/prompt1.txt \
                        --models src/models.yaml \
                        --output Results/1_simply_ask/ \
                        --model deepseek/deepseek-r1
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

import yaml
from openai import OpenAI


def load_models(path: str) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def query_model(client: OpenAI, model_id: str, prompt: str) -> dict:
    """Send prompt to a model via OpenRouter, return response dict."""
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    return {
        "response": choice.message.content,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


def save_result(output_dir: Path, model_id: str, prompt: str, result: dict):
    """Save query result as JSON, append-only with date in path."""
    day_dir = output_dir / date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model id for filename: anthropic/claude-3.5-sonnet -> claude-3.5-sonnet
    filename = model_id.split("/")[-1] + ".json"
    filepath = day_dir / filename

    record = {
        "model": model_id,
        "date": date.today().isoformat(),
        "prompt": prompt,
        **result,
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"  Saved {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Query LLMs via OpenRouter")
    parser.add_argument("--prompt", required=True, help="Path to prompt text file")
    parser.add_argument("--models", required=True, help="Path to models.yaml")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--model", help="Query only this model (OpenRouter ID)")
    args = parser.parse_args()

    prompt = Path(args.prompt).read_text().strip()
    models = load_models(args.models)
    output_dir = Path(args.output)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY environment variable")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Filter to single model if requested
    if args.model:
        models = [m for m in models if m["id"] == args.model]
        if not models:
            raise SystemExit(f"Model {args.model} not found in {args.models}")

    for model in models:
        model_id = model["id"]
        label = model["label"]
        print(f"Querying {label} ({model_id})...")
        try:
            result = query_model(client, model_id, prompt)
            save_result(output_dir, model_id, prompt, result)
        except Exception as e:
            print(f"  Error: {e}")
            save_result(output_dir, model_id, prompt, {
                "response": None,
                "finish_reason": "error",
                "error": str(e),
                "usage": None,
            })


if __name__ == "__main__":
    main()
