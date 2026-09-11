from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.client import send_request  # noqa: E402
from app.registry import load_models  # noqa: E402


async def run(limit: int) -> None:
    prompts = []
    path = ROOT / "data" / "prompts.jsonl"
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            prompts.append(json.loads(line)["prompt"])
    models = [m for m in load_models() if m.enabled]
    rows = []
    for model in models:
        for prompt in prompts:
            try:
                resp = await send_request(prompt, model, timeout=60.0)
                rows.append(
                    {
                        "model": model.id,
                        "provider": model.provider,
                        "latency_ms": resp.latency_ms,
                        "cost": resp.cost,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "preview": resp.text[:160],
                    }
                )
                print(f"ok {model.id} cost={resp.cost:.6f} latency={resp.latency_ms:.0f}ms")
            except Exception as exc:
                rows.append({"model": model.id, "error": str(exc)})
                print(f"fail {model.id}: {exc}")
    out = ROOT / "artifacts" / "baseline_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send the same prompts to every registry model.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args.limit))


if __name__ == "__main__":
    main()
