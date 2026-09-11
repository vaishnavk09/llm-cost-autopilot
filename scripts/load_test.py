from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def run(url: str, n: int, rpm: int) -> None:
    prompts = []
    with (ROOT / "data" / "prompts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    if not prompts:
        raise SystemExit("no prompts")
    delay = 60.0 / max(1, rpm)
    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            started = time.perf_counter()
            try:
                resp = await client.post(
                    f"{url}/v1/completions",
                    json={"messages": [{"role": "user", "content": prompt}], "verify": False},
                )
                latency = (time.perf_counter() - started) * 1000
                data = resp.json()
                routing = data.get("routing") or {}
                results.append(
                    {
                        "ok": resp.status_code == 200,
                        "status": resp.status_code,
                        "model": data.get("model"),
                        "cost": routing.get("cost"),
                        "baseline_cost": routing.get("baseline_cost"),
                        "gpt4o_cost": routing.get("gpt4o_cost"),
                        "tier": routing.get("complexity_tier"),
                        "latency_ms": latency,
                    }
                )
                print(f"{i+1}/{n} {resp.status_code} model={data.get('model')} tier={routing.get('complexity_tier')}")
            except Exception as exc:
                results.append({"ok": False, "error": str(exc)})
                print(f"{i+1}/{n} error {exc}")
            await asyncio.sleep(delay)

    routed = sum(r.get("cost") or 0 for r in results)
    baseline = sum(r.get("baseline_cost") or 0 for r in results)
    gpt4o = sum(r.get("gpt4o_cost") or 0 for r in results)
    ok = sum(1 for r in results if r.get("ok"))
    report = {
        "n": n,
        "ok": ok,
        "routed_cost": routed,
        "baseline_cost": baseline,
        "gpt4o_cost": gpt4o,
        "cost_reduction_pct": ((baseline - routed) / baseline * 100) if baseline else 0,
        "cost_reduction_vs_gpt4o_pct": ((gpt4o - routed) / gpt4o * 100) if gpt4o else 0,
        "results": results,
    }
    out = ROOT / "artifacts" / "load_test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "results"}, indent=2))
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--rpm", type=int, default=30, help="Stay under Groq free-tier RPM")
    args = p.parse_args()
    asyncio.run(run(args.url, args.n, args.rpm))


if __name__ == "__main__":
    main()
