from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import fetch_stats, init_db  # noqa: E402
from app.pipeline import handle_completion, reload_classifier  # noqa: E402
from app.settings import Settings
import app.client as client_mod
import app.db as db_mod
import app.pipeline as pipeline_mod
import app.router as router_mod
import app.settings as settings_mod


async def run(n: int = 200) -> dict:
    labeled = []
    with (ROOT / "data" / "prompts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            labeled.append(json.loads(line)["prompt"])
    prompts = [labeled[i % len(labeled)] for i in range(n)]

    s = Settings(mock_llm=True, database_path=str(ROOT / "data" / "app.db"))
    settings_mod.get_settings = lambda: s  # type: ignore
    db_mod.get_settings = lambda: s  # type: ignore
    pipeline_mod.get_settings = lambda: s  # type: ignore
    client_mod.get_settings = lambda: s  # type: ignore
    router_mod.get_settings = lambda: s  # type: ignore
    reload_classifier()
    init_db()

    for prompt in prompts:
        await handle_completion(
            [{"role": "user", "content": prompt}],
            verify_async=False,
        )
    stats = fetch_stats()
    out = ROOT / "artifacts" / "load_test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": n,
        "mode": "offline_mock",
        **{k: stats[k] for k in stats if k not in ("daily", "quality_scores", "routing_distribution")},
        "routing_distribution": stats["routing_distribution"],
        "cost_reduction_pct": stats["cost_reduction_pct"],
        "cost_reduction_vs_gpt4o_pct": stats["cost_reduction_vs_gpt4o_pct"],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    asyncio.run(run(int(sys.argv[1]) if len(sys.argv) > 1 else 500))
