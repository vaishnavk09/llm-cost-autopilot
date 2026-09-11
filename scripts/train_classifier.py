from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.classifier import train_from_rows  # noqa: E402
from app.settings import abs_path, get_settings  # noqa: E402


def load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    prompts, labels = [], []
    if not path.exists():
        return prompts, labels
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            labels.append(int(row["tier"]))
    return prompts, labels


def main() -> None:
    labeled = ROOT / "data" / "prompts.jsonl"
    feedback = ROOT / "data" / "feedback.jsonl"
    prompts, labels = load_jsonl(labeled)
    fp, fl = load_jsonl(feedback)
    prompts.extend(fp)
    labels.extend(fl)
    if len(prompts) < 20:
        raise SystemExit("Need labeled data. Run python scripts/generate_dataset.py first.")
    clf, metrics = train_from_rows(prompts, labels)
    dest = clf.save(abs_path(get_settings().classifier_path))
    report = ROOT / "artifacts" / "classifier_metrics.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(dest), **metrics}, indent=2))
    if metrics["accuracy"] < 0.80:
        raise SystemExit(f"accuracy {metrics['accuracy']:.3f} below 0.80 target")


if __name__ == "__main__":
    main()
