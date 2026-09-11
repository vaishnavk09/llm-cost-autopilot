from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "prompts.jsonl"

SIMPLE = [
    "Reformat this list as comma-separated values: apples bananas cherries dates",
    "Extract the email address from: contact Jane at jane.doe{i}@example.com today",
    "What is the capital of France?",
    "Lowercase this title: The Quick Brown Fox",
    "Convert 72 degrees Fahrenheit to Celsius. Only the number.",
    "Strip HTML tags from: <p>Hello <b>world</b></p>",
    "Return JSON with keys name and age from: Maya is 29 years old.",
    "Translate to Spanish: Good morning",
    "How many words are in: the cat sat on the mat",
    "Fix spelling: recieve the packge tommorow",
    "Is 17 a prime number? Answer yes or no.",
    "Format this phone number as (XXX) XXX-XXXX: 4155550198",
    "Extract dates from: Meeting on 2024-03-12 and follow-up 12 April 2024.",
    "Given the following note: 'buy milk'. Rewrite as a checklist item.",
    "What color is the sky on a clear day? One word.",
    "Copy-edit: i went to paris last summer",
    "Convert this bullet list to numbered: red, green, blue",
    "Extract the URL: see https://docs.python.org/3/ for details",
    "True or false: water boils at 100C at sea level.",
    "Trim whitespace and return:    padded text   ",
]

MODERATE = [
    "Summarize the following passage in 3 bullets. Passage: Urban bees declined 18% after drought year {i}.",
    "Classify the sentiment of this review as positive, negative, or mixed: The battery lasts but the camera is grainy.",
    "Outline the main claims in this article excerpt: Researchers argue remote work raises output but weakens mentoring.",
    "Extract structured JSON with fields company, amount, date from: Acme invoiced $12,400 on May 3.",
    "Categorize this support ticket: billing, technical, or account: I was charged twice for March.",
    "Summarize the transcript: Agent greeted customer, reset password, confirmed login.",
    "Label this news headline topic: Markets rally as inflation cools",
    "Compare these two product blurbs at a high level: A is cheaper, B has longer warranty.",
    "Produce a table of risks vs mitigations for launching a beta waitlist.",
    "Classify intent: book_flight, cancel, status — User: Can I change my Tuesday seat?",
    "Summarize this README paragraph for a non-engineer. Context: The CLI uploads CSV files to S3.",
    "Extract key-value pairs from: env=prod region=us-east-1 replicas=3",
    "Write a structured analysis of why NPS dropped from 62 to 51 in one quarter.",
    "Categorize emails as spam or ham: Congratulations you won a cruise if you click now.",
    "Summarize meeting notes into decisions, owners, and dates.",
    "Classify the document type: invoice, contract, or memo. Text: Payment due net 30 for services rendered.",
    "Outline a 5-step onboarding checklist for a new SRE.",
    "Extract entities (people, orgs, locations) from: Satya spoke in Redmond about OpenAI.",
    "Sentiment + aspect labels for: The UI is gorgeous but search is slow.",
    "Summarize this policy in plain language. Context: PTO accrues 1.5 days per month, max 20.",
]

COMPLEX = [
    "Analyze the trade-offs between eventual consistency and strong consistency for a payments ledger. Reason step by step.",
    "Compare RAG vs fine-tuning for a legal Q&A bot and justify a recommendation with constraints: must cite sources, latency under 2s.",
    "Design a multi-step incident response plan for a leaked API key, including detection, rotation, and customer comms. Be nuanced.",
    "Evaluate whether this A/B test (p=0.06, n=900) is enough to ship a checkout change. Critique the methodology.",
    "Synthesize a strategy to cut LLM spend 40% without harming support CSAT. Consider routing, caching, and human fallback.",
    "Debate the ethics of using customer tickets to train models. Present both sides then a reasoned policy.",
    "Prove informally why binary search is O(log n) and when it would be the wrong tool.",
    "Analyze this architecture: sync monolith + nightly ETL. Propose an event-driven redesign and compare failure modes.",
    "Write a nuanced judgment: should we block a model that refuses medical advice but answers related chemistry questions?",
    "Multi-step reasoning: given three vendor SLAs, pick one for a hospital paging system and justify.",
    "Critique this prompt and redesign it for fewer hallucinations. Then explain why the new version is safer.",
    "Compare CAP theorem implications for a global inventory service. Design around partition events.",
    "Evaluate two candidate embeddings for legal search; reason about recall vs cost vs license constraints.",
    "Create a creative but technically accurate explainer of KV-cache for a VP of Finance, then a deeper appendix for engineers.",
    "Analyze failure of a canary that looked green: metrics, hidden cohorts, and a decision tree to roll back.",
    "Design an eval harness for a classifier with shifting labels. Include sampling, judges, and cost controls.",
    "Reason about whether 8B vs 70B is appropriate for contract redlining. Include risk of missed clauses.",
    "Synthesize a 90-day plan to replace a vendor LLM with open-source, listing blockers and rollback criteria.",
    "Nuanced judgment call: a model slightly misquotes a statute. Escalate, rewrite, or abstain? Defend the choice.",
    "Compare streaming vs batch inference for nightly report generation; include energy, latency, and ops trade-offs.",
]


def expand(templates: list[str], n: int, tier: int) -> list[dict]:
    rows: list[dict] = []
    i = 0
    while len(rows) < n:
        base = templates[i % len(templates)]
        text = base.format(i=i) if "{i}" in base else f"{base} (variant {i})"
        if tier == 1 and i % 3 == 0:
            text = f"Reformat only. {text}"
        if tier == 2 and i % 4 == 0:
            text = f"{text}\nReturn JSON. Must include a confidence field."
        if tier == 3 and i % 2 == 0:
            text = f"{text} You must analyze, compare alternatives, and justify the recommendation."
        rows.append({"prompt": text, "tier": tier})
        i += 1
    return rows


def main() -> None:
    rows = expand(SIMPLE, 70, 1) + expand(MODERATE, 70, 2) + expand(COMPLEX, 70, 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {OUT}")


if __name__ == "__main__":
    main()
