# Case study: LLM Cost Autopilot

I built a routing layer that reduced shadow LLM cost by **48.9%** versus always calling a 70B model, and by **94.6%** versus hypothetical GPT-4o list prices, while keeping a verification loop that escalates misses. Complexity is a scikit-learn random forest (**92.9%** held-out accuracy). Providers are Groq, Gemini free tier, Hugging Face open-weight models, and optional Ollama — no paid OpenAI or Anthropic keys.

## The business problem

Teams default every prompt to the strongest model. Reformatting a list and redlining a contract do not have the same unit economics. Autopilot treats routing as a cost-and-quality control loop, not a model picker UI.

## How routing works

1. Extract features from the prompt (token count, instruction verbs, constraints, whether context is attached, requested output format).
2. Classify into three complexity tiers: simple, moderate, complex.
3. Map the tier to an ordered list of models in `config/routing.yaml`. The first healthy provider wins; failures fail over down the list.
4. Return the completion with routing metadata (`complexity_tier`, model, shadow cost, baseline cost).
5. A background worker sends the same prompt to a high-tier judge, scores agreement, logs routing failures, and appends them to `data/feedback.jsonl` for weekly retraining.

Sync auto-escalation is available when the client sets `wait_for_quality: true` and the first call finishes inside the latency budget.

## Evidence (offline mock load, 500 prompts)

Source: `artifacts/load_test_report.json` (`python scripts/offline_savings.py 500`).

| Metric | Value |
|---|---|
| Requests | 500 |
| Cost reduction vs Groq Llama 3.3 70B shadow rates | **48.9%** |
| Cost reduction vs GPT-4o list prices (never called) | **94.6%** |
| Routing mix | Ollama 3B 192 · Gemini Flash-Lite 168 · Groq 70B 140 |
| Classifier held-out accuracy | 92.9% (42-example test split) |

Cash spend on Groq / Gemini / Hugging Face free tiers is **$0**. Shadow dollars exist so the dashboard can show “you saved $X” against published list rates.

## Why this is portfolio-ready

A VP of Engineering can read one number (cost reduction %), see the fallback map, and inspect the SQLite audit trail per request. The flywheel is explicit: routing failures become new labeled examples and the classifier is retrained with `python scripts/retrain.py`.
