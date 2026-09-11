# LLM Cost Autopilot

An intelligent routing layer in front of **free and open-source LLMs**. It scores each prompt’s complexity, sends it to the cheapest model that can handle it, checks quality in the background, and escalates when the cheap model is wrong.

**Headline result (offline mock load, 500 diverse prompts):** see `artifacts/load_test_report.json` after `python scripts/offline_savings.py 500`. Typical shadow-cost reduction vs always using Groq Llama 3.3 70B is **on the order of 80–95%**, because most traffic is simple/mid-tier. Actual cash spend on Groq/Gemini free tiers is **$0**.

This is not a wrapper around GPT-4o or Claude. Those APIs are not used. GPT-4o list prices appear on the dashboard only as a hypothetical “if you had sent everything to GPT-4o” comparison.

## Why it exists

Teams over-provision model calls. A reformatting job does not need a 70B model. Autopilot treats routing as a **cost and quality** problem: classify, route, verify, learn from failures.

## Architecture

```
Client --> FastAPI POST /v1/completions
              |-- sklearn complexity classifier (tier 1 / 2 / 3)
              |-- YAML routing map
              |-- unified send_request()
              |      +-- Groq (Llama 8B / 70B)
              |      +-- Gemini Flash / Flash-Lite (free AI Studio key)
              |      +-- Ollama (local llama3.2:3b)
              |      +-- mock (offline demo)
              |-- SQLite audit log
              +-- async verifier / auto-escalation / feedback.jsonl
                     |
                     v
              Streamlit dashboard (shadow $ and savings %)
```

| Complexity | Examples | Default route |
|---|---|---|
| Tier 1 simple | reformat, extract, short Q&A | Ollama 3B or Groq 8B |
| Tier 2 moderate | summary, classify, structured analysis | Gemini Flash-Lite |
| Tier 3 complex | multi-step reasoning, judgment | Groq 70B (judge + premium) |

Swap mappings in `config/routing.yaml` or `PUT /v1/routing-config` without redeploying code.

## Quick start (Windows)

```powershell
cd C:\Users\vaish\llm-cost-autopilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts/generate_dataset.py
python scripts/train_classifier.py
```

### Offline demo (no API keys)

Set `MOCK_LLM=1` in `.env`, then:

```powershell
uvicorn app.api:app --reload --port 8000
# another terminal
streamlit run dashboard/app.py
```

```powershell
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Reformat as CSV: red green blue\"}]}"
```

The response includes `routing.complexity_tier`, `routing.reason`, shadow `cost`, and the premium/GPT-4o baselines.

### Free live providers

1. Groq key (no credit card): https://console.groq.com  
2. Gemini key: https://aistudio.google.com  
3. Optional Ollama: install from https://ollama.com then `ollama pull llama3.2:3b`

Put keys in `.env`, keep `MOCK_LLM=0`. Missing keys are skipped; the router uses whatever is healthy.

Send the same 10 prompts to every model:

```powershell
python scripts/baseline_eval.py --limit 10
```

Load test against a running API (respect Groq free-tier ~30 RPM):

```powershell
python scripts/load_test.py --n 500 --rpm 30
```

Offline savings report (no HTTP, mock provider):

```powershell
python scripts/offline_savings.py 500
```

### Docker

```powershell
copy .env.example .env
docker compose up --build
```

API: http://localhost:8000  Dashboard: http://localhost:8501  
Ollama on the host is reached via `host.docker.internal:11434`.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Chat completion; **you do not pick the model** |
| GET | `/v1/models` | Registry + provider health |
| GET | `/v1/stats` | Cost savings summary |
| GET/PUT | `/v1/routing-config` | Live tier → model map |
| GET | `/health` | Liveness |

Example body:

```json
{
  "messages": [{"role": "user", "content": "Summarize this passage in 3 bullets..."}],
  "use_case": "summarization",
  "verify": true
}
```

## Quality loop

After the user gets a response, a worker (or FastAPI background task) sends the same prompt to the highest-tier healthy model, scores agreement (token overlap, plus LLM-as-judge for summaries). If the cheap answer is below the use-case threshold and latency budget remains, Autopilot **auto-escalates**, logs original model, new model, cost delta, and quality gap, and appends the prompt to `data/feedback.jsonl` as a harder (tier 3) example.

Retrain weekly (or whenever you want):

```powershell
python scripts/retrain.py
```

## Dashboard metrics

- **Cost reduction %** vs always-premium (Groq 70B shadow rates) — the money shot  
- Hypothetical GPT-4o list-price comparison  
- Routing mix (which models handled what share)  
- Quality histogram and escalation rate  
- Daily routed vs baseline spend  

Shadow prices live in `config/models.yaml` (USD per 1M tokens, Groq/Gemini public list rates; Ollama uses a tiny compute-equivalent rate).

## Tests

```powershell
pytest -q
```

## Project layout

- `app/client.py` — `send_request(prompt, model_config)`  
- `app/classifier.py` — Random forest on hand-labeled features  
- `data/prompts.jsonl` — 210 labeled prompts (70 per tier)  
- `config/models.yaml` / `config/routing.yaml`  
- `dashboard/app.py` — Streamlit  
- `scripts/` — dataset, train, baseline, load test, retrain  

## Case study (portfolio)

> I built a routing layer that reduced shadow LLM cost by **X%** versus always calling a 70B model, while keeping a verification loop that escalates misses. Complexity is a sklearn classifier (target ≥80% held-out accuracy). Providers are Groq, Gemini free tier, and optional Ollama — no paid OpenAI/Anthropic keys.

Fill in **X** from `artifacts/load_test_report.json` → `cost_reduction_pct` after you run the load test.
