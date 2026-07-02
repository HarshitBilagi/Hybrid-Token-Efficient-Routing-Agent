# Track 1 — Hybrid Token-Efficient Routing Agent

An autonomous routing layer that decides at inference time whether to send a query to a lightweight local model (Intel AI Boost NPU) or a powerful remote LLM (Groq/Fireworks AI), optimizing for token usage while maintaining accuracy constraints.

Built as a portfolio project inspired by the AMD Developer Hackathon ACT II problem statement.

---

## What This Is

Most LLM applications blindly send every query to a powerful (expensive, slow) remote model. This system routes queries intelligently:

- **Simple factual queries** → local NPU inference (Phi-3.5-mini INT4, free, ~9s)
- **Code generation, complex reasoning, current-events queries** → remote LLM (Llama-3.3-70B via Groq, fast, token cost)

The routing decision is made by a classifier that runs before inference. Two classifiers are implemented and benchmarked against each other: a rule-based heuristic and an LLM-judged self-assessor.

---

## Hardware

- **Machine:** Asus ROG Zephyrus G16
- **CPU:** Intel Core Ultra 7 155H
- **GPU:** NVIDIA RTX 4060
- **NPU:** Intel AI Boost (Intel AI Boost) — primary local inference device
- **RAM:** 16GB DDR5
- **OS:** Windows 11

The NPU is the core differentiator. Local inference runs on the Intel AI Boost via OpenVINO GenAI, not CPU or GPU.

---

## Architecture

```
Incoming query
      │
      ▼
Query Classifier  ──────────────────────────────────────────
(runs on NPU)     rule-based (instant) or LLM-judged (~10s)
      │
      ├── confidence ≥ threshold, short output, no code ──► Local Inference (NPU)
      │                                                      Phi-3.5-mini INT4
      │                                                      OpenVINO GenAI
      │
      └── low confidence / code / current-knowledge ──────► Remote Inference
                                                             Groq (Llama-3.3-70B)
                                                             swap-ready for Fireworks AI
      │
      ▼
Response envelope
{ response, route_taken, classifier_used, tokens, latency_ms,
  classifier_signals, fallback_triggered, fallback_reason }
```

### Routing Policy

The routing policy is **output-length-aware**, not just difficulty-aware:

- Route local when: predicted output ≤ 80 tokens AND no code signals AND no current-knowledge signals AND classifier confidence ≥ 0.6
- Route remote otherwise

This is closer to what production routing systems (Martian, Unify) actually do than naive easy/hard classification.

### Runtime Fallback

If local inference fails at runtime (NPU driver error, OOM, etc.), the router automatically retries on remote rather than erroring to the caller. The `fallback_triggered` and `fallback_reason` fields in the response envelope surface this for debugging and eval logging.

---

## Project Structure

```
routing-agent/
│
├── models/                          # downloaded model weights (gitignored)
│   └── phi35-mini-int4-gq-ov/       # OpenVINO INT4 group-wise quantized
│
├── src/
│   ├── classifier/
│   │   ├── rule_based.py            # heuristic classifier (instant)
│   │   └── llm_judged.py            # LLM self-assessment classifier
│   ├── local_inference/
│   │   └── phi_pipeline.py          # OpenVINO NPU wrapper (model-agnostic)
│   ├── remote_inference/
│   │   └── remote_client.py         # provider-agnostic remote client
│   ├── router/
│   │   └── router.py                # routing orchestration layer
│   └── api/
│       └── main.py                  # FastAPI server
│
├── eval/
│   ├── benchmark.py                 # benchmark runner
│   ├── scorer.py                    # aggregation and tradeoff analysis
│   └── datasets/
│       └── benchmark_v1.json        # 9-query hand-curated evaluation set
│
├── tests/
│   ├── test_npu.py                  # NPU baseline latency measurement
│   ├── test_remote.py               # remote client smoke test
│   ├── test_rule_classifier.py      # rule-based classifier unit test
│   ├── test_llm_classifier.py       # LLM-judged classifier unit test
│   └── test_router.py               # full router integration test
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Stack

| Component | Technology |
|---|---|
| Local inference | OpenVINO GenAI (`openvino-genai`) |
| Local model | Phi-3.5-mini-instruct INT4 group-wise (`OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov`) |
| NPU runtime | Intel AI Boost via Level Zero |
| Remote inference | OpenAI-compatible SDK pointed at Groq / Fireworks AI |
| Remote model | `llama-3.3-70b-versatile` (Groq) |
| API server | FastAPI + Uvicorn |
| Evaluation | Custom benchmark runner + manual scoring |
| Language | Python 3.11 |

---

## Setup

### Prerequisites

- Python 3.10 or 3.11 (3.12 has known OpenVINO NPU dispatch issues)
- Intel NPU driver ≥ 31.0.101.5082 (standalone from intel.com/arc-drivers)
- Groq API key (free at console.groq.com) or Fireworks AI API key

### Installation

```bash
git clone https://github.com/yourhandle/routing-agent
cd routing-agent
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Download the model

```bash
hf download OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov --local-dir ./models/phi35-mini-int4-gq-ov
```

~2.1GB. Requires `huggingface_hub` CLI (`pip install huggingface_hub`).

### Configure environment

```bash
cp .env.example .env
# fill in:
#   GROQ_API_KEY=...
#   REMOTE_PROVIDER=groq        # or fireworks
#   ROUTING_THRESHOLD=0.75
```

### Verify NPU is available

```python
import openvino as ov
print(ov.Core().available_devices)  # must include "NPU"
```

### Run the server

```bash
uvicorn src.api.main:app --port 8000
```

First startup compiles the NPU blob (~30s). Subsequent startups use cache (~6s).

---

## API

### `GET /health`

```json
{ "status": "ok", "npu_ready": true }
```

### `POST /route`

**Request:**
```json
{
  "query": "What is the capital of France?",
  "context": [],
  "constraints": {
    "max_latency_ms": null,
    "prefer_local": false
  },
  "metadata": {
    "session_id": "abc123"
  },
  "classifier": "rule_based"
}
```

`classifier` options: `"rule_based"` (default, instant) or `"llm_judged"` (~10s overhead).

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "route_taken": "local",
  "classifier_used": "rule_based",
  "classifier_signals": {
    "word_count": 6,
    "has_code": false,
    "is_simple_pattern": true,
    "requires_current": false,
    "complexity_score": -2
  },
  "classifier_latency_ms": 0,
  "fallback_triggered": false,
  "fallback_reason": null,
  "tokens": { "completion": 78, "prompt": null },
  "latency_ms": 9072,
  "model_used": "Phi-3.5-mini-instruct-int4-gq"
}
```

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Classifiers

### Rule-Based Classifier (`rule_based`)

Pure heuristic, zero inference cost. Scores queries using:

- Keyword sets: code (`write a function`, `implement`, `debug`...), reasoning (`prove`, `derive`, `explain why`...), current-knowledge (`current`, `latest`, `today`...)
- Pattern matching: simple question forms (`^what is`, `^who is`, `^how many`...)
- Structural signals: word count, multi-part question detection

**Known limitation:** no concept of knowledge-currency beyond keyword matching. "Who is the current CEO of X" routes correctly; a more subtle current-events query might not.

### LLM-Judged Classifier (`llm_judged`)

Prompts the local Phi-3.5-mini model to self-assess the query before generating an answer. Outputs a structured JSON confidence score:

```json
{
  "confidence": 0.9996,
  "requires_current_knowledge": false,
  "predicted_complexity": "simple",
  "predicted_output_tokens": 40
}
```

**Advantages over rule-based:** catches current-knowledge requirements even without keyword matches, more nuanced complexity estimation.

**Disadvantages:** adds ~10s latency per query (one extra local inference call), and INT4 quantization makes JSON output unreliable — ~20-30% of outputs fail validation and fall back to rule-based. All parsed fields are validated and type-coerced before use; invalid outputs trigger the rule-based fallback, never silently bad values.

**Fallback chain:** LLM judge → JSON validation → if failed, rule-based classifier. The fallback rate itself is logged and reported as a benchmark metric.

---

## Benchmark Results

### Dataset

19 hand-curated queries across 4 categories (9 original + 10 adversarial):

| Category | Count | Routing expectation |
|---|---|---|
| simple_factual | 9 | Local preferred, failure-prone for INT4 |
| current_events | 4 | Remote required |
| code | 3 | Remote required |
| reasoning | 3 | Remote required |

### Accuracy

| Classifier | Correct | Total | Accuracy |
|---|---|---|---|
| rule_based | 17 | 19 | 89.5% |
| llm_judged | 19 | 19 | 100.0% |

### Token Economics (vs always-remote baseline)

| Classifier | Baseline tokens | Actual tokens | Saved | Saving % |
|---|---|---|---|---|
| rule_based | 3,572 | 2,888 | 684 | 19.1% |
| llm_judged | 3,823 | 3,277 | 546 | 14.3% |

### Route Performance

| Classifier | Route | Queries | Accuracy | Avg latency | Completion tokens |
|---|---|---|---|---|---|
| rule_based | local | 9 (47.4%) | 77.8% | 8,929ms | 684 |
| rule_based | remote | 10 (52.6%) | 100.0% | 1,077ms | 2,445 |
| llm_judged | local | 7 (36.8%) | 100.0% | 8,782ms | 546 |
| llm_judged | remote | 12 (63.2%) | 100.0% | 963ms | 2,747 |

### Core Tradeoff

The LLM-judged classifier trades 4.8% token savings for a 10.5 percentage
point accuracy gain over rule-based. The entire accuracy gap is in the
`simple_factual` category (77.8% vs 100%) — the LLM judge correctly identifies
which simple queries the INT4 local model will handle unreliably, and routes
them remote instead. All other categories score 100% with both classifiers.

This confirms the central thesis: smarter routing (at some cost) produces
better accuracy/efficiency tradeoffs than heuristic routing.
---

## Known Limitations and Documented Failure Modes

These are documented findings from the evaluation runs, not hidden bugs.

### 1. INT4 Quantization Artifacts (Local Path)

The `Phi-3.5-mini-instruct-int4-gq-ov` model exhibits recurring text-coherence defects under INT4 group-wise quantization:

**Incoherent token insertion:**
> "The Louvre Museum, which is the world**' tubes to see**."

Expected: "world's most visited museum" or similar. The quantized model consistently inserts the token "tubes" in this context across multiple independent runs — a stable, reproducible artifact.

**Hallucinated supporting detail (fluent hallucination):**
> "Gold is a transition metal... known for its bright, **slightly blue, and slightly green** luster."

Gold has a yellow metallic luster. This is factually wrong, grammatically perfect, and indistinguishable from correct text without domain knowledge. This is the dangerous failure mode — fluent hallucination vs. obvious incoherence.

**Historical hallucination:**
> "Bengaluru was established in 1537 by Vijayanagara Emperor **Krishnadeva Wode Bandara**."

The actual founder was Kempe Gowda I. The model appears to have blended "Krishnadeva Raya" (a real emperor of the era) with a fabricated surname. This is confident, contextually plausible, and entirely wrong.

**Implication for routing policy:** code generation and factual queries requiring precision should route remote. The classifier correctly implements this; these examples validate the routing logic rather than undermining it.

### 2. LLM Judge Reliability

The LLM-judged classifier produces valid, parseable JSON roughly 70–80% of the time on the INT4 quantized model. Common failure modes:

- Trailing commas in JSON values (`"0.98,"` instead of `0.98`)
- String booleans (`"false"` instead of `false`)
- Extremely low token predictions (e.g., `predicted_output_tokens: 2`) that would starve generation

All are handled by the `_parse_and_validate` method; invalid outputs fall back to rule-based silently. The fallback rate is a first-class metric in the eval harness.

### 3. Current-Events Queries

Neither local nor remote path can answer queries requiring live information (current president, latest product releases). The remote model (Groq/Llama-3.3-70B) correctly discloses its knowledge cutoff rather than confabulating. Adding web-search augmentation to the remote path is a documented future extension, not a current capability.

### 4. Local Inference Latency

At ~10–12 tok/s, local NPU inference is slower than the remote path for typical queries. This is a hardware-ceiling limitation of the Core Ultra 7 155H NPU with INT4 quantization. The routing system's value is token-cost reduction, not latency reduction, for the current hardware baseline.

---

## Swapping Providers and Models

### Swap remote provider (e.g., Groq → Fireworks AI)

Change one line in `.env`:
```
REMOTE_PROVIDER=fireworks
FIREWORKS_API_KEY=your_key_here
```

No code changes anywhere. The `RemoteInferenceClient` interface abstracts all provider differences.

### Swap local model (e.g., to Gemma 4 on NPU if OpenVINO checkpoint becomes available)

```python
pipeline = PhiNPUPipeline(
    model_path="./models/gemma4-e2b-int4-ov",
    chat_template="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model",
)
```

The `chat_template` parameter makes `PhiNPUPipeline` model-agnostic. Router, classifiers, and API layer require no changes.

---

## Running Evaluations

```bash
# rule-based classifier benchmark
python eval/benchmark.py --classifier rule_based --output-prefix rule_based

# LLM-judged classifier benchmark
python eval/benchmark.py --classifier llm_judged --output-prefix llm_judged
```

Results are written to `eval/results/` as both JSON and CSV. Open the CSV, fill in the `correct` column (TRUE/FALSE) by comparing `response` against `expected_answer`, then run the scorer:

```bash
python eval/scorer.py --results eval/results/rule_based_TIMESTAMP.csv
```

---

## Future Extensions

- **Web-search augmentation** on the remote path for current-events queries
- **Fine-tuned BERT classifier** trained on labeled query-difficulty data, replacing the LLM-judged heuristic with a proper trained classifier deployable on NPU
- **Gemma 4 E2B on NPU** — pending OpenVINO INT4 checkpoint availability (model released April 2026, NPU-optimized checkpoint not yet confirmed)
- **Streamlit demo UI** — thin wrapper calling the FastAPI `/route` endpoint, displaying route taken and classifier signals per query
- **Streaming responses** — `openvino_genai` supports streamer callbacks (already used for token counting); expose this via FastAPI SSE for real-time local inference output

---

## Commit History (Key Milestones)

```
chore: init project structure
feat: NPU inference pipeline wrapper (Phi-3.5-mini INT4 on Intel AI Boost)
feat: provider-agnostic remote inference client (Groq default, Fireworks swap-ready)
feat: rule-based baseline classifier (with documented knowledge-cutoff limitation)
feat: LLM-judged classifier with validated JSON parsing and safe fallback
feat: routing agent with classifier dispatch, latency constraints, and local-failure fallback
feat: FastAPI server with /health and /route endpoints
fix: restore _build_prompt after model-agnostic refactor, add fallback_reason logging
eval: first benchmark run complete, rule_based 8/9, llm_judged 9/9 on 9-query starter set
```

---

## License

MIT