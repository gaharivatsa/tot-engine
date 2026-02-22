# Tree of Thought Engine — Usage Guide (Option B)

## Architecture

**Option B: Client Sampling Pattern**

```
You (Thunderbolt)                    ToT Engine
     │                                    │
     │  1. tot_start_run()                │
     │────────────────────────────────────>│
     │                                    │
     │  2. tot_request_samples()          │
     │────────────────────────────────────>│
     │                                    │
     │<─ 3. "Please generate thoughts" ────│
     │     for frontier nodes             │
     │                                    │
     │  4. You use LLM to generate        │
     │     candidates                     │
     │                                    │
     │  5. tot_submit_samples()           │
     │────────────────────────────────────>│
     │     with generated thoughts        │
     │                                    │
     │<─ 6. Engine evaluates & inserts ───│
     │                                    │
     │  (repeat 2-6 until stop)           │
     │                                    │
     │  7. tot_get_best_path()            │
     │────────────────────────────────────>│
     │<─ 8. Final synthesis ──────────────│
```

## Quick Start

```python
import json

# 1. Start a reasoning run
result = mcporter_call("tot_start_run",
    task_prompt="Should we use Redis or PostgreSQL for caching?",
    node_budget=10,
    beam_width=2,
    n_generate=2
)
run_id = result['run_id']

# 2. Request samples
request = mcporter_call("tot_request_samples", run_id=run_id)
# Returns frontier nodes that need expansion

# 3. YOU generate thoughts using your LLM
for sample_req in request['sample_requests']:
    parent_thought = sample_req['request']['parent_thought']
    
    # Use your LLM (juspay/kimi-latest) to generate candidates
    candidates = generate_with_llm(
        task="Should we use Redis or PostgreSQL for caching?",
        parent=parent_thought,
        num_candidates=2
    )

# 4. Submit samples back
samples = [{
    "parent_node_id": "...",
    "candidates": [
        {
            "thought": "Redis: Fast in-memory, eventual consistency",
            "delta": {"approach": "redis"},
            "progress_estimate": 0.8,
            "feasibility_estimate": 0.9,
            "risk_estimate": 0.2,
            "reasoning": "Best for pure cache use case"
        },
        {
            "thought": "PostgreSQL: ACID compliance, relational",
            "delta": {"approach": "postgres"},
            "progress_estimate": 0.7,
            "feasibility_estimate": 0.85,
            "risk_estimate": 0.3,
            "reasoning": "Better if we need persistence"
        }
    ]
}]

result = mcporter_call("tot_submit_samples", 
    run_id=run_id, 
    samples=json.dumps(samples)
)

# 5. Repeat until stop condition
# 6. Get final answer
result = mcporter_call("tot_get_best_path", run_id=run_id)
print(result['final_answer'])
print(f"Confidence: {result['confidence']}")
```

## Why Option B is Better

| Feature | Benefit |
|---------|---------|
| **You generate thoughts** | Use your LLM, context, memory |
| **Full visibility** | See every candidate before it enters tree |
| **Steerable** | Modify/correct thoughts mid-search |
| **No API keys in engine** | Uses your existing OpenClaw setup |
| **Strict validation** | Pydantic schemas reject bad inputs |
| **Atomic budget** | `node_budget` strictly enforced |

## Key Features

### 1. Strict Budget Enforcement
- `node_budget` = **total nodes** including root
- Checked **atomically** before every insertion
- Returns `budget_exhausted` status when hit

### 2. Engine-Owned Evaluation
You provide estimates, engine computes:
```
score = validity * (0.45*progress + 0.35*feasibility + 0.20*confidence) - 0.25*risk
violations → score = 0 (pruned)
```

### 3. Deduplication
- `(run_id, parent_id, state_hash, thought_hash)` unique constraint
- Duplicate thoughts rejected automatically

### 4. Pydantic Validation
All inputs validated:
- `progress`, `feasibility`, `confidence`, `risk` ∈ [0, 1]
- `thought` minimum length 5
- `reasoning` required

## Available Tools

| Tool | Purpose |
|------|---------|
| `tot_start_run()` | Initialize with strict schema |
| `tot_request_samples()` | Get frontier, request generation |
| `tot_submit_samples()` | Submit generated candidates |
| `tot_get_frontier()` | Check current frontier |
| `tot_get_best_path()` | Get synthesized final answer |
| `tot_get_metrics()` | View search metrics |
| `tot_list_runs()` | List all runs |
| `tot_cancel_run()` | Cancel a run |

## Schema Reference

### State
```json
{
  "facts": ["string"],        // min 1 required
  "assumptions": ["string"],
  "work": {},
  "constraints": ["string"]
}
```

### Evaluation (you provide)
```json
{
  "progress": 0.0-1.0,        // toward goal
  "feasibility": 0.0-1.0,     // can we do it
  "confidence": 0.0-1.0,      // certainty
  "risk": 0.0-1.0,            // downsides
  "violations": ["string"],   // constraint breaks
  "reasoning": "string"       // min 5 chars
}
```

### ThoughtCandidate (you generate)
```json
{
  "thought": "string",        // the reasoning step
  "delta": {},                // what changed
  "progress_estimate": 0.0-1.0,
  "feasibility_estimate": 0.0-1.0,
  "risk_estimate": 0.0-1.0,
  "reasoning": "string"
}
```

## Example: Architecture Decision

```python
# Start
run = tot_start_run(
    task_prompt="Microservices vs Monolith for new feature",
    node_budget=15,
    target_score=0.85
)

# First expansion
request = tot_request_samples(run_id=run['run_id'])
# You generate:
# - "Microservices: Scalable but complex"
# - "Monolith: Simple but harder to scale"
# - "Modular monolith: Best of both"

# Submit, engine evaluates & inserts
# Repeat...

# Final answer
result = tot_get_best_path(run_id=run['run_id'])
# → "Modular monolith with clean boundaries"
# → Confidence: 0.87
```

## Status

**Version**: 2.0 (Option B)  
**Pattern**: MCP Client Sampling  
**Grade**: A- (Production Ready)

---

*Last updated: 2026-02-22*
