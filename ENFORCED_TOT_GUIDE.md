# Enforced Exhaustive Tree of Thought Research

## Overview

A wrapper system that **enforces minimum budget consumption** for ToT research. No shortcuts possible - the system will consume allocated resources before allowing synthesis.

## The Problem with Standard ToT

```python
# Before: Could stop early
run = start_tot_run(node_budget=200)
# ... 3 iterations later ...
result = get_best_path(run)  # Only 13 nodes used! ❌

# After: ENFORCED consumption
result = enforced_tot_research(
    depth_level="deep"  # MUST use 135+ of 150 nodes ✅
)
```

## Depth Levels

| Level | Budget | Min Required | Validation | Literature | Time |
|-------|--------|--------------|------------|------------|------|
| **shallow** | 20 | 16 (80%) | ❌ | ❌ | 5 min |
| **medium** | 50 | 43 (85%) | 1 test | ✅ | 30 min |
| **deep** | 150 | 135 (90%) | 3 tests | ✅ | 1-2 hrs |
| **exhaustive** | 500 | 475 (95%) | 10 tests | ✅ | 4-8 hrs |

## Usage

### Python API

```python
from enforced_tot import enforced_tot_research

# DEEP research - enforced 135+ nodes
result = enforced_tot_research(
    task_prompt="Design self-modelling architecture...",
    depth_level="deep",
    constraints=[
        "Pi 4 compatible",
        "RLHF-free",
        "Single developer maintainable"
    ]
)

print(f"Nodes used: {result['nodes_explored']}/{result['node_budget']}")
print(f"Requirement met: {result['requirement_met']}")  # True if 135+ used
print(f"Confidence: {result['confidence']}")
print(f"Recommendation: {result['recommendation']}")
```

### CLI

```bash
# Shallow check
python3 .thunderbolt/enforced_tot.py \
  --prompt "Which database?" \
  --depth shallow

# Deep research with validation
python3 .thunderbolt/enforced_tot.py \
  --prompt "Design self-modelling architecture..." \
  --depth deep \
  --constraint "Pi 4" \
  --constraint "RLHF-free" \
  --output research/result.json

# Exhaustive (publication-grade)
python3 .thunderbolt/enforced_tot.py \
  --prompt "Design autonomous agent architecture..." \
  --depth exhaustive
```

## Enforcement Mechanisms

### 1. Minimum Consumption Ratio

```python
min_required = int(node_budget * min_consumption_ratio)

while nodes_used < min_required:
    # Continue expanding - NO EARLY EXIT
    expand_frontier()
    
if nodes_used < min_required:
    # Warning if under-utilized
    print("WARNING: Only used X/Y required nodes")
```

### 2. Forced Continuation

```python
if should_stop and nodes_used < min_required:
    # Natural stop but budget under-utilized
    print("Forcing deeper exploration...")
    continue  # Keep going!
```

### 3. Empirical Validation (Deep+)

```python
if config.validation_required:
    for test_id in range(config.empirical_tests):
        result = run_empirical_test(architecture, test_id)
        # Must PASS all tests
        if not result["test_passed"]:
            return {"success": False, "error": "Validation failed"}
```

### 4. Sensitivity Analysis (Deep+)

```python
for run in range(config.sensitivity_runs):
    winner = run_tot_with_seed(run)
    winners.append(winner)

consistency = count_most_common(winners) / len(winners)
# Warn if < 60% consistency
```

### 5. Literature Integration (Medium+)

```python
if config.literature_search:
    sources = search_literature(query)
    # Feed into candidate generation
    candidates = generate_with_sources(sources)
```

## Result Structure

```json
{
  "success": true,
  "depth": "deep",
  "recommendation": "Observer + Resource Guardian...",
  "confidence": 0.8085,
  "nodes_explored": 135,
  "node_budget": 150,
  "budget_utilization": 0.90,
  "requirement_met": true,
  "iterations": 8,
  "sources_consulted": 12,
  "validation_results": [
    {"test_passed": true, "cpu_overhead": 1.2, ...},
    {"test_passed": true, "cpu_overhead": 1.3, ...},
    {"test_passed": true, "cpu_overhead": 1.1, ...}
  ],
  "sensitivity_analysis": {
    "runs": 3,
    "consistency": 1.0,
    "most_common": "Resource Guardian"
  },
  "peer_review": {
    "critiques": [...],
    "mitigations": [...]
  }
}
```

## Files Created

| File | Purpose |
|------|---------|
| `.thunderbolt/enforced_tot.py` | Main enforcement engine (30KB) |
| `research/demo_enforced_tot.py` | Usage demonstration |
| `ENFORCED_TOT_GUIDE.md` | This documentation |

## Key Differences from Standard ToT

| Aspect | Standard ToT | Enforced ToT |
|--------|--------------|--------------|
| Budget usage | Optional (13/200 typical) | **Mandatory** (180/200 enforced) |
| Early stopping | Allowed | **Prevented** |
| Validation | None | **Required** (deep+) |
| Literature | Optional | **Mandatory** (medium+) |
| Consistency check | None | **Required** (deep+) |
| Peer review | None | **Required** (exhaustive) |
| Research time | 10-30 min | 30 min - 8 hours |
| Result confidence | Medium | **High** |

## Example: Before vs After

### Before (Shallow)
```
Nodes: 13/200 (6.5%) ❌
Validation: None ❌
Sources: 0 ❌
Time: 5 minutes
Confidence: Unreliable
```

### After (Deep - Enforced)
```
Nodes: 135/150 (90%) ✅
Validation: 3/3 passed ✅
Sources: 12 consulted ✅
Sensitivity: 100% consistency ✅
Time: 1.5 hours
Confidence: 0.85 (validated)
```

## When to Use Each Level

| Level | Use Case | Example |
|-------|----------|---------|
| **shallow** | Quick directional check | "Should I use Redis or SQLite?" |
| **medium** | Balanced exploration | "Which cron schedule is optimal?" |
| **deep** | Important architectural decisions | "Self-modelling architecture design" |
| **exhaustive** | Publication-grade, critical systems | "Autonomous trading system architecture" |

## Next Steps

1. **Test shallow run:**
   ```bash
   python3 .thunderbolt/enforced_tot.py -p "Test question" -d shallow
   ```

2. **Run deep research:**
   ```bash
   python3 .thunderbolt/enforced_tot.py \
     -p "Design self-modelling architecture for Thunderbolt" \
     -d deep \
     -c "Pi 4" -c "RLHF-free" -c "Single developer"
   ```

3. **Integrate into crons:**
   ```python
   # In learner/creator crons
   from enforced_tot import enforced_tot_research
   
   if is_complex_decision(question):
       result = enforced_tot_research(
           task_prompt=question,
           depth_level="medium"
       )
   ```

## Summary

The `enforced_tot.py` wrapper transforms ToT from an **optional exploration tool** into a **rigorous research methodology** with:

- ✅ Mandatory budget consumption (80-95%)
- ✅ Empirical validation for deep levels
- ✅ Literature integration
- ✅ Sensitivity analysis
- ✅ Peer review for exhaustive level

**No more shallow research. Depth is enforced.**
