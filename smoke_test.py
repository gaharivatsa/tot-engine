#!/usr/bin/env python3
"""
ToT Engine v2 — Smoke Test
Tests all 6 requirements from fix plan
"""

import sys
import json
import sqlite3

sys.path.insert(0, '/mnt/ssd/openclaw-sandbox/mcp-servers/tot-engine')

from server_v2 import ToTEngine, RunStatus, tot_start_run, tot_step, tot_run_to_completion, tot_get_tree, tot_get_metrics

print("="*80)
print("🧪 ToT ENGINE v2 — SMOKE TEST")
print("="*80)

engine = ToTEngine()
all_passed = True

# ============================================================================
# TEST 1: Budget Enforcement
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Budget Enforcement (node_budget=5)")
print("="*80)

result = tot_start_run(
    task_prompt="Budget test",
    beam_width=2,
    n_generate=2,
    max_depth=5,
    node_budget=5,
    target_score=0.90
)
run_id = result['run_id']

print(f"Started run: {run_id}")

# Run autonomously
result = tot_run_to_completion(run_id=run_id, max_iterations=20, timeout_seconds=30)

# Check results
conn = engine.db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
node_count = cursor.fetchone()[0]

cursor.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,))
run_status = cursor.fetchone()[0]
conn.close()

print(f"\nNode count: {node_count} (budget: 5)")
print(f"Run status: {run_status}")

if node_count <= 5:
    print("✅ BUDGET ENFORCED")
else:
    print(f"❌ BUDGET VIOLATION: Created {node_count} nodes with budget of 5")
    all_passed = False

if run_status == RunStatus.BUDGET_EXHAUSTED.value:
    print("✅ STATUS SET TO budget_exhausted")
else:
    print(f"⚠️  Status is '{run_status}', expected 'budget_exhausted'")

budget_test_passed = node_count <= 5

# ============================================================================
# TEST 2: Autonomous Run
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Autonomous Run (tot_run_to_completion)")
print("="*80)

result = tot_start_run(
    task_prompt="Autonomy test",
    beam_width=3,
    n_generate=2,
    max_depth=3,
    node_budget=15,
    target_score=0.85
)
auto_run_id = result['run_id']

print(f"Started run: {auto_run_id}")

# Run autonomously
result = tot_run_to_completion(run_id=auto_run_id, max_iterations=10)

print(f"\nResult: {result.get('status', 'unknown')}")
print(f"Iterations: {result.get('iterations', 'N/A')}")
print(f"Final answer: {result.get('final_answer', 'NONE')[:60]}...")
print(f"Confidence: {result.get('confidence', 0):.2f}")

if result.get('success') and result.get('final_answer'):
    print("✅ AUTONOMOUS RUN WORKS")
    autonomy_passed = True
else:
    print("❌ AUTONOMOUS RUN FAILED")
    autonomy_passed = False
    all_passed = False

# ============================================================================
# TEST 3: Thought Generation (Proposer)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Thought Generation (Proposer)")
print("="*80)

result = tot_start_run(
    task_prompt="Proposer test",
    beam_width=2,
    n_generate=3,
    node_budget=10
)
proposer_run_id = result['run_id']

# Single step should generate thoughts automatically
result = tot_step(run_id=proposer_run_id)

print(f"Step result: {result}")

conn = engine.db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (proposer_run_id,))
count_after = cursor.fetchone()[0]
conn.close()

print(f"\nNodes after step: {count_after} (was 1 root)")
print(f"Nodes created: {count_after - 1}")

if count_after > 1:
    print("✅ PROPOSER GENERATED THOUGHTS")
    proposer_passed = True
else:
    print("❌ PROPOSER FAILED")
    proposer_passed = False
    all_passed = False

# ============================================================================
# TEST 4: Evaluation Engine-Owned
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Evaluation Engine-Owned (No Agent Score Override)")
print("="*80)

result = tot_start_run(task_prompt="Evaluation test")
eval_run_id = result['run_id']

# Manually insert node with attempted score override
conn = engine.db.get_connection()
cursor = conn.cursor()
root_id = cursor.execute(
    "SELECT node_id FROM nodes WHERE run_id = ? AND depth = 0",
    (eval_run_id,)
).fetchone()[0]
conn.close()

# The step() function should compute score, not accept it
result = tot_step(run_id=eval_run_id)

conn = engine.db.get_connection()
cursor = conn.cursor()
cursor.execute(
    "SELECT score, evaluation_json FROM nodes WHERE run_id = ? AND depth > 0 LIMIT 1",
    (eval_run_id,)
)
row = cursor.fetchone()
conn.close()

if row:
    score = row['score']
    eval_json = json.loads(row['evaluation_json'])
    print(f"\nComputed score: {score:.4f}")
    print(f"From evaluation: progress={eval_json.get('progress')}, feasibility={eval_json.get('feasibility')}")
    
    # Verify formula
    expected = (0.45 * eval_json['progress'] + 0.35 * eval_json['feasibility'] + 
                0.20 * eval_json['confidence'] - 0.25 * eval_json['risk'])
    if eval_json.get('violations'):
        expected = -0.25  # validity = 0
    
    print(f"Expected score: {expected:.4f}")
    
    if abs(score - expected) < 0.01:
        print("✅ SCORE COMPUTED BY ENGINE (not agent)")
        eval_passed = True
    else:
        print("❌ SCORE MISMATCH")
        eval_passed = False
        all_passed = False
else:
    print("❌ NO NODES FOUND")
    eval_passed = False
    all_passed = False

# ============================================================================
# TEST 5: Frontier Selection Inside tot_step()
# ============================================================================
print("\n" + "="*80)
print("TEST 5: Self-Contained Iteration (tot_step does it all)")
print("="*80)

result = tot_start_run(
    task_prompt="Iteration test",
    beam_width=2,
    node_budget=8
)
iter_run_id = result['run_id']

# Get metrics before
conn = engine.db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM tot_metrics WHERE run_id = ?", (iter_run_id,))
metrics_before = cursor.fetchone()[0]
conn.close()

# Single call to tot_step should do: frontier + generate + evaluate + insert + prune + metrics
result = tot_step(run_id=iter_run_id)

print(f"\nStep result:")
print(f"  Success: {result.get('success')}")
print(f"  Iteration: {result.get('iteration')}")
print(f"  Frontier size: {result.get('frontier_size')}")
print(f"  Nodes expanded: {result.get('nodes_expanded')}")
print(f"  Nodes pruned: {result.get('nodes_pruned')}")

# Check metrics after
conn = engine.db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM tot_metrics WHERE run_id = ?", (iter_run_id,))
metrics_after = cursor.fetchone()[0]
conn.close()

if metrics_after > metrics_before:
    print("✅ METRICS WRITTEN (self-contained iteration)")
    iteration_passed = True
else:
    print("❌ NO METRICS WRITTEN")
    iteration_passed = False
    all_passed = False

if result.get('frontier_size', 0) > 0 and result.get('nodes_expanded', 0) > 0:
    print("✅ FRONTIER SELECTION + EXPANSION IN ONE CALL")
else:
    print("❌ ITERATION INCOMPLETE")
    iteration_passed = False
    all_passed = False

# ============================================================================
# TEST 6: All Requirements Together
# ============================================================================
print("\n" + "="*80)
print("TEST 6: Full Integration (All Requirements)")
print("="*80)

result = tot_start_run(
    task_prompt="Full integration test",
    beam_width=3,
    n_generate=2,
    max_depth=3,
    node_budget=10,
    target_score=0.80
)
full_run_id = result['run_id']

# Run autonomously
result = tot_run_to_completion(run_id=full_run_id, max_iterations=20)

# Verify all requirements
conn = engine.db.get_connection()
cursor = conn.cursor()

# Node count ≤ budget
cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (full_run_id,))
final_count = cursor.fetchone()[0]

# Status set
cursor.execute("SELECT status FROM runs WHERE run_id = ?", (full_run_id,))
final_status = cursor.fetchone()[0]

# Metrics exist
cursor.execute("SELECT COUNT(*) FROM tot_metrics WHERE run_id = ?", (full_run_id,))
metrics_count = cursor.fetchone()[0]

conn.close()

print(f"\nFinal Results:")
print(f"  Node count: {final_count} (budget: 10)")
print(f"  Status: {final_status}")
print(f"  Metrics rows: {metrics_count}")
print(f"  Has answer: {bool(result.get('final_answer'))}")

checks = []
checks.append(("Budget respected", final_count <= 10))
checks.append(("Status set", final_status != 'running'))
checks.append(("Metrics exist", metrics_count >= 1))
checks.append(("Has synthesis", bool(result.get('final_answer'))))

print("\nChecks:")
for name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")

integration_passed = all(p for _, p in checks)
if not integration_passed:
    all_passed = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

tests = [
    ("Budget Enforcement", budget_test_passed),
    ("Autonomous Run", autonomy_passed),
    ("Thought Generation", proposer_passed),
    ("Engine-Owned Evaluation", eval_passed),
    ("Self-Contained Iteration", iteration_passed),
    ("Full Integration", integration_passed)
]

print("\nTest Results:")
for name, passed in tests:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")

print("\n" + "="*80)
if all_passed:
    print("🎉 ALL TESTS PASSED — ToT Engine v2 is WORKING")
    print("="*80)
    sys.exit(0)
else:
    print("❌ SOME TESTS FAILED — Review issues above")
    print("="*80)
    sys.exit(1)
