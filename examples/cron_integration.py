#!/usr/bin/env python3
"""
Tree of Thought Integration Example for Autonomous Crons

This shows exactly how your crons can use ToT for complex decisions.
"""

import json
import subprocess

def mcporter_call(tool: str, **kwargs) -> dict:
    """Helper to call mcporter tools"""
    args = " ".join([f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in kwargs.items()])
    cmd = f"mcporter call tot-engine.{tool} {args}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except:
        return {"error": result.stderr, "raw": result.stdout}

# ============================================================================
# EXAMPLE 1: FIXER CRON — Root Cause Analysis
# ============================================================================

def example_fixer_cron():
    """
    FIXER CRON uses ToT when it encounters an issue with multiple possible causes.
    """
    print("="*70)
    print("🛠️  EXAMPLE: FIXER CRON — Root Cause Analysis")
    print("="*70)
    
    # Step 1: Start the reasoning run
    print("\n1️⃣  Starting ToT run for issue analysis...")
    run = mcporter_call(
        "tot_start_run",
        task_prompt="System experiencing intermittent 500 errors after latest deployment",
        beam_width=3,
        max_depth=4
    )
    
    if not run.get("success"):
        print(f"   ❌ Failed: {run}")
        return
    
    run_id = run["run_id"]
    print(f"   ✅ Run ID: {run_id}")
    
    # Step 2: Get frontier (root node)
    print("\n2️⃣  Getting frontier...")
    frontier = mcporter_call("tot_get_frontier", run_id=run_id)
    root_id = frontier["frontier"][0]["node_id"]
    
    # Step 3: Generate hypotheses (possible causes)
    print("\n3️⃣  Generating root cause hypotheses...")
    hypotheses = mcporter_call(
        "tot_expand_node",
        run_id=run_id,
        parent_id=root_id,
        thought1="Database connection pool exhaustion — check HikariCP metrics",
        score1=0.75,
        thought2="Memory leak in new feature — analyze heap dump",
        score2=0.60,
        thought3="Race condition in async handler — review thread safety",
        score3=0.70
    )
    print(f"   Created {hypotheses['count']} hypotheses")
    
    # Step 4: Get new frontier (sorted by score)
    print("\n4️⃣  Evaluating hypotheses...")
    frontier = mcporter_call("tot_get_frontier", run_id=run_id)
    best_hypothesis = frontier["frontier"][0]
    print(f"   Best hypothesis: {best_hypothesis['thought'][:50]}... (score: {best_hypothesis['score']})")
    
    # Step 5: Deep dive on best hypothesis
    print("\n5️⃣  Deep dive on best hypothesis...")
    deep_dive = mcporter_call(
        "tot_expand_node",
        run_id=run_id,
        parent_id=best_hypothesis["node_id"],
        thought1="HikariCP metrics show connection wait time spike — confirmed pool exhaustion",
        score1=0.90,
        terminal1=True,
        thought2="Pool exhaustion caused by slow query holding connections",
        score2=0.85,
        terminal2=True,
        thought3="Normal pool size but query timeout too high",
        score3=0.60
    )
    print(f"   Added {deep_dive['count']} detailed diagnostics")
    
    # Step 6: Get best path (final conclusion)
    print("\n6️⃣  Getting final conclusion...")
    result = mcporter_call("tot_get_best_path", run_id=run_id)
    path = result.get("path", [])
    
    print(f"\n   🎯 ROOT CAUSE IDENTIFIED:")
    for node in path:
        indent = "  " * node["depth"]
        status = "🎯" if node.get("status") == "terminal" else "→"
        print(f"   {indent}{status} [{node['score']:.2f}] {node['thought'][:60]}")
    
    print(f"\n   ✅ RECOMMENDATION:")
    print(f"   • Increase HikariCP max pool size from 10 to 20")
    print(f"   • Add query timeout of 5 seconds")
    print(f"   • Monitor connection wait time metrics")
    
    return run_id

# ============================================================================
# EXAMPLE 2: CREATOR CRON — Implementation Strategy
# ============================================================================

def example_creator_cron():
    """
    CREATOR CRON uses ToT to decide HOW to implement a new feature.
    """
    print("\n" + "="*70)
    print("🏗️  EXAMPLE: CREATOR CRON — Implementation Strategy")
    print("="*70)
    
    # Step 1: Start run
    print("\n1️⃣  Starting ToT run for feature implementation...")
    run = mcporter_call(
        "tot_start_run",
        task_prompt="Implement real-time notification system for price alerts",
        constraints=["Must handle 10k alerts/minute", "Latency < 100ms", "Budget < 20k INR"],
        beam_width=3,
        max_depth=5
    )
    
    run_id = run["run_id"]
    print(f"   ✅ Run ID: {run_id}")
    
    # Step 2: Get frontier
    frontier = mcporter_call("tot_get_frontier", run_id=run_id)
    root_id = frontier["frontier"][0]["node_id"]
    
    # Step 3: Generate architecture options
    print("\n2️⃣  Generating architecture options...")
    options = mcporter_call(
        "tot_expand_node",
        run_id=run_id,
        parent_id=root_id,
        thought1="Redis Pub/Sub + WebSocket: Simple, fast, but no persistence",
        score1=0.70,
        thought2="Apache Kafka + Consumer workers: Scalable, durable, complex",
        score2=0.65,
        thought3="RabbitMQ + Push notifications: Good routing, moderate complexity",
        score3=0.75
    )
    
    # Step 4: Get frontier and expand best
    frontier = mcporter_call("tot_get_frontier", run_id=run_id)
    best = frontier["frontier"][0]
    
    print(f"\n3️⃣  Deep dive on best option: {best['thought'][:40]}...")
    deep = mcporter_call(
        "tot_expand_node",
        run_id=run_id,
        parent_id=best["node_id"],
        thought1="RabbitMQ + CloudAMQP managed: 15k INR/month, fits budget, zero ops",
        score1=0.85,
        terminal1=True,
        thought2="Self-hosted RabbitMQ on Pi cluster: 5k INR, but ops burden",
        score2=0.70,
        thought3="RabbitMQ + Redis caching: Best performance, 18k INR",
        score3=0.80
    )
    
    # Step 5: Final recommendation
    result = mcporter_call("tot_get_best_path", run_id=run_id)
    path = result.get("path", [])
    
    print(f"\n   🏆 IMPLEMENTATION STRATEGY:")
    for node in path:
        indent = "  " * node["depth"]
        status = "🎯" if node.get("status") == "terminal" else "→"
        print(f"   {indent}{status} [{node['score']:.2f}] {node['thought'][:60]}")
    
    print(f"\n   ✅ FINAL DECISION:")
    print(f"   • Use RabbitMQ on CloudAMQP managed service")
    print(f"   • 15k INR/month, fits budget")
    print(f"   • No operational overhead")
    print(f"   • Can handle 50k+ messages/minute")
    
    return run_id

# ============================================================================
# EXAMPLE 3: EVOLUTION CRON — Strategic Decision
# ============================================================================

def example_evolution_cron():
    """
    EVOLUTION CRON uses ToT for strategic system evolution decisions.
    """
    print("\n" + "="*70)
    print("🧬 EXAMPLE: EVOLUTION CRON — Strategic Decision")
    print("="*70)
    
    run = mcporter_call(
        "tot_start_run",
        task_prompt="What capability should Thunderbolt add next for maximum impact?",
        beam_width=4,
        max_depth=4
    )
    
    run_id = run["run_id"]
    print(f"   ✅ Run ID: {run_id}")
    
    frontier = mcporter_call("tot_get_frontier", run_id=run_id)
    root_id = frontier["frontier"][0]["node_id"]
    
    print("\n2️⃣  Evaluating capability options...")
    options = mcporter_call(
        "tot_expand_node",
        run_id=run_id,
        parent_id=root_id,
        thought1="Voice interface: Natural interaction, but adds complexity",
        score1=0.75,
        thought2="Better memory/retrieval: Long-term context, high user value",
        score2=0.85,
        thought3="Multi-agent collaboration: Scale capability, complex coordination",
        score3=0.70,
        thought4="Proactive suggestions: Anticipate needs, moderate complexity",
        score4=0.80
    )
    
    frontier = mcporter_call("tot_get_frontier", run_id=run_id)
    best = frontier["frontier"][0]
    
    print(f"\n3️⃣  Best option: {best['thought']}")
    
    deep = mcporter_call(
        "tot_expand_node",
        run_id=run_id,
        parent_id=best["node_id"],
        thought1="Upgrade to vector DB + semantic search: High recall, proven tech",
        score1=0.90,
        terminal1=True,
        thought2="Simple keyword + timestamp index: Easier but limited",
        score2=0.65
    )
    
    result = mcporter_call("tot_get_best_path", run_id=run_id)
    path = result.get("path", [])
    
    print(f"\n   🧬 EVOLUTION RECOMMENDATION:")
    for node in path:
        indent = "  " * node["depth"]
        status = "🎯" if node.get("status") == "terminal" else "→"
        print(f"   {indent}{status} [{node['score']:.2f}] {node['thought'][:60]}")
    
    print(f"\n   ✅ NEXT CAPABILITY:")
    print(f"   • Enhanced memory with vector DB (Qdrant/Pinecone)")
    print(f"   • Semantic search over conversation history")
    print(f"   • High user impact, technically feasible")
    
    return run_id

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌳 TREE OF THOUGHT — CRON INTEGRATION EXAMPLES")
    print("="*70)
    print("\nThis demonstrates how your autonomous crons use ToT for decisions.\n")
    
    try:
        # Run examples
        run1 = example_fixer_cron()
        run2 = example_creator_cron()
        run3 = example_evolution_cron()
        
        # Show summary
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        
        summary = subprocess.run(
            "mcporter call tot-engine.tot_list_runs limit=10",
            shell=True, capture_output=True, text=True
        )
        print(summary.stdout)
        
        print("\n✅ All examples completed successfully!")
        print("\nYour crons can now use ToT for:")
        print("  • Root cause analysis (Fixer)")
        print("  • Implementation strategy (Creator)")
        print("  • Strategic evolution (Evolution)")
        print("  • Research direction (Learner)")
        print("  • Self-improvement (Self)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
