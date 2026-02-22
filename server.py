#!/usr/bin/env python3
"""
Tree of Thought Engine — Option B Implementation
Server requests thought generation from client (MCP sampling pattern)
"""

import os
import sys
import json
import sqlite3
import uuid
import hashlib
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = os.environ.get("TOT_DB_PATH", "/mnt/ssd/openclaw-sandbox/.thunderbolt/tot.db")

# ============================================================================
# Pydantic Schemas (Strict Validation)
# ============================================================================

class State(BaseModel):
    """Evolving world state — strictly validated"""
    facts: List[str] = Field(default_factory=list, min_length=1, description="Verified facts")
    assumptions: List[str] = Field(default_factory=list)
    work: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    
    @field_validator('facts')
    def facts_not_empty(cls, v):
        if not v:
            raise ValueError('State must have at least one fact')
        return v

class Evaluation(BaseModel):
    """Structured evaluation — engine computes final score"""
    progress: float = Field(ge=0.0, le=1.0, description="Progress toward goal (0-1)")
    feasibility: float = Field(ge=0.0, le=1.0, description="Can we actually do this (0-1)")
    confidence: float = Field(ge=0.0, le=1.0, description="Certainty in evaluation (0-1)")
    risk: float = Field(ge=0.0, le=1.0, description="Potential downsides (0-1)")
    violations: List[str] = Field(default_factory=list, description="Constraint violations")
    reasoning: str = Field(min_length=3, description="Brief explanation")
    
    def compute_score(self) -> float:
        """Server-computed score — agents NEVER set this directly"""
        validity = 0.0 if self.violations else 1.0
        base_score = 0.45 * self.progress + 0.35 * self.feasibility + 0.20 * self.confidence
        return validity * base_score - 0.25 * self.risk

class SearchConfig(BaseModel):
    """Search configuration"""
    beam_width: int = Field(default=3, ge=1, le=10)
    n_generate: int = Field(default=2, ge=1, le=5, description="Candidates per expansion")
    max_depth: int = Field(default=5, ge=1, le=20)
    node_budget: int = Field(default=20, ge=2, le=1000, description="TOTAL nodes including root")
    target_score: float = Field(default=0.85, ge=0.0, le=1.0)

class ThoughtRequest(BaseModel):
    """Request for thought generation (sent to client)"""
    task_prompt: str = Field(description="Original problem statement")
    parent_thought: str = Field(description="Thought we're expanding from")
    parent_state: State = Field(description="Current state")
    num_candidates: int = Field(default=3, ge=1, le=5)
    
class ThoughtCandidate(BaseModel):
    """Single candidate thought from client"""
    thought: str = Field(min_length=3, description="The reasoning step")
    delta: Dict[str, Any] = Field(default_factory=dict, description="What changed from parent")
    progress_estimate: float = Field(ge=0.0, le=1.0)
    feasibility_estimate: float = Field(ge=0.0, le=1.0)
    risk_estimate: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=3)

class ThoughtResponse(BaseModel):
    """Client response with generated thoughts"""
    candidates: List[ThoughtCandidate] = Field(min_items=1, max_items=5)

# ============================================================================
# Database
# ============================================================================

class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task_prompt TEXT NOT NULL,
                constraints_json TEXT DEFAULT '[]',
                search_config_json TEXT NOT NULL,
                best_node_id TEXT,
                status TEXT DEFAULT 'running',
                target_score REAL DEFAULT 0.85,
                final_answer TEXT,
                synthesis_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                parent_id TEXT,
                depth INTEGER NOT NULL,
                thought TEXT NOT NULL,
                state_json TEXT NOT NULL,
                delta_json TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                score REAL NOT NULL,
                status TEXT DEFAULT 'active',
                state_hash TEXT NOT NULL,
                thought_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tot_metrics (
                metric_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                frontier_size INTEGER NOT NULL,
                nodes_expanded INTEGER NOT NULL,
                nodes_pruned INTEGER NOT NULL,
                best_score REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Critical: Unique constraint for deduplication
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_node 
            ON nodes(run_id, parent_id, state_hash, thought_hash)
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_depth ON nodes(run_id, depth)")
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

# ============================================================================
# ToT Engine — Option B (Client Sampling)
# ============================================================================

class RunStatus(Enum):
    RUNNING = "running"
    SOLVED = "solved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEPTH_LIMIT = "depth_limit"
    WAITING_FOR_SAMPLES = "waiting_for_samples"
    COMPLETED = "completed"

class ToTEngine:
    def __init__(self):
        self.db = Database()
        self.pending_samples: Dict[str, Dict] = {}  # run_id -> sample request
    
    def check_budget(self, run_id: str, config: SearchConfig, conn) -> Tuple[bool, str]:
        """
        REQUIREMENT #2: Atomic budget enforcement
        Check BEFORE any expansion. Budget = total nodes including root.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
        current_count = cursor.fetchone()[0]
        
        if current_count >= config.node_budget:
            # Update status atomically
            cursor.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (RunStatus.BUDGET_EXHAUSTED.value, datetime.now().isoformat(), run_id)
            )
            conn.commit()
            return False, f"budget_exhausted ({current_count}/{config.node_budget})"
        
        return True, ""
    
    def select_frontier(self, run_id: str, beam_width: int, conn) -> List[Dict]:
        """Select top-k active nodes at current max depth"""
        cursor = conn.cursor()
        
        # Get max depth with active nodes
        cursor.execute(
            "SELECT MAX(depth) FROM nodes WHERE run_id = ? AND status IN ('active', 'terminal')",
            (run_id,)
        )
        max_depth = cursor.fetchone()[0] or 0
        
        # Get frontier at that depth
        cursor.execute(
            """SELECT node_id, thought, score, depth, state_json, status 
               FROM nodes 
               WHERE run_id = ? AND depth = ? AND status IN ('active', 'terminal')
               ORDER BY score DESC 
               LIMIT ?""",
            (run_id, max_depth, beam_width)
        )
        
        return [dict(row) for row in cursor.fetchall()]
    
    def should_stop(self, run_id: str, config: SearchConfig, conn) -> Tuple[bool, str]:
        """Check all stop conditions"""
        cursor = conn.cursor()
        
        # Target achieved?
        cursor.execute(
            """SELECT score FROM nodes 
               WHERE run_id = ? AND status = 'terminal'
               ORDER BY score DESC LIMIT 1""",
            (run_id,)
        )
        best = cursor.fetchone()
        if best and best['score'] >= config.target_score:
            return True, "target_achieved"
        
        # Depth limit?
        cursor.execute("SELECT MAX(depth) FROM nodes WHERE run_id = ?", (run_id,))
        if (cursor.fetchone()[0] or 0) >= config.max_depth:
            return True, "depth_limit"
        
        return False, "continue"
    
    def request_samples(self, run_id: str, frontier: List[Dict], num_candidates: int) -> Dict:
        """
        REQUIREMENT #1 (Option B): Request thought generation from client
        Returns sample request that client should fulfill
        """
        requests = []
        for node in frontier:
            # Parse state
            try:
                state_data = json.loads(node.get('state_json', '{}'))
                state = State(**state_data)
            except:
                state = State(facts=[node['thought']])
            
            req = ThoughtRequest(
                task_prompt="",  # Would be populated from run
                parent_thought=node['thought'],
                parent_state=state,
                num_candidates=num_candidates
            )
            
            requests.append({
                'node_id': node['node_id'],
                'request': req.dict()
            })
        
        # Store pending request
        self.pending_samples[run_id] = {
            'frontier': frontier,
            'requests': requests,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "status": "waiting_for_samples",
            "message": f"Please generate {num_candidates} candidates for each frontier node",
            "sample_requests": requests,
            "instruction": "Call tot_submit_samples() with generated candidates"
        }
    
    def step_with_samples(self, run_id: str, samples: List[Dict]) -> Dict:
        """
        Process submitted samples from client
        REQUIREMENT #5: Self-contained iteration with client-provided thoughts
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get run config
        cursor.execute("SELECT search_config_json, task_prompt FROM runs WHERE run_id = ?", (run_id,))
        run = cursor.fetchone()
        if not run:
            conn.close()
            return {"success": False, "error": "Run not found"}
        
        config = SearchConfig(**json.loads(run['search_config_json']))
        
        # Check stop conditions
        should_stop, stop_reason = self.should_stop(run_id, config, conn)
        if should_stop:
            cursor.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (RunStatus.COMPLETED.value, datetime.now().isoformat(), run_id)
            )
            conn.commit()
            conn.close()
            return {
                "success": True,
                "status": "stopped",
                "stop_reason": stop_reason
            }
        
        # Check budget (atomic, before any insertion)
        can_expand, budget_reason = self.check_budget(run_id, config, conn)
        if not can_expand:
            conn.close()
            return {
                "success": True,
                "status": "stopped",
                "stop_reason": budget_reason
            }
        
        # Process samples
        iteration = self._get_iteration(run_id, conn) + 1
        nodes_expanded = 0
        nodes_pruned = 0
        
        for sample in samples:
            parent_id = sample.get('parent_node_id')
            candidates = sample.get('candidates', [])
            
            if not parent_id or not candidates:
                continue
            
            # Get parent info
            cursor.execute(
                "SELECT depth, state_json FROM nodes WHERE node_id = ?",
                (parent_id,)
            )
            parent = cursor.fetchone()
            if not parent:
                continue
            
            parent_depth = parent['depth']
            
            # Check depth limit
            if parent_depth >= config.max_depth:
                continue
            
            for cand_data in candidates:
                try:
                    # Validate candidate
                    cand = ThoughtCandidate(**cand_data)
                    
                    # Check budget before EACH insert
                    cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
                    if cursor.fetchone()[0] >= config.node_budget:
                        cursor.execute(
                            "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                            (RunStatus.BUDGET_EXHAUSTED.value, datetime.now().isoformat(), run_id)
                        )
                        conn.commit()
                        conn.close()
                        return {
                            "success": True,
                            "status": "stopped",
                            "stop_reason": f"budget_exhausted during expansion"
                        }
                    
                    # REQUIREMENT #4: Engine owns evaluation
                    eval_obj = Evaluation(
                        progress=cand.progress_estimate,
                        feasibility=cand.feasibility_estimate,
                        confidence=0.7,  # Default for client-provided
                        risk=cand.risk_estimate,
                        violations=[],  # Engine could add validation here
                        reasoning=cand.reasoning
                    )
                    score = eval_obj.compute_score()
                    
                    # Determine status
                    if eval_obj.violations:
                        status = "pruned"
                        nodes_pruned += 1
                    elif score >= config.target_score:
                        status = "terminal"
                    else:
                        status = "active"
                    
                    # Compute hashes
                    delta_json = json.dumps(cand.delta)
                    state_hash = hashlib.sha256(delta_json.encode()).hexdigest()[:16]
                    thought_hash = hashlib.sha256(cand.thought.lower().strip().encode()).hexdigest()[:16]
                    
                    # Check duplicate
                    cursor.execute(
                        """SELECT 1 FROM nodes WHERE run_id = ? AND parent_id = ? 
                           AND state_hash = ? AND thought_hash = ?""",
                        (run_id, parent_id, state_hash, thought_hash)
                    )
                    if cursor.fetchone():
                        nodes_pruned += 1
                        continue
                    
                    # Insert
                    node_id = str(uuid.uuid4())
                    cursor.execute(
                        """INSERT INTO nodes 
                           (node_id, run_id, parent_id, depth, thought, state_json, delta_json,
                            evaluation_json, score, status, state_hash, thought_hash)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (node_id, run_id, parent_id, parent_depth + 1,
                         cand.thought,
                         json.dumps({'parent_state': parent['state_json'], 'delta': cand.delta}),
                         delta_json,
                         json.dumps(eval_obj.dict()),
                         score, status, state_hash, thought_hash)
                    )
                    nodes_expanded += 1
                    
                    # Update best if terminal
                    if status == "terminal":
                        cursor.execute(
                            "UPDATE runs SET best_node_id = ? WHERE run_id = ?",
                            (node_id, run_id)
                        )
                        
                except Exception as e:
                    # Log error for debugging
                    print(f"DEBUG: Candidate failed: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Clear pending samples
        if run_id in self.pending_samples:
            del self.pending_samples[run_id]
        
        # Write metrics
        cursor.execute(
            """INSERT INTO tot_metrics 
               (metric_id, run_id, iteration, frontier_size, nodes_expanded, nodes_pruned, best_score)
               VALUES (?, ?, ?, ?, ?, ?, 
                       COALESCE((SELECT MAX(score) FROM nodes WHERE run_id = ?), 0))""",
            (str(uuid.uuid4()), run_id, iteration, len(samples), nodes_expanded, nodes_pruned, run_id)
        )
        
        cursor.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "iteration": iteration,
            "nodes_expanded": nodes_expanded,
            "nodes_pruned": nodes_pruned,
            "status": "running"
        }
    
    def _get_iteration(self, run_id: str, conn) -> int:
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(iteration), 0) FROM tot_metrics WHERE run_id = ?", (run_id,))
        return cursor.fetchone()[0]
    
    def synthesize(self, run_id: str) -> Dict:
        """Generate final answer"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get best terminal, or best active
        cursor.execute(
            """SELECT node_id, thought, score, status FROM nodes 
               WHERE run_id = ? AND status = 'terminal'
               ORDER BY score DESC LIMIT 1""",
            (run_id,)
        )
        best = cursor.fetchone()
        
        if not best:
            cursor.execute(
                """SELECT node_id, thought, score, status FROM nodes 
                   WHERE run_id = ? AND status IN ('active', 'terminal')
                   ORDER BY score DESC LIMIT 1""",
                (run_id,)
            )
            best = cursor.fetchone()
        
        if not best:
            conn.close()
            return {
                "success": False,
                "error": "No solution found",
                "final_answer": "Search did not produce viable solutions",
                "confidence": 0.0
            }
        
        best = dict(best)
        
        # Trace path back
        path = []
        current_id = best['node_id']
        while current_id:
            cursor.execute(
                "SELECT node_id, parent_id, depth, thought, score, status FROM nodes WHERE node_id = ?",
                (current_id,)
            )
            node = cursor.fetchone()
            if not node:
                break
            path.append(dict(node))
            current_id = node['parent_id']
        
        path.reverse()
        
        synthesis = {
            "final_answer": best['thought'],
            "confidence": best['score'],
            "path_length": len(path),
            "is_terminal": best.get('status') == 'terminal',
            "reasoning_chain": [{"depth": p['depth'], "thought": p['thought'][:80], "score": p['score']} for p in path]
        }
        
        cursor.execute(
            "UPDATE runs SET synthesis_json = ?, final_answer = ? WHERE run_id = ?",
            (json.dumps(synthesis), best['thought'], run_id)
        )
        conn.commit()
        conn.close()
        
        return {"success": True, **synthesis}

# ============================================================================
# MCP Server
# ============================================================================

mcp = FastMCP("Tree of Thought Engine — Option B (Client Sampling)")
engine = ToTEngine()

@mcp.tool()
def tot_start_run(
    task_prompt: str,
    constraints: List[str] = None,
    beam_width: int = 3,
    n_generate: int = 2,
    max_depth: int = 5,
    node_budget: int = 20,
    target_score: float = 0.85
) -> Dict:
    """Initialize a new ToT run with strict schema validation"""
    
    if constraints is None:
        constraints = []
    
    config = SearchConfig(
        beam_width=beam_width,
        n_generate=n_generate,
        max_depth=max_depth,
        node_budget=node_budget,
        target_score=target_score
    )
    
    run_id = str(uuid.uuid4())
    
    conn = engine.db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT INTO runs (run_id, task_prompt, constraints_json, search_config_json, status)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, task_prompt, json.dumps(constraints), json.dumps(config.dict()), "running")
    )
    
    # Create root node (counts toward budget)
    root_id = str(uuid.uuid4())
    root_state = State(facts=[f"Task: {task_prompt}"])
    root_eval = Evaluation(progress=0.0, feasibility=1.0, confidence=1.0, risk=0.0, violations=[], reasoning="Root")
    
    cursor.execute(
        """INSERT INTO nodes 
           (node_id, run_id, parent_id, depth, thought, state_json, delta_json,
            evaluation_json, score, status, state_hash, thought_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (root_id, run_id, None, 0, f"ROOT: {task_prompt}",
         root_state.json(), json.dumps({"root": True}),
         root_eval.json(), root_eval.compute_score(), "active",
         hashlib.sha256(b"root").hexdigest()[:16],
         hashlib.sha256(task_prompt.lower().encode()).hexdigest()[:16])
    )
    
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "run_id": run_id,
        "root_node_id": root_id,
        "config": config.dict(),
        "message": "Run started. Call tot_request_samples() to begin expansion."
    }

@mcp.tool()
def tot_request_samples(run_id: str) -> Dict:
    """
    REQUIREMENT #1 (Option B): Request thought generation from client
    
    This initiates the sampling pattern:
    1. Engine identifies frontier nodes
    2. Returns sample requests to client
    3. Client (you) generates candidates using LLM
    4. Client calls tot_submit_samples() with results
    """
    conn = engine.db.get_connection()
    cursor = conn.cursor()
    
    # Get config
    cursor.execute("SELECT search_config_json FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    config = SearchConfig(**json.loads(run['search_config_json']))
    
    # Select frontier
    frontier = engine.select_frontier(run_id, config.beam_width, conn)
    conn.close()
    
    if not frontier:
        return {"success": False, "error": "No frontier nodes to expand"}
    
    # Return sample requests
    return engine.request_samples(run_id, frontier, config.n_generate)

@mcp.tool()
def tot_submit_samples(
    run_id: str,
    samples: List[Dict] = Field(..., description="List of {parent_node_id, candidates[]}")
) -> Dict:
    """
    Submit generated thoughts from client
    
    Each sample should have:
    - parent_node_id: Which node to expand
    - candidates: List of ThoughtCandidate objects
    
    Engine will evaluate, score, and insert valid candidates
    """
    return engine.step_with_samples(run_id, samples)

@mcp.tool()
def tot_run_to_completion(
    run_id: str,
    max_iterations: int = Field(default=50, ge=1, le=200),
    timeout_seconds: int = Field(default=120, ge=10, le=600)
) -> Dict:
    """
    REQUIREMENT #2: Run full ToT search autonomously
    
    In Option B, this still requires client interaction for sampling,
    but can be automated if the client provides a sampling callback.
    
    For now, returns guidance on manual iteration.
    """
    return {
        "success": True,
        "status": "requires_sampling",
        "message": "Option B requires client-side sampling. Use tot_request_samples() + tot_submit_samples() iteratively.",
        "workflow": [
            "1. tot_request_samples(run_id) → get frontier",
            "2. Generate candidates using your LLM",
            "3. tot_submit_samples(run_id, samples) → expand",
            "4. Repeat until stop condition"
        ]
    }

@mcp.tool()
def tot_get_frontier(run_id: str) -> Dict:
    """Get current frontier nodes"""
    conn = engine.db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT search_config_json FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    config = SearchConfig(**json.loads(run['search_config_json']))
    frontier = engine.select_frontier(run_id, config.beam_width, conn)
    
    # Check stop
    should_stop, reason = engine.should_stop(run_id, config, conn)
    
    conn.close()
    
    return {
        "success": True,
        "frontier": frontier,
        "should_stop": should_stop,
        "stop_reason": reason if should_stop else None
    }

@mcp.tool()
def tot_get_best_path(run_id: str) -> Dict:
    """Get synthesized final answer"""
    return engine.synthesize(run_id)

@mcp.tool()
def tot_get_metrics(run_id: str) -> Dict:
    """Get run metrics"""
    conn = engine.db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tot_metrics WHERE run_id = ? ORDER BY iteration", (run_id,))
    metrics = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute(
        """SELECT COUNT(*) as total,
                  SUM(CASE WHEN status='terminal' THEN 1 ELSE 0 END) as terminal,
                  SUM(CASE WHEN status='pruned' THEN 1 ELSE 0 END) as pruned,
                  MAX(score) as best
           FROM nodes WHERE run_id = ?""",
        (run_id,)
    )
    summary = dict(cursor.fetchone())
    
    conn.close()
    
    return {"success": True, "metrics": metrics, "summary": summary}

@mcp.tool()
def tot_list_runs(limit: int = 10) -> Dict:
    """List recent runs"""
    conn = engine.db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT run_id, task_prompt, status, created_at FROM runs 
           ORDER BY created_at DESC LIMIT ?""",
        (limit,)
    )
    runs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"success": True, "runs": runs}

@mcp.tool()
def tot_cancel_run(run_id: str) -> Dict:
    """Cancel a run"""
    conn = engine.db.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE runs SET status = 'cancelled' WHERE run_id = ?",
        (run_id,)
    )
    conn.commit()
    conn.close()
    return {"success": True, "status": "cancelled"}

if __name__ == "__main__":
    print("Starting ToT Engine — Option B (Client Sampling)...", file=sys.stderr)
    mcp.run()
