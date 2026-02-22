#!/usr/bin/env python3
"""
Tree of Thought Engine v2.0 - Unified MCP Server

Production-grade MCP server with unified API for both regular and enforced modes.

Usage:
    # Regular mode (flexible)
    mcporter call tot-engine.tot_start_run 
        mode="regular"
        task_prompt="Which database?"
        beam_width=3
        node_budget=50
    
    # Enforced mode (guaranteed depth)
    mcporter call tot-engine.tot_start_run
        mode="enforced"
        task_prompt="Design architecture"
        exploration_level="deep"
        
    # Enforced with overrides
    mcporter call tot-engine.tot_start_run
        mode="enforced"
        task_prompt="..."
        exploration_level="moderate"
        custom_beam_width=5  # Override preset
"""

import os
import sys
import json
import sqlite3
import uuid
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Import enforcement and config modules
from enforcement import (
    EnforcementConfig,
    ENFORCEMENT_CONFIGS,
    get_enforcement_config,
    calculate_score,
)
from config import get_depth_guideline, DEPTH_CONFIGS

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = os.environ.get("TOT_DB_PATH", "/tmp/tot_engine.db")

# Default parameters for regular mode
DEFAULTS = {
    "beam_width": 3,
    "n_generate": 2,
    "max_depth": 5,
    "node_budget": 20,
    "target_score": 0.85,
}

# ============================================================================
# Database Setup
# ============================================================================

class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task_prompt TEXT NOT NULL,
                constraints_json TEXT,
                search_config_json TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                best_node_id TEXT,
                mode TEXT DEFAULT 'regular',
                exploration_level TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                parent_id TEXT,
                depth INTEGER NOT NULL,
                thought TEXT NOT NULL,
                state_json TEXT,
                delta_json TEXT,
                evaluation_json TEXT,
                score REAL NOT NULL,
                status TEXT DEFAULT 'active',
                state_hash TEXT,
                thought_hash TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

db = Database()

# ============================================================================
# MCP Server
# ============================================================================

mcp = FastMCP("tot-engine-v2")

class ThoughtCandidate(BaseModel):
    """Single candidate thought"""
    thought: str = Field(min_length=3)
    delta: Dict[str, Any] = Field(default_factory=dict)
    progress_estimate: float = Field(ge=0.0, le=1.0)
    feasibility_estimate: float = Field(ge=0.0, le=1.0)
    risk_estimate: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=3)


@mcp.tool()
def tot_start_run(
    task_prompt: str = Field(..., min_length=5, description="The problem to solve"),
    mode: str = Field(default="regular", pattern="^(regular|enforced)$", description="Mode: regular (flexible) or enforced (guaranteed depth)"),
    constraints: List[str] = Field(default_factory=list, description="Hard constraints"),
    # Regular mode params (used when mode="regular" or as overrides)
    beam_width: Optional[int] = Field(default=None, ge=1, le=10, description="Nodes to keep per iteration"),
    n_generate: Optional[int] = Field(default=None, ge=1, le=5, description="Children per node"),
    max_depth: Optional[int] = Field(default=None, ge=1, le=20, description="Max tree depth"),
    node_budget: Optional[int] = Field(default=None, ge=2, le=1000, description="Total node budget"),
    target_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Terminal threshold"),
    # Enforced mode params (used when mode="enforced")
    exploration_level: Optional[str] = Field(default=None, pattern="^(shallow|moderate|deep|exhaustive)$", description="Enforcement preset"),
    # Override flags for enforced mode
    custom_beam_width: Optional[int] = Field(default=None, ge=1, le=10, description="Override beam_width in enforced mode"),
    custom_target_score: Optional[float] = Field(default=None, ge=0.5, le=1.0, description="Override target_score in enforced mode"),
) -> Dict:
    """
    UNIFIED: Start a Tree of Thought run (regular or enforced mode).
    
    Use 'mode' parameter to select behavior:
    - "regular": Full control over all parameters
    - "enforced": Use exploration_level preset, with optional overrides
    
    Examples:
        # Regular mode
        tot_start_run(mode="regular", task_prompt="...", beam_width=3)
        
        # Enforced mode (preset)
        tot_start_run(mode="enforced", task_prompt="...", exploration_level="deep")
        
        # Enforced with override
        tot_start_run(mode="enforced", exploration_level="moderate", custom_beam_width=5)
    """
    
    # Resolve parameters based on mode
    if mode == "enforced":
        if not exploration_level:
            return {
                "success": False,
                "error": "exploration_level required when mode='enforced'",
                "hint": "Use: exploration_level='shallow'|'moderate'|'deep'|'exhaustive'"
            }
        
        # Get enforcement config
        enf_config = get_enforcement_config(exploration_level)
        
        # Apply preset values
        resolved_beam = custom_beam_width or enf_config.beam_width
        resolved_n_gen = enf_config.n_generate
        resolved_max_depth = enf_config.max_depth
        resolved_budget = enf_config.node_budget
        resolved_target = custom_target_score or enf_config.target_score
        
        mode_label = "enforced"
        level_label = exploration_level
        
    else:  # regular mode
        # Use provided values or defaults
        resolved_beam = beam_width or DEFAULTS["beam_width"]
        resolved_n_gen = n_generate or DEFAULTS["n_generate"]
        resolved_max_depth = max_depth or DEFAULTS["max_depth"]
        resolved_budget = node_budget or DEFAULTS["node_budget"]
        resolved_target = target_score or DEFAULTS["target_score"]
        
        mode_label = "regular"
        level_label = None
    
    # Create run in database
    conn = db.get_connection()
    cursor = conn.cursor()
    
    run_id = str(uuid.uuid4())
    config = {
        "beam_width": resolved_beam,
        "n_generate": resolved_n_gen,
        "max_depth": resolved_max_depth,
        "node_budget": resolved_budget,
        "target_score": resolved_target,
        "mode": mode,
        "exploration_level": level_label,
    }
    
    cursor.execute(
        """INSERT INTO runs 
           (run_id, task_prompt, constraints_json, search_config_json, 
            status, created_at, mode, exploration_level)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, task_prompt, json.dumps(constraints), json.dumps(config),
         'active', datetime.now().isoformat(), mode_label, level_label)
    )
    
    # Create root node
    root_id = str(uuid.uuid4())
    cursor.execute(
        """INSERT INTO nodes 
           (node_id, run_id, parent_id, depth, thought, state_json, delta_json,
            evaluation_json, score, status, state_hash, thought_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (root_id, run_id, None, 0, f"ROOT: {task_prompt[:100]}",
         json.dumps({"constraints": constraints, "mode": mode, "level": level_label}),
         json.dumps({}), json.dumps({"progress": 0.5, "feasibility": 0.5}),
         0.5, 'active', 'root', 
         hashlib.sha256(task_prompt.lower().encode()).hexdigest()[:16])
    )
    
    conn.commit()
    conn.close()
    
    # Build response
    response = {
        "success": True,
        "run_id": run_id,
        "root_node_id": root_id,
        "mode": mode_label,
        "config": config,
    }
    
    # Add enforcement-specific info
    if mode == "enforced":
        enf_config = get_enforcement_config(exploration_level)
        guideline = get_depth_guideline(0)
        
        response["enforcement"] = {
            "exploration_level": exploration_level,
            "min_required_nodes": enf_config.min_required_nodes,
            "min_consumption_ratio": enf_config.min_consumption_ratio,
            "message": f"MUST use {enf_config.min_required_nodes}+ nodes before synthesis",
        }
        response["scoring_guidelines"] = {
            "depth_0": {
                "range": guideline.score_range,
                "strategy": guideline.strategy,
            }
        }
    
    return response


@mcp.tool()
def tot_request_samples(run_id: str) -> Dict:
    """Request frontier nodes for expansion"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT search_config_json FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    config = json.loads(run["search_config_json"])
    
    cursor.execute(
        """SELECT node_id, depth, thought, score 
           FROM nodes 
           WHERE run_id = ? AND status = 'active'
           ORDER BY score DESC, depth ASC
           LIMIT ?""",
        (run_id, config["beam_width"])
    )
    frontier = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if not frontier:
        return {"success": False, "error": "No frontier nodes"}
    
    response = {
        "success": True,
        "run_id": run_id,
        "mode": config.get("mode", "regular"),
        "frontier_size": len(frontier),
        "sample_requests": [
            {
                "node_id": n["node_id"],
                "depth": n["depth"],
                "thought": n["thought"],
                "score": n["score"],
                "num_candidates": config["n_generate"],
            }
            for n in frontier
        ],
    }
    
    # Add enforcement info
    if config.get("mode") == "enforced":
        level = config.get("exploration_level")
        if level:
            from enforcement import get_enforcement_config
            enf_config = get_enforcement_config(level)
            guideline = get_depth_guideline(frontier[0]["depth"])
            
            response["enforcement"] = {
                "level": level,
                "min_required": enf_config.min_required_nodes,
                "scoring_range": guideline.score_range,
                "strategy": guideline.strategy,
            }
    
    return response


@mcp.tool()
def tot_submit_samples(
    run_id: str,
    samples: List[Dict] = Field(..., description="List of {parent_node_id, candidates[]}")
) -> Dict:
    """Submit generated thoughts for evaluation"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT search_config_json FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    config = json.loads(run["search_config_json"])
    mode = config.get("mode", "regular")
    exploration_level = config.get("exploration_level")
    
    nodes_expanded = 0
    nodes_pruned = 0
    
    for sample in samples:
        parent_id = sample.get("parent_node_id")
        candidates = sample.get("candidates", [])
        
        cursor.execute("SELECT depth FROM nodes WHERE node_id = ?", (parent_id,))
        parent = cursor.fetchone()
        if not parent:
            continue
        
        parent_depth = parent["depth"]
        if parent_depth >= config["max_depth"]:
            continue
        
        for cand_data in candidates:
            try:
                # Check budget
                cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
                if cursor.fetchone()[0] >= config["node_budget"]:
                    conn.commit()
                    conn.close()
                    return {
                        "success": True,
                        "status": "stopped",
                        "stop_reason": "budget_exhausted",
                        "nodes_expanded": nodes_expanded,
                    }
                
                # Calculate score with mode-appropriate weights
                cand = ThoughtCandidate(**cand_data)
                score = calculate_score(
                    cand.progress_estimate,
                    cand.feasibility_estimate,
                    0.7,
                    cand.risk_estimate,
                    exploration_level or "moderate"
                )
                
                status = "terminal" if score >= config["target_score"] else "active"
                
                node_id = str(uuid.uuid4())
                delta_json = json.dumps(cand.delta)
                cursor.execute(
                    """INSERT INTO nodes 
                       (node_id, run_id, parent_id, depth, thought, delta_json,
                        evaluation_json, score, status, state_hash, thought_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (node_id, run_id, parent_id, parent_depth + 1,
                     cand.thought, delta_json,
                     json.dumps({"progress": cand.progress_estimate, 
                                "feasibility": cand.feasibility_estimate,
                                "risk": cand.risk_estimate}),
                     score, status,
                     hashlib.sha256(delta_json.encode()).hexdigest()[:16],
                     hashlib.sha256(cand.thought.lower().encode()).hexdigest()[:16])
                )
                nodes_expanded += 1
                
            except Exception:
                nodes_pruned += 1
    
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "nodes_expanded": nodes_expanded,
        "nodes_pruned": nodes_pruned,
        "status": "running",
    }


@mcp.tool()
def tot_get_best_path(
    run_id: str,
    enforce_completion: bool = Field(default=False, description="For enforced mode: require min nodes")
) -> Dict:
    """Get best path and final synthesis"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT mode, exploration_level, search_config_json FROM runs WHERE run_id = ?",
        (run_id,)
    )
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    mode = run["mode"]
    level = run["exploration_level"]
    config = json.loads(run["search_config_json"])
    
    # Check enforcement requirements
    if enforce_completion and mode == "enforced" and level:
        from enforcement import get_enforcement_config
        enf_config = get_enforcement_config(level)
        
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
        node_count = cursor.fetchone()[0]
        min_required = enf_config.min_required_nodes
        
        if node_count < min_required:
            conn.close()
            return {
                "success": False,
                "error": "Enforcement requirements not met",
                "nodes_used": node_count,
                "nodes_required": min_required,
                "message": f"Must use {min_required}+ nodes. Continue expanding.",
            }
    
    # Find best node
    cursor.execute(
        """SELECT node_id, thought, score, depth 
           FROM nodes WHERE run_id = ? ORDER BY score DESC, depth DESC LIMIT 1""",
        (run_id,)
    )
    best = cursor.fetchone()
    
    if not best:
        conn.close()
        return {"success": False, "error": "No nodes found"}
    
    # Build path
    path = []
    current_id = best["node_id"]
    while current_id:
        cursor.execute(
            "SELECT node_id, parent_id, thought, score, depth FROM nodes WHERE node_id = ?",
            (current_id,)
        )
        node = cursor.fetchone()
        if not node:
            break
        path.append({
            "node_id": node["node_id"],
            "depth": node["depth"],
            "thought": node["thought"],
            "score": node["score"],
        })
        current_id = node["parent_id"]
    path.reverse()
    
    # Update run
    cursor.execute(
        "UPDATE runs SET status = ?, completed_at = ?, best_node_id = ? WHERE run_id = ?",
        ('completed', datetime.now().isoformat(), best["node_id"], run_id)
    )
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "final_answer": best["thought"],
        "confidence": best["score"],
        "path_length": len(path),
        "mode": mode,
        "reasoning_chain": path,
    }


@mcp.tool()
def tot_get_enforcement_status(run_id: str) -> Dict:
    """Get enforcement status (useful for enforced runs)"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT mode, exploration_level, search_config_json FROM runs WHERE run_id = ?",
        (run_id,)
    )
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    mode = run["mode"]
    
    cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
    node_count = cursor.fetchone()[0]
    
    if mode != "enforced":
        conn.close()
        return {
            "success": True,
            "mode": "regular",
            "nodes_used": node_count,
            "message": "Regular mode - no enforcement",
        }
    
    level = run["exploration_level"]
    from enforcement import get_enforcement_config
    enf_config = get_enforcement_config(level)
    
    cursor.execute("SELECT MAX(depth) FROM nodes WHERE run_id = ?", (run_id,))
    max_depth = cursor.fetchone()[0] or 0
    
    min_required = enf_config.min_required_nodes
    requirement_met = node_count >= min_required
    
    conn.close()
    
    return {
        "success": True,
        "mode": "enforced",
        "level": level,
        "progress": {
            "nodes_used": node_count,
            "nodes_required": min_required,
            "requirement_met": requirement_met,
            "max_depth": max_depth,
        },
        "message": "Requirements met" if requirement_met else f"Need {min_required - node_count} more nodes",
    }


@mcp.tool()
def tot_get_exploration_guide(level: str = Field(default="moderate", pattern="^(shallow|moderate|deep|exhaustive)$")) -> Dict:
    """Get exploration guide for a level"""
    config = DEPTH_CONFIGS[level]
    guideline = get_depth_guideline(0)
    
    return {
        "success": True,
        "level": level,
        "config": config,
        "scoring": {
            "range": guideline.score_range,
            "strategy": guideline.strategy,
            "example_scores": guideline.example_scores,
        },
        "use_when": config.get("use_when", []),
        "avoid_when": config.get("avoid_when", []),
    }


@mcp.tool()
def tot_list_runs(limit: int = Field(default=10, ge=1, le=100)) -> Dict:
    """List recent runs"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT run_id, task_prompt, status, created_at, mode, exploration_level
           FROM runs ORDER BY created_at DESC LIMIT ?""",
        (limit,)
    )
    runs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"success": True, "runs": runs}


if __name__ == "__main__":
    mcp.run(transport='stdio')
