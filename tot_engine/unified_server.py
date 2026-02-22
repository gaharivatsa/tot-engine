#!/usr/bin/env python3
"""
Tree of Thought Engine - Unified MCP Server

Production-grade MCP server providing both standard and enforced Tree of Thought reasoning.

Features:
- Standard mode: Flexible exploration with full control
- Enforced mode: Guaranteed depth with parameter validation
- Both modes accessible through unified API
- Production-ready with comprehensive error handling

Usage:
    # Standard mode
    mcporter call tot-engine.tot_start_run task_prompt="..."
    
    # Enforced mode  
    mcporter call tot-engine.tot_start_run_enforced 
        task_prompt="..."
        exploration_level="deep"
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

# Import enforcement components
from .enforcement import (
    EnforcementEngine, 
    EnforcementConfig,
    ExplorationLevel,
    ENFORCEMENT_CONFIGS,
    get_enforcement_config,
    calculate_score
)
from .config import (
    get_exploration_config,
    get_depth_guideline,
    get_candidate_strategies,
    DEPTH_CONFIGS
)

# ============================================================================
# Configuration
# ============================================================================

DB_PATH = os.environ.get("TOT_DB_PATH", "/tmp/tot_engine.db")
DEFAULT_BEAM_WIDTH = 3
DEFAULT_N_GENERATE = 2
DEFAULT_MAX_DEPTH = 5
DEFAULT_NODE_BUDGET = 20
DEFAULT_TARGET_SCORE = 0.85

# ============================================================================
# Pydantic Schemas
# ============================================================================

class State(BaseModel):
    """Evolving world state"""
    facts: List[str] = Field(default_factory=list, min_length=1)
    assumptions: List[str] = Field(default_factory=list)
    work: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)

class ThoughtCandidate(BaseModel):
    """Single candidate thought from client"""
    thought: str = Field(min_length=3, description="The reasoning step")
    delta: Dict[str, Any] = Field(default_factory=dict, description="What changed from parent")
    progress_estimate: float = Field(ge=0.0, le=1.0, description="Progress toward goal (0-1)")
    feasibility_estimate: float = Field(ge=0.0, le=1.0, description="Implementation feasibility (0-1)")
    risk_estimate: float = Field(ge=0.0, le=1.0, description="Risk level (0-1, higher=riskier)")
    reasoning: str = Field(min_length=3, description="Explanation for estimates")

class SearchConfig(BaseModel):
    """Search configuration"""
    beam_width: int = Field(default=3, ge=1, le=10)
    n_generate: int = Field(default=2, ge=1, le=5)
    max_depth: int = Field(default=5, ge=1, le=20)
    node_budget: int = Field(default=20, ge=2, le=1000)
    target_score: float = Field(default=0.85, ge=0.0, le=1.0)
    exploration_level: Optional[str] = Field(default=None, description="enforced mode level")

class StartRunRequest(BaseModel):
    """Request to start a ToT run"""
    task_prompt: str = Field(min_length=5, description="The problem to solve")
    constraints: List[str] = Field(default_factory=list)
    beam_width: int = Field(default=3, ge=1, le=10)
    n_generate: int = Field(default=2, ge=1, le=5)
    max_depth: int = Field(default=5, ge=1, le=20)
    node_budget: int = Field(default=20, ge=2, le=1000)
    target_score: float = Field(default=0.85, ge=0.0, le=1.0)

class StartEnforcedRunRequest(BaseModel):
    """Request to start an ENFORCED ToT run"""
    task_prompt: str = Field(min_length=5, description="The problem to solve")
    exploration_level: str = Field(
        default="moderate",
        pattern="^(shallow|moderate|deep|exhaustive)$",
        description="Enforcement level: shallow/moderate/deep/exhaustive"
    )
    constraints: List[str] = Field(default_factory=list)
    custom_budget: Optional[int] = Field(default=None, ge=10, le=2000)
    custom_target: Optional[float] = Field(default=None, ge=0.5, le=1.0)

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
        
        # Runs table
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
                is_enforced BOOLEAN DEFAULT 0,
                enforcement_level TEXT
            )
        """)
        
        # Nodes table
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
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                FOREIGN KEY (parent_id) REFERENCES nodes(node_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

# Initialize database
db = Database()

# ============================================================================
# MCP Server
# ============================================================================

mcp = FastMCP("tot-engine-unified")

@mcp.tool()
def tot_start_run(
    task_prompt: str,
    constraints: List[str] = Field(default_factory=list),
    beam_width: int = Field(default=3, ge=1, le=10),
    n_generate: int = Field(default=2, ge=1, le=5),
    max_depth: int = Field(default=5, ge=1, le=20),
    node_budget: int = Field(default=20, ge=2, le=1000),
    target_score: float = Field(default=0.85, ge=0.0, le=1.0)
) -> Dict:
    """
    STANDARD MODE: Start a flexible Tree of Thought run.
    
    Use this for exploratory research where you want full control over
    the exploration process. No minimum budget enforcement.
    
    Args:
        task_prompt: The problem or question to solve
        constraints: List of hard constraints (e.g., ["budget < 1000", " Pi 4 compatible"])
        beam_width: How many top nodes to keep per iteration (1-10)
        n_generate: How many children to generate per node (1-5)
        max_depth: Maximum tree depth (1-20)
        node_budget: Total nodes allowed including root (2-1000)
        target_score: Score threshold to mark node as terminal (0.0-1.0)
    
    Returns:
        Dictionary with run_id and configuration details
    
    Example:
        mcporter call tot-engine.tot_start_run 
            task_prompt="Which database for edge deployment?"
            constraints=["Pi 4 compatible", "Self-hosted"]
            beam_width=3
            n_generate=3
            node_budget=50
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    run_id = str(uuid.uuid4())
    config = SearchConfig(
        beam_width=beam_width,
        n_generate=n_generate,
        max_depth=max_depth,
        node_budget=node_budget,
        target_score=target_score
    )
    
    # Insert run
    cursor.execute(
        """INSERT INTO runs 
           (run_id, task_prompt, constraints_json, search_config_json, status, created_at, is_enforced)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run_id, task_prompt, json.dumps(constraints), config.json(), 
         'active', datetime.now().isoformat(), False)
    )
    
    # Create root node
    root_id = str(uuid.uuid4())
    cursor.execute(
        """INSERT INTO nodes 
           (node_id, run_id, parent_id, depth, thought, state_json, delta_json,
            evaluation_json, score, status, state_hash, thought_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (root_id, run_id, None, 0, f"ROOT: {task_prompt[:100]}",
         json.dumps({"constraints": constraints}),
         json.dumps({}), json.dumps({"progress": 0.5, "feasibility": 0.5, "confidence": 0.5}),
         0.5, 'active', 'root', hashlib.sha256(task_prompt.lower().encode()).hexdigest()[:16])
    )
    
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "run_id": run_id,
        "root_node_id": root_id,
        "mode": "standard",
        "config": config.dict(),
        "message": "Run started. Call tot_request_samples() to begin expansion."
    }

@mcp.tool()
def tot_start_run_enforced(
    task_prompt: str,
    exploration_level: str = Field(default="moderate", pattern="^(shallow|moderate|deep|exhaustive)$"),
    constraints: List[str] = Field(default_factory=list),
    custom_budget: Optional[int] = Field(default=None, ge=10, le=2000),
    custom_target: Optional[float] = Field(default=None, ge=0.5, le=1.0)
) -> Dict:
    """
    ENFORCED MODE: Start a Tree of Thought run with guaranteed exploration depth.
    
    Use this when you need rigorous, reproducible research with minimum
    budget consumption enforced. The system will not allow early termination
    until exploration requirements are met.
    
    Args:
        task_prompt: The problem or question to solve
        exploration_level: Enforcement preset - shallow/moderate/deep/exhaustive
        constraints: List of hard constraints
        custom_budget: Override default budget for this level (optional)
        custom_target: Override default target score (optional)
    
    Exploration Levels:
        shallow:    20 nodes,  80% min, 2 depths  (5 min, quick decisions)
        moderate:   50 nodes,  85% min, 3 depths  (30 min, tech selection)
        deep:       150 nodes, 90% min, 5 depths  (2 hrs, architecture)
        exhaustive: 500 nodes, 95% min, 8 depths  (8 hrs, publication-grade)
    
    Returns:
        Dictionary with run_id, enforcement config, and guidelines
    
    Example:
        mcporter call tot-engine.tot_start_run_enforced
            task_prompt="Design self-modelling architecture"
            exploration_level="deep"
            constraints=["Pi 4", "RLHF-free"]
    """
    # Get enforcement configuration
    enf_config = get_enforcement_config(exploration_level)
    
    # Apply custom overrides if provided
    if custom_budget:
        enf_config.node_budget = custom_budget
    if custom_target:
        enf_config.target_score = custom_target
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    run_id = str(uuid.uuid4())
    
    # Create search config from enforcement settings
    config = SearchConfig(
        beam_width=enf_config.beam_width,
        n_generate=enf_config.n_generate,
        max_depth=enf_config.max_depth,
        node_budget=enf_config.node_budget,
        target_score=enf_config.target_score,
        exploration_level=exploration_level
    )
    
    # Insert run with enforcement metadata
    cursor.execute(
        """INSERT INTO runs 
           (run_id, task_prompt, constraints_json, search_config_json, status, 
            created_at, is_enforced, enforcement_level)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, task_prompt, json.dumps(constraints), config.json(),
         'active', datetime.now().isoformat(), True, exploration_level)
    )
    
    # Create root node
    root_id = str(uuid.uuid4())
    cursor.execute(
        """INSERT INTO nodes 
           (node_id, run_id, parent_id, depth, thought, state_json, delta_json,
            evaluation_json, score, status, state_hash, thought_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (root_id, run_id, None, 0, f"ROOT: {task_prompt[:100]}",
         json.dumps({"constraints": constraints, "enforcement": exploration_level}),
         json.dumps({}), json.dumps({"progress": 0.5, "feasibility": 0.5, "confidence": 0.5}),
         0.5, 'active', 'root', hashlib.sha256(task_prompt.lower().encode()).hexdigest()[:16])
    )
    
    conn.commit()
    conn.close()
    
    # Get scoring guidelines
    guideline = get_depth_guideline(0)
    
    return {
        "success": True,
        "run_id": run_id,
        "root_node_id": root_id,
        "mode": "enforced",
        "exploration_level": exploration_level,
        "config": {
            "node_budget": enf_config.node_budget,
            "min_required_nodes": enf_config.min_required_nodes,
            "min_consumption_ratio": enf_config.min_consumption_ratio,
            "beam_width": enf_config.beam_width,
            "n_generate": enf_config.n_generate,
            "max_depth": enf_config.max_depth,
            "target_score": enf_config.target_score,
        },
        "scoring_guidelines": {
            "depth_0": {
                "range": guideline.score_range,
                "strategy": guideline.strategy,
                "example_scores": guideline.example_scores,
            }
        },
        "message": f"ENFORCED run started. MUST use {enf_config.min_required_nodes}+ nodes before synthesis."
    }

@mcp.tool()
def tot_request_samples(run_id: str) -> Dict:
    """
    Request frontier nodes for expansion.
    
    Returns the current frontier (best-scoring active nodes) that should be
    expanded. Use this with both standard and enforced modes.
    
    Args:
        run_id: The run ID from tot_start_run or tot_start_run_enforced
    
    Returns:
        Dictionary with frontier nodes and sample requests
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get config
    cursor.execute("SELECT search_config_json, is_enforced, enforcement_level FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    config = SearchConfig(**json.loads(run['search_config_json']))
    is_enforced = run['is_enforced']
    enf_level = run['enforcement_level']
    
    # Select frontier (top beam_width active nodes)
    cursor.execute(
        """SELECT node_id, depth, thought, score, state_json 
           FROM nodes 
           WHERE run_id = ? AND status = 'active'
           ORDER BY score DESC, depth ASC
           LIMIT ?""",
        (run_id, config.beam_width)
    )
    frontier = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if not frontier:
        return {"success": False, "error": "No frontier nodes to expand"}
    
    # Build sample requests
    sample_requests = []
    for node in frontier:
        sample_requests.append({
            "node_id": node["node_id"],
            "depth": node["depth"],
            "thought": node["thought"],
            "score": node["score"],
            "num_candidates": config.n_generate,
        })
    
    response = {
        "success": True,
        "run_id": run_id,
        "mode": "enforced" if is_enforced else "standard",
        "frontier_size": len(frontier),
        "sample_requests": sample_requests,
    }
    
    # Add enforcement info if applicable
    if is_enforced:
        enf_config = get_enforcement_config(enf_level)
        guideline = get_depth_guideline(frontier[0]["depth"] if frontier else 0)
        
        response["enforcement"] = {
            "level": enf_level,
            "min_required_nodes": enf_config.min_required_nodes,
            "scoring_range": guideline.score_range,
            "scoring_strategy": guideline.strategy,
        }
    
    return response

@mcp.tool()
def tot_submit_samples(
    run_id: str,
    samples: List[Dict] = Field(..., description="List of {parent_node_id, candidates[]}")
) -> Dict:
    """
    Submit generated thoughts for evaluation.
    
    Submit candidates for each frontier node. The engine will evaluate,
    score, and insert valid candidates into the tree.
    
    Args:
        run_id: The run ID
        samples: List of samples, each with parent_node_id and candidates list
    
    Returns:
        Dictionary with expansion results and statistics
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get config
    cursor.execute("SELECT search_config_json FROM runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    config = SearchConfig(**json.loads(run['search_config_json']))
    
    nodes_expanded = 0
    nodes_pruned = 0
    
    for sample in samples:
        parent_id = sample["parent_node_id"]
        candidates = sample["candidates"]
        
        # Get parent info
        cursor.execute(
            "SELECT depth, state_json FROM nodes WHERE node_id = ?",
            (parent_id,)
        )
        parent = cursor.fetchone()
        if not parent:
            continue
        
        parent_depth = parent["depth"]
        
        # Check depth limit
        if parent_depth >= config.max_depth:
            continue
        
        for cand_data in candidates:
            try:
                # Validate candidate
                cand = ThoughtCandidate(**cand_data)
                
                # Check budget
                cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
                if cursor.fetchone()[0] >= config.node_budget:
                    conn.commit()
                    conn.close()
                    return {
                        "success": True,
                        "status": "stopped",
                        "stop_reason": "budget_exhausted",
                        "nodes_expanded": nodes_expanded,
                        "nodes_pruned": nodes_pruned,
                    }
                
                # Calculate score
                score = calculate_score(
                    cand.progress_estimate,
                    cand.feasibility_estimate,
                    0.7,  # default confidence
                    cand.risk_estimate,
                    config.exploration_level or "moderate"
                )
                
                # Determine status
                if score >= config.target_score:
                    status = "terminal"
                else:
                    status = "active"
                
                # Compute hashes
                delta_json = json.dumps(cand.delta)
                state_hash = hashlib.sha256(delta_json.encode()).hexdigest()[:16]
                thought_hash = hashlib.sha256(cand.thought.lower().strip().encode()).hexdigest()[:16]
                
                # Insert node
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
                     json.dumps({
                         "progress": cand.progress_estimate,
                         "feasibility": cand.feasibility_estimate,
                         "risk": cand.risk_estimate,
                         "reasoning": cand.reasoning
                     }),
                     score, status, state_hash, thought_hash)
                )
                nodes_expanded += 1
                
            except Exception as e:
                nodes_pruned += 1
                continue
    
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "nodes_expanded": nodes_expanded,
        "nodes_pruned": nodes_pruned,
        "status": "running",
    }

@mcp.tool()
def tot_get_best_path(run_id: str, enforce_completion: bool = Field(default=False)) -> Dict:
    """
    Get the best path and final synthesis.
    
    Args:
        run_id: The run ID
        enforce_completion: If True, rejects request if enforced run requirements not met
    
    Returns:
        Dictionary with final answer and reasoning chain
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get run info
    cursor.execute(
        "SELECT is_enforced, enforcement_level, search_config_json FROM runs WHERE run_id = ?",
        (run_id,)
    )
    run = cursor.fetchone()
    
    is_enforced = run["is_enforced"] if run else False
    enf_level = run["enforcement_level"] if run else None
    config = SearchConfig(**json.loads(run["search_config_json"])) if run else None
    
    # Check enforcement requirements
    if enforce_completion and is_enforced:
        enf_config = get_enforcement_config(enf_level)
        
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
        node_count = cursor.fetchone()[0]
        min_required = enf_config.min_required_nodes
        
        if node_count < min_required:
            conn.close()
            return {
                "success": False,
                "error": f"Enforcement requirements not met",
                "nodes_used": node_count,
                "nodes_required": min_required,
                "message": f"Must use {min_required}+ nodes before synthesis. Continue expanding.",
            }
    
    # Find best terminal or highest-scoring node
    cursor.execute(
        """SELECT node_id, thought, score, depth 
           FROM nodes 
           WHERE run_id = ? 
           ORDER BY score DESC, depth DESC
           LIMIT 1""",
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
        "is_terminal": best["score"] >= (config.target_score if config else 0.85),
        "reasoning_chain": path,
        "mode": "enforced" if is_enforced else "standard",
    }

@mcp.tool()
def tot_get_enforcement_status(run_id: str) -> Dict:
    """
    Get enforcement status for a run.
    
    Shows current progress against enforcement requirements.
    Only useful for enforced runs (returns minimal info for standard runs).
    
    Args:
        run_id: The run ID
    
    Returns:
        Dictionary with enforcement metrics and recommendations
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT is_enforced, enforcement_level, search_config_json FROM runs WHERE run_id = ?",
        (run_id,)
    )
    run = cursor.fetchone()
    
    if not run:
        conn.close()
        return {"success": False, "error": "Run not found"}
    
    is_enforced = run["is_enforced"]
    
    if not is_enforced:
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
        node_count = cursor.fetchone()[0]
        conn.close()
        return {
            "success": True,
            "mode": "standard",
            "nodes_used": node_count,
            "message": "Standard mode - no enforcement requirements",
        }
    
    # Get enforcement stats
    enf_level = run["enforcement_level"]
    enf_config = get_enforcement_config(enf_level)
    config = SearchConfig(**json.loads(run["search_config_json"]))
    
    cursor.execute("SELECT COUNT(*) FROM nodes WHERE run_id = ?", (run_id,))
    node_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(depth) FROM nodes WHERE run_id = ?", (run_id,))
    max_depth = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(score) FROM nodes WHERE run_id = ?", (run_id,))
    avg_score = cursor.fetchone()[0] or 0
    
    conn.close()
    
    min_required = enf_config.min_required_nodes
    requirement_met = node_count >= min_required
    
    return {
        "success": True,
        "mode": "enforced",
        "exploration_level": enf_level,
        "progress": {
            "nodes_used": node_count,
            "nodes_required": min_required,
            "nodes_remaining": max(0, min_required - node_count),
            "budget_utilization": node_count / enf_config.node_budget,
            "requirement_met": requirement_met,
            "max_depth": max_depth,
            "max_depth_required": 3 if enf_level in ["deep", "exhaustive"] else 2,
            "avg_score": round(avg_score, 4),
        },
        "config": {
            "node_budget": enf_config.node_budget,
            "target_score": enf_config.target_score,
            "beam_width": enf_config.beam_width,
        },
        "recommendations": [] if requirement_met else [
            f"Continue expansion: need {min_required - node_count} more nodes",
            f"Current depth {max_depth}, target depth {enf_config.max_depth}",
        ],
    }

@mcp.tool()
def tot_list_runs(limit: int = Field(default=10, ge=1, le=100)) -> Dict:
    """List recent ToT runs with their modes"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT run_id, task_prompt, status, created_at, is_enforced, enforcement_level
           FROM runs 
           ORDER BY created_at DESC 
           LIMIT ?""",
        (limit,)
    )
    runs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"success": True, "runs": runs}

@mcp.tool()
def tot_get_exploration_guide(level: str = Field(default="moderate", pattern="^(shallow|moderate|deep|exhaustive)$")) -> Dict:
    """
    Get detailed exploration guide for a specific level.
    
    Returns scoring guidelines, when to use, and best practices.
    
    Args:
        level: Exploration level (shallow/moderate/deep/exhaustive)
    """
    config = get_exploration_config(level)
    
    guidelines = []
    for depth in range(0, min(config["max_depth"] + 1, 5)):
        guideline = get_depth_guideline(depth)
        guidelines.append({
            "depth": depth,
            "description": guideline.description,
            "score_range": guideline.score_range,
            "strategy": guideline.strategy,
            "example_scores": guideline.example_scores,
        })
    
    return {
        "success": True,
        "exploration_level": level,
        "config": config,
        "depth_guidelines": guidelines,
        "candidate_strategies": get_candidate_strategies(level),
        "best_practices": {
            "do": config.get("use_when", []),
            "avoid": config.get("avoid_when", []),
        },
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    mcp.run(transport='stdio')
