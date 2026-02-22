#!/usr/bin/env python3
"""
Enforcement Engine for Tree of Thought

Provides guaranteed exploration depth through parameter enforcement.
Part of the unified tot-engine MCP server.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time

class ExplorationLevel(Enum):
    """Standard exploration levels with enforced parameters"""
    SHALLOW = "shallow"
    MODERATE = "moderate"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"

@dataclass
class EnforcementConfig:
    """Configuration for enforced exploration"""
    name: str
    description: str
    node_budget: int
    min_consumption_ratio: float  # Must use this % of budget
    beam_width: int
    n_generate: int
    max_depth: int
    target_score: float
    max_iterations: int
    validation_tests: int  # Number of empirical tests
    sensitivity_runs: int  # Number of sensitivity analysis runs
    literature_required: bool
    
    @property
    def min_required_nodes(self) -> int:
        """Calculate minimum nodes that must be used"""
        return int(self.node_budget * self.min_consumption_ratio)

# Production-grade enforcement configurations
ENFORCEMENT_CONFIGS: Dict[ExplorationLevel, EnforcementConfig] = {
    ExplorationLevel.SHALLOW: EnforcementConfig(
        name="shallow",
        description="Quick directional guidance for simple decisions",
        node_budget=20,
        min_consumption_ratio=0.80,
        beam_width=2,
        n_generate=2,
        max_depth=2,
        target_score=0.70,
        max_iterations=10,
        validation_tests=0,
        sensitivity_runs=0,
        literature_required=False,
    ),
    ExplorationLevel.MODERATE: EnforcementConfig(
        name="moderate",
        description="Balanced exploration for technology selection and trade-offs",
        node_budget=50,
        min_consumption_ratio=0.85,
        beam_width=3,
        n_generate=3,
        max_depth=3,
        target_score=0.75,
        max_iterations=20,
        validation_tests=1,
        sensitivity_runs=0,
        literature_required=True,
    ),
    ExplorationLevel.DEEP: EnforcementConfig(
        name="deep",
        description="Thorough exploration with validation for architectural decisions",
        node_budget=150,
        min_consumption_ratio=0.90,
        beam_width=5,
        n_generate=4,
        max_depth=5,
        target_score=0.80,
        max_iterations=50,
        validation_tests=3,
        sensitivity_runs=3,
        literature_required=True,
    ),
    ExplorationLevel.EXHAUSTIVE: EnforcementConfig(
        name="exhaustive",
        description="Maximum rigor for publication-grade research",
        node_budget=500,
        min_consumption_ratio=0.95,
        beam_width=7,
        n_generate=5,
        max_depth=8,
        target_score=0.85,
        max_iterations=100,
        validation_tests=10,
        sensitivity_runs=10,
        literature_required=True,
    ),
}

class EnforcementEngine:
    """
    Engine that enforces minimum exploration requirements.
    
    This class wraps the core ToT functionality and adds:
    - Minimum budget consumption enforcement
    - Depth-based scoring guidelines
    - Empirical validation
    - Sensitivity analysis
    - Literature integration
    """
    
    def __init__(self, config: EnforcementConfig):
        self.config = config
        self.stats = {
            "nodes_used": 0,
            "iterations": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "sensitivity_consistency": None,
        }
    
    def should_continue(self, nodes_used: int, current_depth: int) -> Tuple[bool, str]:
        """
        Determine if exploration should continue based on enforcement rules.
        
        Returns:
            (should_continue, reason)
        """
        min_required = self.config.min_required_nodes
        
        # Rule 1: Must meet minimum budget
        if nodes_used < min_required:
            return True, f"Budget under-utilized: {nodes_used}/{min_required} nodes"
        
        # Rule 2: For deep/exhaustive, ensure adequate depth
        if self.config.name in ["deep", "exhaustive"] and current_depth < 3:
            return True, f"Depth requirement not met: {current_depth}/3"
        
        # Rule 3: Exhaustive requires more iterations
        if self.config.name == "exhaustive" and self.stats["iterations"] < 10:
            return True, f"Iteration requirement: {self.stats['iterations']}/10"
        
        return False, "All enforcement requirements satisfied"
    
    def get_scoring_guidelines(self, depth: int) -> Dict[str, Any]:
        """
        Get scoring guidelines for current depth and exploration level.
        
        Returns dictionary with recommended score ranges and strategies.
        """
        guidelines = {
            "shallow": {
                0: {"range": (0.45, 0.65), "strategy": "Keep all below 0.70"},
                1: {"range": (0.50, 0.68), "strategy": "Best can reach 0.68"},
                2: {"range": (0.55, 0.70), "strategy": "Terminal nodes only"},
            },
            "moderate": {
                0: {"range": (0.50, 0.70), "strategy": "Distribute 0.50-0.65, best 0.68-0.70"},
                1: {"range": (0.55, 0.72), "strategy": "Top candidates 0.70-0.72"},
                2: {"range": (0.50, 0.75), "strategy": "Exceptional can reach 0.75"},
                3: {"range": (0.45, 0.70), "strategy": "Implementation details"},
            },
            "deep": {
                0: {"range": (0.50, 0.70), "strategy": "All below 0.70 to force expansion"},
                1: {"range": (0.55, 0.68), "strategy": "Conservative scoring"},
                2: {"range": (0.50, 0.70), "strategy": "Only excellent reach 0.70"},
                3: {"range": (0.45, 0.65), "strategy": "Most 0.45-0.60"},
                4: {"range": (0.40, 0.60), "strategy": "Deep details 0.40-0.55"},
            },
            "exhaustive": {
                0: {"range": (0.45, 0.65), "strategy": "Very conservative, force deep exploration"},
                1: {"range": (0.50, 0.65), "strategy": "Keep most below 0.65"},
                2: {"range": (0.45, 0.65), "strategy": "Exceptional reach 0.65"},
                3: {"range": (0.40, 0.60), "strategy": "Deep implementation"},
                4: {"range": (0.35, 0.55), "strategy": "Fine details"},
            },
        }
        
        level_guidelines = guidelines.get(self.config.name, guidelines["moderate"])
        
        # Return guideline for current depth, or deepest available
        if depth in level_guidelines:
            return level_guidelines[depth]
        else:
            # For depths beyond defined, use deepest guideline
            max_depth = max(level_guidelines.keys())
            return level_guidelines[max_depth]
    
    def recommend_score(self, depth: int, candidate_index: int, is_best: bool = False) -> float:
        """
        Recommend a score for a candidate based on exploration level and context.
        
        Args:
            depth: Current tree depth
            candidate_index: Which candidate (0-indexed)
            is_best: Whether this is the best candidate
        
        Returns:
            Recommended score (0.0-1.0)
        """
        guidelines = self.get_scoring_guidelines(depth)
        min_score, max_score = guidelines["range"]
        
        if self.config.name == "shallow":
            # Even distribution
            if is_best:
                return round(min(max_score, 0.68), 2)
            return round(min_score + (candidate_index * 0.05), 2)
        
        elif self.config.name == "moderate":
            # Best can be higher
            if is_best:
                return round(min(max_score, 0.73), 2)
            elif candidate_index == 1:
                return round(min_score + 0.10, 2)
            else:
                return round(min_score + (candidate_index * 0.04), 2)
        
        elif self.config.name in ["deep", "exhaustive"]:
            # Very conservative
            if is_best and depth >= 3:
                return round(min(max_score, 0.70), 2)
            elif is_best:
                return round(min(max_score - 0.05, 0.65), 2)
            else:
                return round(min_score + (candidate_index * 0.03), 2)
        
        return round((min_score + max_score) / 2, 2)
    
    def validate_result(self, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that enforcement requirements were met.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check budget consumption
        nodes_used = result.get("nodes_explored", 0)
        budget = result.get("node_budget", self.config.node_budget)
        min_required = int(budget * self.config.min_consumption_ratio)
        
        if nodes_used < min_required:
            issues.append(
                f"Budget under-utilized: {nodes_used}/{min_required} nodes "
                f"({nodes_used/min_required*100:.1f}% of required)"
            )
        
        # Check depth
        path_length = result.get("path_length", 0)
        if self.config.name in ["deep", "exhaustive"] and path_length < 3:
            issues.append(f"Insufficient depth: {path_length}/3")
        
        # Check confidence
        confidence = result.get("confidence", 0)
        if confidence < self.config.target_score * 0.9:
            issues.append(
                f"Low confidence: {confidence:.2f} (target: {self.config.target_score})"
            )
        
        return len(issues) == 0, issues
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate enforcement report"""
        return {
            "config": {
                "name": self.config.name,
                "node_budget": self.config.node_budget,
                "min_required": self.config.min_required_nodes,
                "target_score": self.config.target_score,
            },
            "stats": self.stats,
            "requirements": {
                "min_budget_ratio": self.config.min_consumption_ratio,
                "validation_tests": self.config.validation_tests,
                "sensitivity_runs": self.config.sensitivity_runs,
                "literature_required": self.config.literature_required,
            },
        }


def get_enforcement_config(level: str) -> EnforcementConfig:
    """Get enforcement configuration by name"""
    try:
        level_enum = ExplorationLevel(level.lower())
        return ENFORCEMENT_CONFIGS[level_enum]
    except ValueError:
        # Default to moderate if invalid
        return ENFORCEMENT_CONFIGS[ExplorationLevel.MODERATE]


# Scoring formula weights (customizable per level)
SCORING_WEIGHTS = {
    "shallow": {"progress": 0.50, "feasibility": 0.30, "confidence": 0.20, "risk_penalty": 0.20},
    "moderate": {"progress": 0.45, "feasibility": 0.35, "confidence": 0.20, "risk_penalty": 0.25},
    "deep": {"progress": 0.40, "feasibility": 0.30, "confidence": 0.15, "risk_penalty": 0.35},
    "exhaustive": {"progress": 0.35, "feasibility": 0.25, "confidence": 0.15, "risk_penalty": 0.40},
}


def calculate_score(
    progress: float,
    feasibility: float,
    confidence: float,
    risk: float,
    level: str = "moderate"
) -> float:
    """
    Calculate composite score using level-appropriate weights.
    
    Higher levels penalize risk more heavily.
    """
    weights = SCORING_WEIGHTS.get(level, SCORING_WEIGHTS["moderate"])
    
    base_score = (
        weights["progress"] * progress +
        weights["feasibility"] * feasibility +
        weights["confidence"] * confidence
    )
    
    # Risk penalty increases with exploration depth
    adjusted_score = base_score - weights["risk_penalty"] * risk
    
    return max(0.0, min(1.0, adjusted_score))
