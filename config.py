#!/usr/bin/env python3
"""
Configuration and Scoring Guidelines for Tree of Thought

Provides depth-based scoring recommendations and exploration strategies.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class DepthGuideline:
    """Scoring guidelines for a specific depth"""
    description: str
    score_range: Tuple[float, float]
    strategy: str
    example_scores: List[float]
    diversity_requirement: str

# Depth-based scoring guidelines
DEPTH_GUIDELINES: Dict[int, DepthGuideline] = {
    0: DepthGuideline(
        description="Root - Core architectural patterns",
        score_range=(0.50, 0.70),
        strategy="Generate 4 distinct architectural approaches. Score based on constraint satisfaction and potential. Keep all below 0.70 to ensure expansion.",
        example_scores=[0.55, 0.60, 0.58, 0.62],
        diversity_requirement="Maximum diversity - orthogonal approaches",
    ),
    1: DepthGuideline(
        description="Depth 1 - Architecture refinements",
        score_range=(0.55, 0.72),
        strategy="Refinements of parent pattern. Top candidates can reach 0.70-0.72, but keep most at 0.55-0.65.",
        example_scores=[0.58, 0.65, 0.60, 0.72],
        diversity_requirement="High diversity - explore variations",
    ),
    2: DepthGuideline(
        description="Depth 2 - Specific mechanisms",
        score_range=(0.50, 0.75),
        strategy="Specific implementations. Exceptional mechanisms can reach 0.70-0.75. Alternatives stay 0.50-0.65.",
        example_scores=[0.52, 0.68, 0.55, 0.73],
        diversity_requirement="Medium diversity - implementation variants",
    ),
    3: DepthGuideline(
        description="Depth 3 - Implementation details",
        score_range=(0.45, 0.70),
        strategy="Implementation specifics. Scores trend lower (0.45-0.60) unless exceptional insight.",
        example_scores=[0.48, 0.55, 0.50, 0.65],
        diversity_requirement="Low diversity - focus on details",
    ),
    4: DepthGuideline(
        description="Depth 4+ - Deep refinements",
        score_range=(0.40, 0.65),
        strategy="Deep details, edge cases, optimizations. Most scores 0.40-0.55. Only breakthrough insights reach 0.60-0.65.",
        example_scores=[0.42, 0.48, 0.45, 0.58],
        diversity_requirement="Minimal diversity - refinement only",
    ),
}

def get_depth_guideline(depth: int) -> DepthGuideline:
    """Get scoring guideline for specific depth"""
    if depth in DEPTH_GUIDELINES:
        return DEPTH_GUIDELINES[depth]
    # For depths > 4, use depth 4 guideline
    return DEPTH_GUIDELINES[4]

# Exploration level configurations
DEPTH_CONFIGS = {
    "shallow": {
        "name": "shallow",
        "description": "Surface-level exploration for quick decisions",
        "node_budget": 20,
        "min_consumption_ratio": 0.80,
        "beam_width": 2,
        "n_generate": 2,
        "max_depth": 2,
        "target_score": 0.70,
        "score_range": (0.40, 0.65),
        "use_when": [
            "Simple binary decisions",
            "Well-understood problems",
            "Time-critical (< 5 min)",
            "Low-stakes choices",
        ],
        "avoid_when": [
            "Novel problems",
            "High-stakes decisions",
            "Safety-critical systems",
        ],
    },
    "moderate": {
        "name": "moderate",
        "description": "Balanced exploration for technology selection",
        "node_budget": 50,
        "min_consumption_ratio": 0.85,
        "beam_width": 3,
        "n_generate": 3,
        "max_depth": 3,
        "target_score": 0.75,
        "score_range": (0.50, 0.75),
        "use_when": [
            "Technology selection",
            "Architecture patterns",
            "Medium-stakes decisions",
            "2-4 competing approaches",
        ],
        "avoid_when": [
            "Trivial decisions",
            "Critical safety systems",
            "Novel research",
        ],
    },
    "deep": {
        "name": "deep",
        "description": "Thorough exploration with validation",
        "node_budget": 150,
        "min_consumption_ratio": 0.90,
        "beam_width": 5,
        "n_generate": 4,
        "max_depth": 5,
        "target_score": 0.80,
        "score_range": (0.45, 0.70),
        "use_when": [
            "Novel research problems",
            "High-stakes architecture",
            "Safety-critical systems",
            "Confidence > 0.85 required",
        ],
        "avoid_when": [
            "Simple obvious answers",
            "Time-critical",
            "Low-stakes choices",
        ],
    },
    "exhaustive": {
        "name": "exhaustive",
        "description": "Maximum rigor for publication-grade research",
        "node_budget": 500,
        "min_consumption_ratio": 0.95,
        "beam_width": 7,
        "n_generate": 5,
        "max_depth": 8,
        "target_score": 0.85,
        "score_range": (0.40, 0.70),
        "use_when": [
            "Publication-grade research",
            "Fundamental design decisions",
            "Maximum confidence required",
            "Months of work at stake",
        ],
        "avoid_when": [
            "Simple decisions",
            "Time < 30 min",
            "Budget constraints",
        ],
    },
}

def get_exploration_config(level: str) -> Dict[str, Any]:
    """Get configuration for exploration level"""
    return DEPTH_CONFIGS.get(level.lower(), DEPTH_CONFIGS["moderate"])

# Candidate generation strategies
CANDIDATE_STRATEGIES = {
    "shallow": [
        "Standard approach",
        "Alternative approach",
        "Hybrid (if obvious)",
    ],
    "moderate": [
        "Established pattern A",
        "Established pattern B",
        "Hybrid A+B",
        "Novel/emerging pattern",
    ],
    "deep": [
        "Established pattern A",
        "Established pattern B",
        "Established pattern C",
        "Hybrid A+B",
        "Hybrid A+C",
        "Radical alternative",
    ],
    "exhaustive": [
        "Established pattern A",
        "Established pattern B",
        "Established pattern C",
        "Hybrid A+B",
        "Hybrid A+C",
        "Hybrid B+C",
        "Radical alternative",
        "Edge case optimized",
        "Risk-minimized conservative",
        "Wild card unconventional",
    ],
}

def get_candidate_strategies(level: str) -> List[str]:
    """Get candidate generation strategies for level"""
    return CANDIDATE_STRATEGIES.get(level.lower(), CANDIDATE_STRATEGIES["moderate"])

# Scoring formula explanation
SCORING_FORMULA = """
Score = (progress_weight × progress) + 
        (feasibility_weight × feasibility) + 
        (confidence_weight × confidence) - 
        (risk_penalty × risk)

Weights by level:
- shallow:    progress=0.50, feasibility=0.30, confidence=0.20, risk=0.20
- moderate:   progress=0.45, feasibility=0.35, confidence=0.20, risk=0.25
- deep:       progress=0.40, feasibility=0.30, confidence=0.15, risk=0.35
- exhaustive: progress=0.35, feasibility=0.25, confidence=0.15, risk=0.40

Note: Higher levels penalize risk more heavily.
"""
