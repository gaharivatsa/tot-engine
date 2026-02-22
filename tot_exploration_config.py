#!/usr/bin/env python3
"""
Tree of Thought Exploration Configuration Dictionary
Guides candidate generation and scoring for different exploration levels
"""

from typing import Dict, List, Callable
from dataclasses import dataclass

@dataclass
class ExplorationConfig:
    """Configuration for a specific exploration level"""
    name: str
    description: str
    
    # Scoring guidelines
    score_range: tuple  # (min, max) for progress_estimate
    score_strategy: str  # How to distribute scores
    
    # Candidate generation
    diversity_requirement: str  # How different candidates should be
    depth_focus: str  # What to focus on at this depth
    
    # When to use
    use_when: List[str]
    avoid_when: List[str]
    
    # Action guidelines
    what_to_do: List[str]
    what_to_avoid: List[str]
    
    # Scoring formula weights (can override default)
    progress_weight: float = 0.45
    feasibility_weight: float = 0.35
    confidence_weight: float = 0.20
    risk_penalty: float = 0.25

# ============================================================================
# EXPLORATION LEVEL CONFIGURATIONS
# ============================================================================

LOW_EXPLORATION = ExplorationConfig(
    name="low",
    description="Surface-level exploration for quick decisions or obvious choices",
    
    # Scoring: Keep MOST candidates below target to force minimal expansion
    score_range=(0.40, 0.65),
    score_strategy="Distribute scores evenly between 0.40-0.65. Only ONE candidate should approach 0.70. This ensures multiple expansion rounds are needed.",
    
    # Candidate generation
    diversity_requirement="Minimal diversity - focus on 2-3 distinct approaches, then variations",
    depth_focus="Surface-level descriptions only. No implementation details.",
    
    # When to use
    use_when=[
        "Simple binary decisions (A vs B)",
        "Well-understood problem with known solutions",
        "Time-critical decisions (< 5 min)",
        "Low-stakes choices (reversible, low cost)",
        "Initial feasibility check before deeper research"
    ],
    
    avoid_when=[
        "Novel or unprecedented problems",
        "High-stakes architectural decisions",
        "Problems with >3 competing constraints",
        "Safety-critical systems",
        "When confidence > 0.80 is required"
    ],
    
    # Action guidelines
    what_to_do=[
        "Generate 2-3 genuinely different approaches",
        "Score conservatively (0.40-0.65 range)",
        "Focus on high-level concepts",
        "Stop after 2-3 depths if target not reached",
        "Document assumptions clearly"
    ],
    
    what_to_avoid=[
        "Scoring anything above 0.70 (stops exploration too early)",
        "Going deeper than depth 3",
        "Over-engineering simple solutions",
        "Spending > 10 minutes",
        "Adding unnecessary complexity"
    ]
)

MODERATE_EXPLORATION = ExplorationConfig(
    name="moderate",
    description="Balanced exploration for important decisions with trade-offs",
    
    # Scoring: Balanced distribution with few high scores
    score_range=(0.50, 0.75),
    score_strategy="Most candidates 0.50-0.65. Top 1-2 candidates can reach 0.70-0.75. This allows some pruning while maintaining exploration pressure.",
    
    # Candidate generation
    diversity_requirement="High diversity - explore orthogonal approaches and hybrids",
    depth_focus="Architecture-level details. Implementation sketch but not code.",
    
    # When to use
    use_when=[
        "Technology selection (database, framework, etc.)",
        "Architecture pattern decisions",
        "Resource allocation trade-offs",
        "Medium-stakes decisions (weeks of work)",
        "When 2-4 competing approaches exist",
        "Need confidence 0.75-0.85"
    ],
    
    avoid_when=[
        "Trivial decisions with clear winners",
        "Existential/critical safety decisions",
        "Problems requiring novel research",
        "When empirical validation is impossible"
    ],
    
    # Action guidelines
    what_to_do=[
        "Generate 4 distinct architectural approaches",
        "Explore hybrid combinations (A+B, A+C)",
        "Score top candidates 0.70-0.75, others 0.50-0.65",
        "Expand to depth 4-5 for promising paths",
        "Include at least one 'wild card' unconventional option",
        "Document trade-offs explicitly"
    ],
    
    what_to_avoid=[
        "Scoring > 0.80 (reserve for exhaustive)",
        "Getting lost in implementation details too early",
        "Ignoring constraints",
        "Biasing toward familiar solutions",
        "Stopping before depth 3"
    ]
)

HIGH_EXPLORATION = ExplorationConfig(
    name="high",
    description="Deep exhaustive exploration for critical, novel, or complex decisions",
    
    # Scoring: Keep most low to force maximum expansion
    score_range=(0.45, 0.70),
    score_strategy="Distribute 0.45-0.60 for most candidates. Only exceptional candidates reach 0.65-0.70. This forces deep tree exploration before any path is marked terminal.",
    
    # Candidate generation
    diversity_requirement="Maximum diversity - explore adjacent possibles, radical alternatives, and edge cases",
    depth_focus="Deep implementation details, failure modes, verification strategies",
    
    # When to use
    use_when=[
        "Novel research problems",
        "High-stakes architectural decisions (months of work)",
        "Safety-critical systems",
        "When confidence > 0.85 required",
        "Problems with 5+ competing constraints",
        "Fundamental design decisions affecting entire system",
        "Publication-grade research"
    ],
    
    avoid_when=[
        "Simple decisions with obvious answers",
        "Time-critical (need answer in < 30 min)",
        "Low-stakes reversible choices",
        "When budget constraints prohibit deep exploration"
    ],
    
    # Action guidelines
    what_to_do=[
        "Generate 4+ radically different approaches",
        "Explore 3+ depths for each promising path",
        "Score conservatively: most 0.45-0.60, best 0.65-0.70",
        "Include edge cases and failure modes",
        "Generate hybrid approaches at depth 2-3",
        "Explore 'what if' scenarios (what if X fails?)",
        "Add verification/validation strategies",
        "Document uncertainty explicitly",
        "Consider adversarial perspectives"
    ],
    
    what_to_avoid=[
        "Scoring anything > 0.75 until depth 5+",
        "Stopping at depth 3 or earlier",
        "Generating similar candidates (low diversity)",
        "Ignoring long-tail risks",
        "Assuming constraints are satisfied without verification",
        "Biasing toward first plausible solution",
        "Ignoring empirical validation needs"
    ],
    
    # Override weights for high-stakes decisions
    progress_weight=0.40,
    feasibility_weight=0.30,
    confidence_weight=0.15,
    risk_penalty=0.35  # Higher risk penalty for critical decisions
)

# ============================================================================
# SCORING GUIDELINES BY DEPTH
# ============================================================================

DEPTH_SCORING_GUIDE = {
    0: {
        "description": "Root - Core architectural patterns",
        "score_range": (0.50, 0.70),
        "guidance": "At root, generate 4 distinct architectural approaches. Score based on: (1) Constraint satisfaction, (2) Novelty, (3) Potential. None should exceed 0.70 to ensure expansion.",
        "example_scores": [0.55, 0.60, 0.58, 0.62]
    },
    1: {
        "description": "Depth 1 - Architecture refinements",
        "score_range": (0.55, 0.72),
        "guidance": "Refinements of parent pattern. Score improvements over parent. Can reach 0.72 for exceptional refinements, but keep most at 0.55-0.65.",
        "example_scores": [0.58, 0.65, 0.60, 0.72]
    },
    2: {
        "description": "Depth 2 - Specific mechanisms",
        "score_range": (0.50, 0.75),
        "guidance": "Specific implementations. Can score higher (0.70-0.75) if mechanism is elegant and well-suited. But keep alternatives at 0.50-0.65.",
        "example_scores": [0.52, 0.68, 0.55, 0.73]
    },
    3: {
        "description": "Depth 3 - Implementation details",
        "score_range": (0.45, 0.70),
        "guidance": "Implementation specifics. Scores should trend lower (0.45-0.60) unless exceptional insight. This depth tests feasibility.",
        "example_scores": [0.48, 0.55, 0.50, 0.65]
    },
    4: {
        "description": "Depth 4+ - Deep refinements",
        "score_range": (0.40, 0.65),
        "guidance": "Deep details, edge cases, optimizations. Most scores 0.40-0.55. Only breakthrough insights reach 0.60-0.65.",
        "example_scores": [0.42, 0.48, 0.45, 0.58]
    }
}

# ============================================================================
# CANDIDATE GENERATION STRATEGIES
# ============================================================================

CANDIDATE_STRATEGIES = {
    "diversity_patterns": {
        "low": [
            "Standard approach",
            "Alternative approach",
            "Hybrid (if obvious)"
        ],
        "moderate": [
            "Established pattern A",
            "Established pattern B",
            "Hybrid A+B",
            "Novel/emerging pattern"
        ],
        "high": [
            "Established pattern A",
            "Established pattern B",
            "Established pattern C",
            "Hybrid A+B",
            "Hybrid A+C",
            "Radical alternative",
            "Edge case optimized",
            "Risk-minimized conservative"
        ]
    },
    
    "evaluation_dimensions": {
        "progress_estimate": {
            "description": "How well does this solve the problem? (0-1)",
            "low": "0.40-0.65: Partial solutions acceptable",
            "moderate": "0.50-0.75: Good solutions expected",
            "high": "0.45-0.70: Only thorough solutions score high"
        },
        "feasibility_estimate": {
            "description": "Can we actually implement this? (0-1)",
            "low": "0.50-0.80: Rough feasibility okay",
            "moderate": "0.60-0.85: Must be implementable",
            "high": "0.55-0.80: Rigorous feasibility check"
        },
        "risk_estimate": {
            "description": "What could go wrong? (0-1, higher = riskier)",
            "low": "0.10-0.40: Rough risk assessment",
            "moderate": "0.15-0.35: Consider known failure modes",
            "high": "0.20-0.50: Deep risk analysis required"
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_exploration_config(level: str) -> ExplorationConfig:
    """Get configuration for exploration level"""
    configs = {
        "low": LOW_EXPLORATION,
        "moderate": MODERATE_EXPLORATION,
        "high": HIGH_EXPLORATION
    }
    return configs.get(level.lower(), MODERATE_EXPLORATION)

def get_depth_scoring_guide(depth: int) -> dict:
    """Get scoring guidance for specific depth"""
    if depth in DEPTH_SCORING_GUIDE:
        return DEPTH_SCORING_GUIDE[depth]
    # For depths > 4, use depth 4 guidance
    return DEPTH_SCORING_GUIDE[4]

def generate_score_recommendation(
    exploration_level: str,
    depth: int,
    candidate_index: int,
    is_best_candidate: bool = False
) -> float:
    """
    Recommend a score based on exploration level and context
    
    Args:
        exploration_level: 'low', 'moderate', or 'high'
        depth: Current tree depth
        candidate_index: Which candidate (0-3 typically)
        is_best_candidate: Is this the strongest candidate?
    
    Returns:
        Recommended score in range (0.40-0.75)
    """
    config = get_exploration_config(exploration_level)
    depth_guide = get_depth_scoring_guide(depth)
    
    min_score, max_score = depth_guide["score_range"]
    
    if exploration_level == "low":
        # Even distribution, slight bias for best
        if is_best_candidate:
            return round(min(max_score, 0.65), 2)
        else:
            return round(min_score + (candidate_index * 0.05), 2)
    
    elif exploration_level == "moderate":
        # Best candidate can reach higher, others conservative
        if is_best_candidate:
            return round(min(max_score, 0.73), 2)
        elif candidate_index == 1:
            return round(min_score + 0.10, 2)
        else:
            return round(min_score + (candidate_index * 0.03), 2)
    
    else:  # high
        # Very conservative, only exceptional candidates score high
        if is_best_candidate and depth >= 3:
            return round(min(max_score, 0.70), 2)
        elif is_best_candidate:
            return round(min(max_score - 0.05, 0.65), 2)
        else:
            return round(min_score + (candidate_index * 0.04), 2)

def should_continue_exploration(
    nodes_used: int,
    node_budget: int,
    exploration_level: str,
    current_depth: int
) -> bool:
    """
    Determine if exploration should continue
    
    Returns True if we should keep expanding
    """
    config = get_exploration_config(exploration_level)
    min_required = int(node_budget * config.min_consumption_ratio)
    
    # Must meet minimum budget
    if nodes_used < min_required:
        return True
    
    # For high exploration, go deeper
    if exploration_level == "high" and current_depth < 5:
        return True
    
    return False

def print_exploration_guide(level: str):
    """Print human-readable exploration guide"""
    config = get_exploration_config(level)
    
    print(f"\n{'='*70}")
    print(f"EXPLORATION GUIDE: {config.name.upper()}")
    print(f"{'='*70}")
    print(f"\nDescription: {config.description}")
    print(f"\nScore Range: {config.score_range[0]:.2f} - {config.score_range[1]:.2f}")
    print(f"\nScore Strategy:\n  {config.score_strategy}")
    
    print(f"\n{'='*70}")
    print("WHEN TO USE:")
    print(f"{'='*70}")
    for item in config.use_when:
        print(f"  ✅ {item}")
    
    print(f"\n{'='*70}")
    print("AVOID WHEN:")
    print(f"{'='*70}")
    for item in config.avoid_when:
        print(f"  ❌ {item}")
    
    print(f"\n{'='*70}")
    print("WHAT TO DO:")
    print(f"{'='*70}")
    for item in config.what_to_do:
        print(f"  ✅ {item}")
    
    print(f"\n{'='*70}")
    print("WHAT TO AVOID:")
    print(f"{'='*70}")
    for item in config.what_to_avoid:
        print(f"  ❌ {item}")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Tree of Thought Exploration Configuration")
    print("=" * 70)
    
    # Show all guides
    for level in ["low", "moderate", "high"]:
        print_exploration_guide(level)
    
    # Show depth scoring
    print("\n" + "="*70)
    print("DEPTH SCORING GUIDE")
    print("="*70)
    
    for depth, guide in DEPTH_SCORING_GUIDE.items():
        print(f"\nDepth {depth}: {guide['description']}")
        print(f"  Range: {guide['score_range']}")
        print(f"  Example scores: {guide['example_scores']}")
    
    # Show score recommendations
    print("\n" + "="*70)
    print("SCORE RECOMMENDATION EXAMPLES")
    print("="*70)
    
    for level in ["low", "moderate", "high"]:
        print(f"\n{level.upper()} exploration at depth 1:")
        for i, is_best in enumerate([False, False, False, True]):
            score = generate_score_recommendation(level, 1, i, is_best)
            marker = " (BEST)" if is_best else ""
            print(f"  Candidate {i+1}: {score:.2f}{marker}")
