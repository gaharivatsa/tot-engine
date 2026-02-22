# Tree of Thought (ToT) Engine

A production-grade Tree of Thought implementation for autonomous AI systems.

## Overview

This repository contains the **Tree of Thought Engine** and **Enforced Exhaustive Search** system used by Thunderbolt - an autonomous AI agent running on Raspberry Pi 4.

## Features

- **Option B Client Sampling**: Agent generates thoughts, engine manages tree structure
- **Enforced Budget Consumption**: Minimum 80-95% node utilization before synthesis
- **Depth-Based Scoring**: Guided scoring for low/moderate/high exploration levels
- **Empirical Validation**: Built-in testing framework for architectures
- **MCP Server**: Native integration with Model Context Protocol

## Quick Start

### Installation

```bash
pip install fastmcp pydantic
python server.py
```

### Basic Usage

```python
from enforced_tot import enforced_tot_research

result = enforced_tot_research(
    task_prompt="Design optimal database architecture",
    depth_level="deep",  # shallow | moderate | deep | exhaustive
    constraints=["Pi 4 compatible", "Self-hosted"]
)

print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']}")
print(f"Nodes explored: {result['nodes_explored']}/{result['node_budget']}")
```

## Exploration Levels

| Level | Budget | Min Required | Use Case |
|-------|--------|--------------|----------|
| **shallow** | 20 | 16 (80%) | Quick decisions |
| **moderate** | 50 | 43 (85%) | Technology selection |
| **deep** | 150 | 135 (90%) | Architecture decisions |
| **exhaustive** | 500 | 475 (95%) | Publication-grade research |

## Files

- `server.py` - Core ToT MCP server
- `enforced_tot.py` - Budget enforcement wrapper
- `tot_exploration_config.py` - Scoring guidelines
- `smoke_test.py` - Test suite
- `USAGE.md` - Detailed documentation
- `ENFORCED_TOT_GUIDE.md` - Enforcement system guide

## Architecture

```
Agent (thinks) ←→ tot-engine (MCP Server) ←→ SQLite/Redis
        ↓
  Generates candidates
        ↓
  Engine evaluates & scores
        ↓
  Beam search pruning
        ↓
  Synthesized answer
```

## License

MIT - Created for Thunderbolt Autonomous AI System
