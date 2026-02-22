"""
Tree of Thought Engine - Production Grade MCP Server

A unified MCP server providing both standard and enforced Tree of Thought reasoning.

Usage:
    # Standard mode (flexible)
    mcporter call tot-engine.tot_start_run task_prompt="..."
    
    # Enforced mode (guaranteed depth)
    mcporter call tot-engine.tot_start_run_enforced 
        task_prompt="..." 
        exploration_level="deep"
"""

__version__ = "1.0.0"
__author__ = "Thunderbolt AI"
__license__ = "MIT"

from .server import mcp, tot_start_run, tot_request_samples, tot_submit_samples
from .enforcement import EnforcementEngine, ExplorationLevel
from .config import DEPTH_CONFIGS, get_exploration_config

__all__ = [
    'mcp',
    'EnforcementEngine',
    'ExplorationLevel',
    'DEPTH_CONFIGS',
    'get_exploration_config',
]
