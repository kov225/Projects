"""
main.py  Project Entry Point  Milestone 2

Thin wrapper that delegates to the experiment orchestration module.
Separating the entry point from the orchestration module allows the
orchestrator to be imported and tested independently without triggering
a full pipeline run.

Usage:
    python main.py
"""

import sys
import os

# Ensure the src directory is on the path regardless of the working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from run_experiments import main

if __name__ == "__main__":
    main()
