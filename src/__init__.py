"""
src package initializer

This file marks the 'src' directory as a Python package.
It also helps set up the import path when scripts are run
from different directories (like 'experiments/').
"""

import os
import sys

# Automatically add the project root (one level above 'src') to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Optional: simple confirmation print (uncomment if debugging)
# print(f"[src] Package initialized, project root: {project_root}")
