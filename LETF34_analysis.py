"""
LETF ULTIMATE v8.1 - Refactored Package Launcher

Backward-compatible entry point. All code now lives in the letf/ package.
Run with: python LETF34_analysis.py
"""
import sys
import os

# Ensure the package directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from letf import run

if __name__ == "__main__":
    run()
