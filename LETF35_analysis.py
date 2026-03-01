# CHANGELOG v35.0 (Development Branch):
#
# • Created LETF35 from LETF34 frozen baseline
# • Version increment for safe experimental development
# • No functional changes yet
# • Reserved for future calibration and accuracy improvements

"""
LETF ULTIMATE v35.0 - Refactored Package Launcher

Backward-compatible entry point. All code now lives in the letf/ package.
Run with: python LETF35_analysis.py
"""
import sys
import os

# Ensure the package directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from letf import run

if __name__ == "__main__":
    print("Running LETF35_analysis.py — Development Version 35.0")
    run()
