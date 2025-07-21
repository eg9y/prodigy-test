#!/usr/bin/env python3
"""
Helper script to run Prodigy commands in a cross-platform way
"""

import subprocess
import sys
import os
from pathlib import Path

def run_prodigy_command(args):
    """
    Run a Prodigy command with proper environment handling
    
    Args:
        args: List of command arguments
    """
    
    # Try different ways to run Prodigy
    commands_to_try = [
        # Try uv first
        ["uv", "run", "python", "-m", "prodigy"] + args,
        # Try direct python
        ["python", "-m", "prodigy"] + args,
        # Try python3
        ["python3", "-m", "prodigy"] + args,
        # Try with explicit path on Windows
        [sys.executable, "-m", "prodigy"] + args,
    ]
    
    for cmd in commands_to_try:
        try:
            # Try to run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # If successful, return the result
            if result.returncode == 0 or "prodigy" in result.stdout.lower() or "prodigy" in result.stderr.lower():
                return result
                
        except FileNotFoundError:
            # Try next command
            continue
        except Exception as e:
            # Try next command
            continue
    
    # If all attempts failed, provide helpful error message
    print("Error: Could not run Prodigy. Please ensure:")
    print("1. Prodigy is installed: pip install prodigy")
    print("2. You have a valid Prodigy license")
    print("3. You're in the correct Python environment")
    print("\nTry running manually:")
    print(f"  {sys.executable} -m prodigy " + " ".join(args))
    
    return None

if __name__ == "__main__":
    # Pass all arguments to Prodigy
    if len(sys.argv) > 1:
        result = run_prodigy_command(sys.argv[1:])
        if result:
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
        else:
            sys.exit(1)
    else:
        print("Usage: python run_prodigy.py [prodigy command and arguments]")
        print("Example: python run_prodigy.py db-out goalkeeper_detection output.jsonl")