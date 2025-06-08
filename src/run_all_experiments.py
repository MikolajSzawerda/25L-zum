#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def main():
    scripts = [
        'run_algorithm_comparison.py',
        'run_preprocessing_study.py', 
        'run_dimensionality_study.py',
        'run_class_balancing_study.py'
    ]
    
    for script in scripts:
        script_path = Path(__file__).parent / script
        subprocess.run([sys.executable, str(script_path)], check=True)

if __name__ == "__main__":
    main() 