#!/usr/bin/env python3
"""
Setup script for Token2Metrics framework.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully!")

def setup_environment():
    """Setup the environment for Token2Metrics."""
    print("Setting up Token2Metrics environment...")
    
    # Create output directories
    output_dirs = [
        "outputs",
        "outputs/models",
        "outputs/results",
        "outputs/logs"
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    print("Environment setup complete!")

def main():
    """Main setup function."""
    print("Token2Metrics Setup")
    print("=" * 50)
    
    try:
        install_requirements()
        setup_environment()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run tests: pytest tests/ -v")
        print("2. Run demo: python demo.py")
        print("3. Train models: python main.py train --all-models")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
