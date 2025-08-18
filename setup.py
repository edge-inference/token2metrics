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

def main():
    """Main setup function."""
    print("Token2Metrics Setup")
    print("=" * 50)
    
    try:
        install_requirements()
        print("\n✅ Setup completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
