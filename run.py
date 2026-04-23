#!/usr/bin/env python3
"""
Root-level launcher for ParkGuideAI application.
Runs the unified UI from src/app.py with proper path handling.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if __name__ == "__main__":
    from app import build_ui
    app = build_ui()
    app.launch()
