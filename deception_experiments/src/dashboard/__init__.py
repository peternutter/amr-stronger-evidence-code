"""Dashboard package for Streamlit visualization application.

This package contains:
- app.py: Main Streamlit application
- data_loader/: Data loading and caching infrastructure
- components/: Reusable UI components and render functions
"""

from src.dashboard.data_loader import DashboardDataLoader

__all__ = [
    "DashboardDataLoader",
]
