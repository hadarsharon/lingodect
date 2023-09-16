"""
Configuration module for project constants, paths, and other helper variables.
"""

from pathlib import Path


class Paths:
    """Helper class for OS-agnostic filepaths around the project root"""
    PROJECT_ROOT = Path(__file__).parent.parent
    DATASETS = PROJECT_ROOT / "datasets"
    UTILS = PROJECT_ROOT / "utils"
    MODELS = PROJECT_ROOT / "models"
