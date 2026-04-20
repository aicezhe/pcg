"""
Project configuration — all constants in one place.

Centralises paths, column names, and experiment settings so they are
defined once and imported everywhere else. If the dataset or column
names change, this is the only file that needs updating.
"""

from pathlib import Path

# ─── Project paths ─────────────────────────────────────────────────
# PROJECT_ROOT is the top-level folder (two levels up from this file:
# src/config.py -> src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"

DATASET_PATH = DATA_DIR / "estrazione.xlsx"

# ─── Dataset column names ──────────────────────────────────────────
# Input features used for both classification and regression
INPUT_COLUMNS = [
    "Condition",
    "Temperature [°C]",
    "Plastic Strain",
    "Max Strain Rate",
    "Effective Stress [MPa]",
]

# Target column names
PCG_COLUMN = "PCG (corretti)"           # classification target (yes/no)
GRAIN_SIZE_COLUMN = "Dimensione Grano"  # regression target (continuous)

# ─── Experiment settings ───────────────────────────────────────────
RANDOM_STATE = 1      # for reproducibility across all experiments
TEST_SIZE = 0.2       # 80/20 train/test split

# ─── Human-readable feature names (for plots) ──────────────────────
# Used on axis labels to keep plots clean and professional
FEATURE_DISPLAY_NAMES = [
    "Condition",
    "Temperature",
    "Plastic Strain",
    "Max Strain Rate",
    "Effective Stress",
]
