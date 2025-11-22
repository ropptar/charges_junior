from pathlib import Path

INPUT_DIR = Path(__file__).parent / "charges"
OUTPUT_DIR = Path(__file__).parent / "output"
SHEAR_MAGNITUDE = [32, 32, 32, 25]
CROP_CONFIG = [
    {"y_start_ratio": 0.28, "y_end_ratio": 0.5},
    {"y_start_ratio": 0.3, "y_end_ratio": 0.5},
    {"y_start_ratio": 0.67, "y_end_ratio": 0.8},
    {"y_start_ratio": 0.75, "y_end_ratio": 1},
]
DARK_CONFIG = [
    [0, 1, 7, 8, 9],
    [0, 8, 9],
    [0, 1, 2, 3, 4, 5, 11, 12, 13, 14],
    [0, 1, 2, 3, 13, 14, 15, 16, 17, 18, 19],
]
