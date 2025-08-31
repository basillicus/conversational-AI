"""prepare_data.py

Creates a small, synthetic intent dataset for the demo.
Outputs `data/intent_data.csv` which is consumed by train_model.py.

This file is intentionally simple and educational: replace it with your
own data pipeline for real projects.
"""

import csv
from pathlib import Path
from examples import (
    examples_initial,
    examples_extended,
    examples_500,
    examples_politeness,
    examples_rudeness,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# examples = examples_initial
# examples = examples_initial + examples_extended
# examples = examples_initial + examples_extended + examples_500
examples = examples_initial + examples_extended + examples_500 + examples_politeness

OUT = DATA_DIR / "intent_data.csv"
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "intent"])
    for text, intent in examples:
        writer.writerow([text, intent])

print(f"Written synthetic dataset to {OUT.resolve()} ({len(examples)} rows).")
