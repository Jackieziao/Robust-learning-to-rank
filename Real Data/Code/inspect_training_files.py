import os
from pathlib import Path

base = Path(r"d:/Desktop/Robust-learning-to-rank/Real Data/almrrc2021/almrrc2021-data-training")

print("Base path:", base)
print("Exists:", base.exists())

for section in ["model_build_inputs", "model_apply_inputs", "model_score_inputs"]:
    sec = base / section
    print(f"\nSection: {section}")
    if not sec.exists():
        print("  missing")
        continue
    for p in sorted(sec.iterdir()):
        try:
            size = p.stat().st_size
        except Exception:
            size = "?"
        print(f"  {p.name} - {size} bytes")
