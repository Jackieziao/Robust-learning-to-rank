from pathlib import Path
from typing import Dict, Any
import json
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def change_to_parent_dir_once(Indicator=False):
    if not hasattr(change_to_parent_dir_once, "cached_path"):
        # Base path = the folder where this .py file is located
        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(base_path)

        if not Indicator:
            os.chdir(parent_path)

        # Cache the result (cwd after the first call)
        change_to_parent_dir_once.cached_path = os.getcwd()

    # Always return the same cached path after the first call
    return change_to_parent_dir_once.cached_path

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_empty_json_file_fast(path: Path) -> bool:
    """
    Fast check for tiny empty JSON files.
    """
    if path.stat().st_size > 128:
        return False

    text = path.read_text(encoding="utf-8").strip()
    return text in {"{}", "null", "[]"}


def load_all_json_dicts(
    base_dir: str | Path = "D:/OneDrive/文档/Synfiles/Project (LTR)/Dataset/almrrc2021/almrrc2021-data-training",
    output_dir: str | Path = "D:/OneDrive/文档/Synfiles/Project (LTR)/Dataset/processed_json_pickles",
    rebuild: bool = False,
) -> Dict[str, dict]:
    """
    Load all JSON files under base_dir.

    Assumption:
    - every non-empty JSON file is a dict

    Behavior:
    - empty JSON files are skipped
    - each JSON dict is saved as one pickle
    - cached pickle is reused if available
    - returns one whole dict: file_key -> JSON dict
    """

    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, dict] = {}

    json_files = sorted(base_dir.rglob("*.json"))

    for json_file in json_files:
        relative = json_file.relative_to(base_dir)

        key = "__".join(relative.with_suffix("").parts)
        pickle_file = output_dir / f"{key}.pkl"

        # 1. Skip empty JSON quickly
        if is_empty_json_file_fast(json_file):
            print(f"Skipping empty JSON: {relative}")

            if pickle_file.exists():
                pickle_file.unlink()

            continue

        # 2. Load cached pickle if valid
        cache_is_valid = (
            pickle_file.exists()
            and not rebuild
            and pickle_file.stat().st_mtime >= json_file.stat().st_mtime
        )

        if cache_is_valid:
            try:
                with pickle_file.open("rb") as f:
                    obj = pickle.load(f)

                if isinstance(obj, dict) and len(obj) > 0:
                    result[key] = obj
                    print(f"Loaded cached: {key}, dict len={len(obj)}")
                    continue

                print(f"Bad or empty cache, rebuilding: {key}")

            except Exception as e:
                print(f"Cache failed, rebuilding: {key}")
                print(f"  reason: {e}")

        # 3. Read raw JSON
        print(f"Reading JSON: {relative}")

        obj = read_json(json_file)

        # 4. Skip empty JSON
        if obj is None or obj == {} or obj == []:
            print(f"Skipping empty JSON after read: {relative}")

            if pickle_file.exists():
                pickle_file.unlink()

            continue

        # 5. Validate dict
        if not isinstance(obj, dict):
            raise TypeError(
                f"Expected dict JSON, but got {type(obj).__name__}: {relative}"
            )

        # 6. Save pickle
        with pickle_file.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        result[key] = obj

        print(f"Saved: {key}, dict len={len(obj)}")

    print(f"\nFinished. Loaded {len(result)} non-empty JSON dictionaries.")
    return result