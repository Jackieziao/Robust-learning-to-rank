import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd
except Exception:
    pd = None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_build_inputs(root: str | Path) -> Dict[str, Any]:
    """Load the model_build_inputs JSON files from the training dataset.

    Args:
        root: path to the folder containing the files (e.g.
            'Real Data/almrrc2021/almrrc2021-data-training/model_build_inputs')

    Returns:
        A dict with keys: 'package_data', 'route_data', 'travel_times',
        'actual_sequences', 'invalid_sequence_scores'. Values are the parsed
        JSON objects. If pandas is available, tabular objects are converted to
        DataFrame when sensible.
    """
    root_path = Path(root)
    files = {
        "package_data": "package_data.json",
        "route_data": "route_data.json",
        "travel_times": "travel_times.json",
        "actual_sequences": "actual_sequences.json",
        "invalid_sequence_scores": "invalid_sequence_scores.json",
    }

    data: Dict[str, Any] = {}
    for key, fname in files.items():
        p = root_path / fname
        if not p.exists():
            data[key] = None
            continue
        obj = _read_json(p)
        # convert lists/dicts to DataFrame when pandas available and object is list
        if pd is not None and isinstance(obj, list):
            try:
                data[key] = pd.DataFrame(obj)
            except Exception:
                data[key] = obj
        else:
            data[key] = obj

    return data


# def load_training_dataset(base_dir: str | Path = "Real Data/almrrc2021/almrrc2021-data-training") -> Dict[str, Any]:
#     """Load the common training dataset groups.

#     Returns a dict with sub-keys 'model_build_inputs', 'model_apply_inputs',
#     'model_score_inputs' when the folders exist.
#     """
#     base = Path(base_dir)
#     result: Dict[str, Any] = {}

#     build_inputs = base / "model_build_inputs"
#     if build_inputs.exists():
#         result["model_build_inputs"] = load_model_build_inputs(build_inputs)
#     else:
#         result["model_build_inputs"] = None

#     apply_inputs = base / "model_apply_inputs"
#     if apply_inputs.exists():
#         # load all JSON files in model_apply_inputs
#         result["model_apply_inputs"] = {p.name: _read_json(p) for p in apply_inputs.glob("*.json")}
#     else:
#         result["model_apply_inputs"] = None

#     score_inputs = base / "model_score_inputs"
#     if score_inputs.exists():
#         result["model_score_inputs"] = {p.name: _read_json(p) for p in score_inputs.glob("*.json")}
#     else:
#         result["model_score_inputs"] = None

#     return result

from pathlib import Path
from typing import Dict, Any
import pickle


def load_training_dataset(
    base_dir: str | Path = "Real Data/almrrc2021/almrrc2021-data-training",
    pickle_path: str | Path = "training_dataset.pkl",
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Load the common training dataset groups.

    If pickle_path exists, load the dataset from pickle.
    Otherwise, read the raw data files and save the loaded result as pickle.

    Parameters
    ----------
    base_dir:
        Path to the ALMRRC training dataset folder.

    pickle_path:
        Path where the loaded dataset should be cached.

    force_reload:
        If True, ignore existing pickle and reload from raw files.

    Returns
    -------
    Dict[str, Any]
        Dataset dict with keys:
        - model_build_inputs
        - model_apply_inputs
        - model_score_inputs
    """

    base = Path(base_dir)
    pickle_path = Path(pickle_path)

    # --------------------------------------------------
    # 1. Load cached result if it already exists
    # --------------------------------------------------

    if pickle_path.exists() and not force_reload:
        print(f"Loading cached dataset from: {pickle_path}")

        with open(pickle_path, "rb") as f:
            result = pickle.load(f)

        return result

    # --------------------------------------------------
    # 2. Otherwise load raw dataset
    # --------------------------------------------------

    print("No cached pickle found. Loading raw dataset...")

    result: Dict[str, Any] = {}

    build_inputs = base / "model_build_inputs"
    if build_inputs.exists():
        result["model_build_inputs"] = load_model_build_inputs(build_inputs)
    else:
        result["model_build_inputs"] = None

    apply_inputs = base / "model_apply_inputs"
    if apply_inputs.exists():
        result["model_apply_inputs"] = {
            p.name: _read_json(p)
            for p in apply_inputs.glob("*.json")
        }
    else:
        result["model_apply_inputs"] = None

    score_inputs = base / "model_score_inputs"
    if score_inputs.exists():
        result["model_score_inputs"] = {
            p.name: _read_json(p)
            for p in score_inputs.glob("*.json")
        }
    else:
        result["model_score_inputs"] = None

    # --------------------------------------------------
    # 3. Save loaded result as pickle
    # --------------------------------------------------

    with open(pickle_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved cached dataset to: {pickle_path}")

    return result


if __name__ == "__main__":
    # Basic CLI test: load the default training folder and print summary
    base = Path("Real Data/almrrc2021/almrrc2021-data-training")
    ds = load_training_dataset(base)
    def pretty_count(obj):
        if obj is None:
            return "missing"
        if pd is not None and isinstance(obj, pd.DataFrame):
            return f"DataFrame rows={len(obj)} cols={len(obj.columns)}"
        if isinstance(obj, dict):
            return f"{len(obj)} items"
        if isinstance(obj, list):
            return f"list len={len(obj)}"
        return type(obj).__name__

    print("Training dataset summary:")
    for section, content in ds.items():
        print(f"- {section}: {pretty_count(content)}")
        if isinstance(content, dict):
            for k, v in content.items():
                # nested check
                if isinstance(v, dict) or isinstance(v, list) or (pd is not None and hasattr(v, "shape")):
                    print(f"    - {k}: {pretty_count(v)}")
