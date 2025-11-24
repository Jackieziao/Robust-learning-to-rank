
import os
import pickle
import torch 
import math
import random

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping

#######################################################################################################################################
######################################################### Common function #############################################################
#######################################################################################################################################

def change_to_parent_dir_once(Indicator=False):
    # Static variables to store the base path and parent path
    if not hasattr(change_to_parent_dir_once, 'BASE_PATH'):
        # Set the base path to the current working directory
        change_to_parent_dir_once.BASE_PATH = os.getcwd()
        change_to_parent_dir_once.PARENT_PATH = os.path.dirname(change_to_parent_dir_once.BASE_PATH)
    if not Indicator:
        # Change to the parent directory
        os.chdir(change_to_parent_dir_once.PARENT_PATH) 
    # Return the current parent path
    return os.getcwd()

class DataWrapper():  # If redefine the dataset, must rewrite len, getitem functions
    
    def __init__(self, x, y):
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y = y if isinstance(x, torch.Tensor) else torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
#######################################################################################################################################
########################################################### Dataset function ##########################################################
#######################################################################################################################################

def load_shortest_path_setting_with_splits(
    base_dir,
    n,
    input_dim,
    deg,
    noise,
    grid_width,
    seed=None,
    batch_size=32,
    train_ratio=0.8,
    valid_ratio=0.1,
    shuffle_train=True,
):
    """
    Load one shortest-path setting and return train/valid/test splits per seed.

    Parameters
    ----------
    base_dir : str
        Directory where 'shortest_path_instances_*.pkl' is stored.
    n : int
        Number of instances per seed (used in filename).
    input_dim : int
        Number of features (used in filename).
    deg : int
        Polynomial degree (used in filename).
    noise : float
        Noise level (used in filename).
    grid_width : int
        Grid width (used in filename).
    seed : int or None
        - If int: return splits ONLY for that seed.
        - If None: return splits for ALL seeds.
    batch_size : int
        Batch size for DataLoaders.
    train_ratio : float
        Fraction of samples used for training.
    valid_ratio : float
        Fraction of samples used for validation.
        Test ratio is inferred as 1 - train_ratio - valid_ratio.
    shuffle_train : bool
        Whether to shuffle batches in the training DataLoader.

    Returns
    -------
    If seed is not None:
        (train_dl, valid_dl, test_dl)

    If seed is None:
        list_of_splits, where each element is a dict:
            {
                "seed": seed_value,
                "train_dl": train_dl,
                "valid_dl": valid_dl,
                "test_dl": test_dl,
            }
    """

    # ------------------------------------------------------------------
    # 1. Load the pickle for this setting
    # ------------------------------------------------------------------
    pkl_path = (
        base_dir +
        f"/shortest_path_instances_n_{n}"
        f"_input_dim_{input_dim}"
        f"_deg_{deg}"
        f"_noise_{noise}"
        f"_grid_width_{grid_width}.pkl"
    )

    with open(pkl_path, "rb") as f:
        instances = pickle.load(f)  # list of dicts: {"seed": s, "x": X_s, "y": Y_s}

    def _make_splits_for_instance(inst):
        """Create train/valid/test DataLoaders for a single seed-instance."""
        X = inst["x"]
        Y = inst["y"]
        n_samples = len(X)

        n_train = int(n_samples * train_ratio)
        n_valid = int(n_samples * valid_ratio)
        n_test = n_samples - n_train - n_valid

        x_train, y_train = X[:n_train], Y[:n_train]
        x_valid, y_valid = X[n_train:n_train + n_valid], Y[n_train:n_train + n_valid]
        x_test,  y_test  = X[n_train + n_valid:],       Y[n_train + n_valid:]

        train_ds = DataWrapper(x_train, y_train)
        valid_ds = DataWrapper(x_valid, y_valid)
        test_ds  = DataWrapper(x_test,  y_test)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        return train_dl, valid_dl, test_dl

    # ------------------------------------------------------------------
    # 2. If a specific seed is requested
    # ------------------------------------------------------------------
    if seed is not None:
        inst = instances[seed]   # assuming seeds are 0..49 in order
        return _make_splits_for_instance(inst)

    # ------------------------------------------------------------------
    # 3. Otherwise, return splits for ALL seeds
    # ------------------------------------------------------------------
    all_splits = []
    for inst in instances:
        s = inst["seed"]
        train_dl, valid_dl, test_dl = _make_splits_for_instance(inst)
        all_splits.append({
            "seed": s,
            "train_dl": train_dl,
            "valid_dl": valid_dl,
            "test_dl": test_dl,
        })

    return all_splits

def load_shortest_path_setting(
    base_dir,
    n,
    input_dim,
    deg,
    noise,
    grid_width,
    seed=None,
    batch_size=32
):
    """
    Load ONE specific shortest-path dataset setting.
    If seed=None, returns a list of 50 wrapped datasets.
    If seed=k, returns only dataset for that seed.

    Returns:
        - If seed is None:
            list_of_dataloaders (length 50)
        - If seed is integer:
            dataloader_for_that_seed
    """

    # Build file name
    pkl_path = (
        base_dir +
        f"/shortest_path_instances_n_{n}"
        f"_input_dim_{input_dim}"
        f"_deg_{deg}"
        f"_noise_{noise}"
        f"_grid_width_{grid_width}.pkl"
    )

    with open(pkl_path, "rb") as f:
        instances = pickle.load(f)  # list of 50 dicts

    # Case 1: User wants ONE seed
    if seed is not None:
        inst = instances[seed]
        dataset = DataWrapper(inst["x"], inst["y"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Case 2: User wants ALL 50 seeds
    loader_list = []
    for inst in instances:
        dataset = DataWrapper(inst["x"], inst["y"])
        loader_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    return loader_list
