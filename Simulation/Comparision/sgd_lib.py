
import os
import pickle
import torch 
import math
import random
import shutil

import numpy as np
import pandas as pd
import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path

from pytorch_lightning import loggers as pl_loggers

#######################################################################################################################################
######################################################### Common function #############################################################
#######################################################################################################################################

def change_to_py_file_dir(Indicator=False):
    # Static variables to store the base path and parent path
    if not hasattr(change_to_py_file_dir, 'BASE_PATH'):
        # Set the base path to the current working directory
        change_to_py_file_dir.BASE_PATH = os.getcwd()
        change_to_py_file_dir.PARENT_PATH = os.path.dirname(change_to_py_file_dir.BASE_PATH)
    if not Indicator:
        # Change to the parent directory
        os.chdir(change_to_py_file_dir.PARENT_PATH) 
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

def figure_size(fig_width):
    # Define the golden ratio
    golden_ratio = (1 + 5**0.5) / 2
    fig_height = fig_width / golden_ratio
    return (fig_width, fig_height)

#######################################################################################################################################
########################################################### Dataset function ##########################################################
#######################################################################################################################################

def sgd_load_shortest_path_setting_with_splits(
    base_dir,
    n,
    input_dim,
    deg,
    noise,
    grid_width,
    seed=None,
    batch_size=32,
    train_ratio=0.9,
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

    # def _make_splits_for_instance(inst):
    #     """Create train/valid/test DataLoaders for a single seed-instance."""
    #     X = inst["x"]
    #     Y = inst["y"]
    #     n_samples = len(X)

    #     n_train = int(n_samples * train_ratio)
    #     n_valid = int(n_samples * valid_ratio)
    #     n_test = n_samples - n_train - n_valid

    #     x_train, y_train = X[:n_train], Y[:n_train]
    #     x_valid, y_valid = X[n_train:n_train + n_valid], Y[n_train:n_train + n_valid]
    #     x_test,  y_test  = X[n_train + n_valid:],       Y[n_train + n_valid:]

    #     train_ds = DataWrapper(x_train, y_train)
    #     valid_ds = DataWrapper(x_valid, y_valid)
    #     test_ds  = DataWrapper(x_test,  y_test)

    #     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    #     valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    #     test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    #     return y_train, x_test, y_test, train_dl, valid_dl, test_dl
    def _make_splits_for_instance(inst):
        """Create train/valid/test DataLoaders for a single seed-instance."""
        X = inst["x"]
        Y = inst["y"]
        n_samples = len(X)

        n_train = int(n_samples * train_ratio)
        n_valid = int(n_samples * valid_ratio)
        n_test = n_samples - n_train

        x_train, y_train = X[:n_train], Y[:n_train]
        x_valid, y_valid = X[n_train-n_valid:n_train], Y[n_train-n_valid:n_train]
        x_test,  y_test  = X[n_train:],       Y[n_train:]

        train_ds = DataWrapper(x_train, y_train)
        valid_ds = DataWrapper(x_valid, y_valid)
        test_ds  = DataWrapper(x_test,  y_test)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        return y_train, x_test, y_test, train_dl, valid_dl, test_dl 

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

#######################################################################################################################################
########################################################### Dataset function ##########################################################
#######################################################################################################################################

class ShortestPathSolver:
    def __init__(self, G):
        """
        G: networkx.DiGraph
        Assumes source is node 0 and sink is the last node.
        """
        self.G = G

        # Precompute incidence matrix (sparse -> numpy array)
        A_sparse = nx.incidence_matrix(self.G, oriented=True)
        self.A = A_sparse.toarray()  # (n_nodes, n_edges)

        # Flow balance vector b
        self.b = np.zeros(self.A.shape[0], dtype=float)
        self.b[0] = -1.0              # source
        self.b[-1] = 1.0              # sink

        # Cache models for (relaxation=False/True)
        self._models = {}

    def _get_model(self, relaxation: bool):
        """
        Create (or retrieve cached) Gurobi model for given relaxation flag.
        """
        key = bool(relaxation)
        if key in self._models:
            return self._models[key]

        model = gp.Model()
        model.setParam('OutputFlag', 0)

        vtype = gp.GRB.CONTINUOUS if relaxation else gp.GRB.BINARY

        # Decision variables: one per edge
        x = model.addMVar(shape=self.A.shape[1], name="x", vtype=vtype)

        # Flow conservation constraints: A x = b
        model.addMConstr(self.A, x, sense='=', b=self.b, name="flow")

        # We will set the objective later for each y
        model.update()
        self._models[key] = (model, x)
        return model, x

    def solve(self, y, relaxation: bool = False):
        """
        Solve shortest path for a given cost vector y.

        y: 1D numpy array or list of length = num_edges
        """
        # Ensure numpy array
        y = np.asarray(y, dtype=float)

        model, x = self._get_model(relaxation)

        # Set objective: minimize y^T x
        x.setAttr(gp.GRB.Attr.Obj, y)
        model.ModelSense = gp.GRB.MINIMIZE
        model.update()
        model.optimize()

        return x.X.copy()  # return a numpy array of variable values


def define_graph(grid_width, showflag=False):
    V = range(grid_width**2)
    E = []

    # build edges (right + down)
    for i in V:
        if (i + 1) % grid_width != 0:
            E.append((i, i + 1))
        if i + grid_width < grid_width**2:
            E.append((i, i + grid_width))

    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    if showflag:
        # grid positions only needed if we actually plot
        pos = {node: (node % grid_width, -(node // grid_width)) for node in V}

        plt.figure(figsize=(6, 6))
        nx.draw_networkx(
            G,
            pos=pos,
            node_size=300,
            arrows=True,
            with_labels=True,
            font_size=8,
            arrowsize=20
        )
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    return G

def batch_solve(solver, y, relaxation: bool = False) -> torch.Tensor:
    """
    Solve shortest path for each row in y using the same solver.

    y: torch.Tensor or numpy array of shape (batch, num_edges) or (num_edges,)
    returns: torch.Tensor of shape (batch, num_edges) on SAME DEVICE as input y (if tensor)
    """
    # Detect if y is a tensor and remember its device
    if isinstance(y, torch.Tensor):
        device = y.device
        y_np = y.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        y_np = np.asarray(y)

    # Ensure 2D
    if y_np.ndim == 1:
        y_np = y_np[None, :]

    sols = [solver.solve(y_np[i], relaxation=relaxation) for i in range(y_np.shape[0])]
    sols_np = np.vstack(sols)  # (batch, num_edges)

    # Return tensor on the original device
    return torch.from_numpy(sols_np).float().to(device)


def regret_fn(solver: ShortestPathSolver, y_hat: torch.Tensor, y: torch.Tensor, minimize: bool = True) -> torch.Tensor:
    """
    Compute regret:
        E[ (x_hat - x_opt)^T y ]    (for minimization problems)

    y_hat: predicted costs, shape (batch, num_edges)
    y:     true costs, shape (batch, num_edges)
    """
    mm = 1.0 if minimize else -1.0

    # Solve for predicted and true costs
    with torch.no_grad():
        sol_hat = batch_solve(solver, y_hat)        # (batch, num_edges)
        sol_true = batch_solve(solver, y)           # (batch, num_edges)

    # Move to same device as y for the arithmetic
    sol_hat = sol_hat.to(y.device)
    sol_true = sol_true.to(y.device)

    # Regret per sample: (x_hat - x_opt) · y
    # regret_per_sample = mm * ((sol_hat - sol_true) * y).sum(dim=1)
    num = mm * ((sol_hat - sol_true) * y).sum(dim=1)
    denom = torch.clamp((sol_true * y).sum(dim=1), min=1e-9)
    regret_per_sample = num / denom

    return regret_per_sample.mean()


def create_csv_logger(log_dir, experiment_name, seed):
    """
    Create a CSVLogger, but if the folder for *this seed* exists, delete only that folder.
    Does NOT delete parent folders.

    log structure:
        log_dir / experiment_name / version_{seed} / metrics.csv
    """

    # Build full path to version folder
    if experiment_name in ["", None]:
        version_dir = os.path.join(log_dir, f"version_{seed}")
    else:
        version_dir = os.path.join(log_dir, experiment_name, f"version_{seed}")

    # 🔥 Remove ONLY this version folder, not parents
    if os.path.exists(version_dir):
        # print(f"[INFO] Removing existing logger folder: {version_dir}")
        shutil.rmtree(version_dir, ignore_errors=True)

    # Create logger normally
    logger = CSVLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=seed
    )

    return logger

#######################################################################################################################################
############################################################ Model function ###########################################################
#######################################################################################################################################
    
class MSEModel(pl.LightningModule):

    def __init__(self, net, solver, lr: float = 1e-2, max_epochs: int = 30):
        """
        Two-stage model:
        1) Predict edge costs via MSE regression
        2) Evaluate via regret using a shortest-path solver.

        Args:
            net:    PyTorch nn.Module mapping x -> y_hat (edge costs)
            solver: ShortestPathSolver instance
            lr:     learning rate
            max_epochs: number of training epochs (for logging/metadata)
        """
        super().__init__()
        self.net = net
        self.solver = solver
        self.lr = lr
        self.max_epochs = max_epochs

        # define loss once to avoid re-creating every step
        self.criterion = nn.MSELoss(reduction='mean')

        # log hparams (ignore big objects like net & solver if desired)
        self.save_hyperparameters(ignore=["net", "solver"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch                      # y shape: (B, E)
        y_hat = self(x)                   # (B, E) – no squeeze to avoid shape issues for B=1
        # Ensure y_hat and y are at least 2D: (B, E)
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(0)
            y = y.unsqueeze(0)

        # Per-element squared error: shape (B, E)
        per_element = self.criterion(y_hat, y)  # reduction='none'

        # ✅ Mean over edges, for each instance: (B,)
        per_instance_mse = per_element.mean(dim=-1)

        # ✅ Batch mean (scalar) used for optimization
        loss = per_instance_mse.mean()

        # Log batch mean train loss
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)

        mse_loss = self.criterion(y_hat, y)
        # regret_fn internally uses the solver; this is expensive, so we only do it in val/test
        regret_loss = regret_fn(self.solver, y_hat, y)

        self.log("val_mse", mse_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_regret", regret_loss, prog_bar=False, on_step=False, on_epoch=True)

        return {"val_mse": mse_loss, "val_regret": regret_loss}

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)

        mse_loss = self.criterion(y_hat, y)
        regret_loss = regret_fn(self.solver, y_hat, y)

        self.log("test_mse", mse_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_regret", regret_loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"test_mse": mse_loss, "test_regret": regret_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # simple optional scheduler (not required, but can help)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_regret",  # reduce LR when val_regret plateaus
            },
        }

def SPOLoss(solver, minimize=True):
    mm = 1 if minimize else -1
    class SPOLoss_cls(torch.autograd.Function):

        @staticmethod
        def forward(ctx, y_hat, y):
            sol_y = batch_solve(solver, y, relaxation=False)
            y_spo = 2*y_hat - y
            sol_spo = batch_solve(solver, y_spo, relaxation=False)

            # SPO+:  x(2ŷ−y)^T y − x(y)^T y
            loss = mm * (sol_spo * y).sum() - mm * (sol_y * y).sum()

            ctx.save_for_backward(y_hat, y, sol_y, sol_spo)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y_hat, y, sol_y, sol_spo = ctx.saved_tensors
            mm_grad = 1.0 if minimize else -1.0

            # # true SPO+ gradient
            # grad_y_hat = 2 * mm_grad * (sol_spo - sol_y)

            # chain rule
            return grad_output *(sol_y - sol_spo)*mm_grad, None
            
    return SPOLoss_cls.apply

class SPOModel(MSEModel):
    def __init__(self, net, solver, lr=1e-1, max_epochs=30):
        """
        Implementaion of SPO+ loss subclass of twostage model
 
        """
        super().__init__(net, solver, lr, max_epochs)

    def training_step(self, batch, batch_idx):
        x,y= batch
        y_hat =  self(x).squeeze()
        loss = SPOLoss(self.solver, minimize=True)(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

def generate_all_feasible_solutions(grid_width=5):
    """
    Build a grid_width x grid_width directed grid (right & down edges),
    and enumerate all feasible s->t paths (from NW to SE corner).

    Returns
    -------
    feasible_solutions : list of np.ndarray
        Each element is a 0–1 vector of length |E| indicating which edges are used.
    edges : list of tuple
        List of directed edges in fixed order consistent with the vectors.
    G : networkx.DiGraph
        The underlying directed grid graph.
    """
    # 1) Build graph
    V = range(grid_width**2)
    E = []
    for i in V:
        # edge to the right
        if (i + 1) % grid_width != 0:
            E.append((i, i + 1))
        # edge downward
        if i + grid_width < grid_width**2:
            E.append((i, i + grid_width))
    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    # 2) Prepare edge indexing
    edges = list(G.edges())
    edge_index = {e: idx for idx, e in enumerate(edges)}

    source = 0
    target = max(G.nodes)

    # 3) DFS to enumerate all s->t paths and convert to 0–1 vectors
    feasible_solutions = []

    def dfs(node, path_edges):
        if node == target:
            vec = np.zeros(len(edges), dtype=int)
            for e in path_edges:
                vec[edge_index[e]] = 1
            feasible_solutions.append(vec)
            return
        for nbr in G.successors(node):
            dfs(nbr, path_edges + [(node, nbr)])

    dfs(source, [])

    return feasible_solutions, edges, G

def listnet_loss(y_hat, y_true, sols, eta, minimize=True):
    """
    Listwise loss (cross-entropy between softmax over true and predicted costs).

    y_hat : (batch, num_edges)  predicted costs  ( \hat c )
    y_true: (batch, num_edges)  true costs       ( c )
    sols  : (|Z|, num_edges)    solution pool (0/1 incidence vectors)
    eta   : temperature η
    """
    mm = 1.0 if minimize else -1.0

    # Ensure sols is on the same device as y_hat (CUDA or CPU)
    sols = sols.to(y_hat.device)

    # costs_* : (batch, |Z|), f(c,z) = (mm * c)ᵀ z
    costs_hat  = (mm * y_hat)  @ sols.t()
    costs_true = (mm * y_true) @ sols.t()

    # logits = - f(c, z) / η
    logits_hat  = -costs_hat  / eta
    logits_true = -costs_true / eta

    # p_true(z) = softmax(-f(c,z)/η)
    p_true    = F.softmax(logits_true, dim=1)        # (batch, |Z|)
    log_p_hat = F.log_softmax(logits_hat, dim=1)     # (batch, |Z|)

    # ℓ_n = - (1/|Z|) ∑_z p_true(z) log p_hat(z)
    loss_per_sample = -(p_true * log_p_hat).mean(dim=1)  # mean over z

    # final scalar loss (mean over batch)
    return loss_per_sample.mean()

def dio_loss(y_hat, y_true, sols, eta, minimize=True):
    """
    DIO loss:
        ℓ_DIO(c, ĉ; Z) = η log ∑_z exp( ( f(ĉ, z*(c)) - f(ĉ, z) ) / η )

    Args
    ----
    y_hat : (batch, num_edges)  predicted costs  (ĉ)
    y_true: (batch, num_edges)  true costs       (c)
    sols  : (|Z|, num_edges)    solution pool (0/1 incidence vectors)
    eta   : temperature η
    minimize : True if we solve a min-cost problem, False for max-cost
    """
    mm = 1.0 if minimize else -1.0

    # Ensure sols is on the same device as y_hat
    sols = sols.to(y_hat.device)

    # f(c, z) = (mm * c)^T z for all z ∈ Z
    # Shapes: (batch, |Z|)
    f_true = (mm * y_true) @ sols.t()   # used only to find z*(c)
    f_hat  = (mm * y_hat)  @ sols.t()   # used in the loss

    # z*(c): index of optimal solution under true costs
    if minimize:
        best_idx = f_true.argmin(dim=1)   # (batch,)
    else:
        best_idx = f_true.argmax(dim=1)

    batch_size = y_hat.size(0)
    # f(ĉ, z*(c)) for each sample
    f_hat_star = f_hat[torch.arange(batch_size, device=y_hat.device), best_idx]  # (batch,)

    # margins: ( f(ĉ, z*(c)) - f(ĉ, z) ) / η  → shape (batch, |Z|)
    margins = (f_hat_star.unsqueeze(1) - f_hat) / eta

    # ℓ_n = η * logsumexp_z margins_n(z)
    loss_per_sample = eta * torch.logsumexp(margins, dim=1)   # (batch,)

    # Final scalar loss (average over batch)
    return loss_per_sample.mean()

def pairwise_loss(y_hat, y_true, sols, eta, minimize=True):
    """
    Pairwise loss:

        L(c, ĉ; Z) = η log ∑_{z,z'∈Z} 1{ f(c,z) ≤ f(c,z') } *
                                 exp( ( f(ĉ,z) - f(ĉ,z') ) / η )

    Args
    ----
    y_hat : (batch, num_edges)  predicted costs  (ĉ)
    y_true: (batch, num_edges)  true costs       (c)
    sols  : (|Z|, num_edges)    solution pool, 0/1 incidence vectors
    eta   : temperature η
    minimize : True if f is a cost (minimization problem), False for max
    """
    mm = 1.0 if minimize else -1.0

    # Ensure sols on same device
    sols = sols.to(y_hat.device)

    # f(c, z) = (mm * c)^T z and f(ĉ, z) = (mm * ĉ)^T z
    # Shapes: (batch, |Z|)
    f_true = (mm * y_true) @ sols.t()
    f_hat  = (mm * y_hat)  @ sols.t()

    B, K = f_true.shape  # batch size, |Z|

    # Build pairwise matrices: f_true(z), f_true(z')
    # Shapes: (B, K, K)
    f_true_i = f_true.unsqueeze(2)  # (B, K, 1)
    f_true_j = f_true.unsqueeze(1)  # (B, 1, K)

    # Indicator 1{ f_true(z) ≤ f_true(z') }
    mask = (f_true_i <= f_true_j)   # bool, shape (B, K, K)

    # Pairwise margins for predicted costs: f_hat(z) - f_hat(z')
    f_hat_i = f_hat.unsqueeze(2)    # (B, K, 1)
    f_hat_j = f_hat.unsqueeze(1)    # (B, 1, K)
    margins = (f_hat_i - f_hat_j) / eta  # (B, K, K)

    # Set margins for invalid pairs to a very negative number so their exp() ≈ 0
    neg_large = torch.finfo(margins.dtype).min  # ~ -1e38 for float32
    margins_masked = torch.where(mask, margins, torch.full_like(margins, neg_large))

    # Flatten pairs (z,z') into a single axis per sample: shape (B, K*K)
    margins_flat = margins_masked.view(B, -1)

    # L_n = η * logsumexp over all valid pairs (z,z')
    loss_per_sample = eta * torch.logsumexp(margins_flat, dim=1)  # (B,)

    # Final scalar loss: average over batch
    return loss_per_sample.mean()


def likelihood_ranking_loss(y_hat, y_true, sols, eta, minimize=True):
    """
    Likelihood-style ranking loss:

        ℓ(c, ĉ; Z) =
            - ∑_{i=1}^{|Z|} f(ĉ, z_{φ(i)})
            + ∑_{i=1}^{|Z|} η log ∑_{k=i}^{|Z|} exp( f(ĉ, z_{φ(k)}) / η ),

    where φ is the permutation that sorts f(c, z) in non-decreasing order.

    Args
    ----
    y_hat : (batch, num_edges)  predicted costs  (ĉ)
    y_true: (batch, num_edges)  true costs       (c)
    sols  : (|Z|, num_edges)    solution pool (0/1 incidence vectors)
    eta   : temperature η
    minimize : True if it's a min-cost problem, False if max-profit
    """
    mm = 1.0 if minimize else -1.0

    # Move sols to correct device
    sols = sols.to(y_hat.device)

    # f(c, z) and f(ĉ, z) for all z ∈ Z
    # Shapes: (batch, |Z|)
    f_true = (mm * y_true) @ sols.t()
    f_hat  = (mm * y_hat)  @ sols.t()

    B, K = f_true.shape  # batch size, |Z|

    # φ: argsort of true costs (ascending)
    # φ[n, i] = index of i-th best z under true costs for sample n
    _, phi = torch.sort(f_true, dim=1, descending=False)  # (B, K)

    # Reorder predicted scores according to φ
    # f_hat_sorted[n, i] = f_hat(n, z_{φ(i)})
    f_hat_sorted = torch.gather(f_hat, dim=1, index=phi)  # (B, K)

    # First term: -∑_i f(ĉ, z_{φ(i)})
    term1 = f_hat_sorted.sum(dim=1)  # (B,)

    # Second term: ∑_i η log ∑_{k=i}^{K} exp( f_hat_sorted[k] / η )
    logits = -f_hat_sorted / eta  # (B, K)
    term2 = torch.zeros(B, device=y_hat.device)

    for i in range(K):
        # log ∑_{k=i}^{K-1} exp( logits[:, k] )
        lse_i = torch.logsumexp(logits[:, i:], dim=1)  # (B,)
        term2 = term2 + eta * lse_i

    loss_per_sample = term1 + term2  # (B,)

    # Final scalar loss (mean over batch)
    return loss_per_sample.mean()

class LSTModel(MSEModel):

    def __init__(self, loss_fn, sols, net, solver, eta, lr=1e-1, max_epochs=30):
        """
        Listwise model using a solution pool.

        Args:
            loss_fn: loss function (y_hat, y_true, sols, eta, minimize=True) -> scalar
            sols:    (|Z|, num_edges) solution pool (tensor, can be on CPU)
            net:     neural network x -> edge costs
            solver:  shortest-path solver (used in parent MSEModel)
            eta:     temperature η
            lr:      learning rate
            max_epochs: maximum number of epochs
        """
        super().__init__(net, solver, lr, max_epochs)

        self.loss_fn = loss_fn

        # ---- Normalize sols to a 2D numpy array on CPU ----
        if isinstance(sols, torch.Tensor):
            sols_np = sols.detach().cpu().numpy()
        elif isinstance(sols, np.ndarray):
            sols_np = sols
        else:
            # assume list-like
            sols_np = np.array(sols, dtype=np.float32)

        sols_np = np.unique(sols_np, axis=0)   # delete repeated elements
        sols_t  = torch.from_numpy(sols_np).float()

        # Register as buffer so it moves with .to(device) and under Lightning accelerators
        self.register_buffer("sols", sols_t)

        self.eta = eta

        self.validation_step_mse = []
        self.validation_step_regret = []

        # Store simple hyperparams for checkpointing
        self.save_hyperparameters({"lr": lr, "tau": eta})

    def training_step(self, batch, batch_idx):
        x, y = batch               # Lightning will already move x,y to the correct device
        y_hat = self(x)

        # If net outputs (batch, num_edges, 1), squeeze only the last dim safely
        if y_hat.dim() == 3 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)

        # self.sols is a buffer, so it's already on the same device as the model
        loss = self.loss_fn(y_hat, y, self.sols, self.eta, minimize=True)

        self.log(
            "train_loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    
MODEL_DICT = {
    "mse": MSEModel,
    "spo": SPOModel,
    "lst": LSTModel
}

def get_model_class(model_type: str):
    model_type = model_type.lower()
    if model_type not in MODEL_DICT:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_DICT[model_type]

#######################################################################################################################################
############################################################ Figure function ##########################################################
#######################################################################################################################################

def plot_training_curves(log_dir, experiment_name, version):
    """
    Plots two parallel figures:
      - Left: Train Loss (log-scale)
      - Right: Validation Regret (log-scale)
    """

    # Build correct path to metrics file
    if experiment_name in ["", None]:
        metrics_path = os.path.join(log_dir, f"version_{version}", "metrics.csv")
    else:
        metrics_path = os.path.join(log_dir, experiment_name, f"version_{version}", "metrics.csv")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)

    # ---- extract train_loss over epochs ----
    if "train_loss" in df.columns:
        m_train = df["train_loss"].notna()
        epochs_train = df.loc[m_train, "epoch"]
        train_loss = df.loc[m_train, "train_loss"]
    else:
        epochs_train, train_loss = [], []

    # ---- extract val_regret over epochs ----
    if "val_regret" in df.columns:
        m_val = df["val_regret"].notna()
        epochs_val = df.loc[m_val, "epoch"]
        val_regret = df.loc[m_val, "val_regret"]
    else:
        epochs_val, val_regret = [], []

    # ---- Create side-by-side subplots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    title_name = experiment_name if experiment_name not in ["", None] else "Experiment"

    # --- Left: Train Loss ---
    if len(train_loss) > 0:
        axes[0].plot(epochs_train, train_loss, linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Train Loss (MSE)", fontsize=12)
    axes[0].set_title(f"{title_name} — Train Loss", fontsize=14)
    axes[0].set_yscale("log")
    axes[0].grid(True)

    # --- Right: Validation Regret ---
    if len(val_regret) > 0:
        axes[1].plot(epochs_val, val_regret, linewidth=2, color='orange')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation Regret", fontsize=12)
    axes[1].set_title(f"{title_name} — Validation Regret", fontsize=14)
    axes[1].set_yscale("log")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
