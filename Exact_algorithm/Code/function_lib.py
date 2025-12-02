
import os
import pickle
import math
import random
import shutil

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from mosek.fusion import Model, Variable, Matrix, Expr, Domain, ObjectiveSense
from tqdm import tqdm

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

def figure_size(fig_width):
    # Define the golden ratio
    golden_ratio = (1 + 5**0.5) / 2
    fig_height = fig_width / golden_ratio
    return (fig_width, fig_height)

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
    train_ratio=0.9
    ):

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
        n_test = n_samples - n_train

        x_train, y_train = X[:n_train], Y[:n_train]
        x_test,  y_test  = X[n_train:],       Y[n_train:]

        return x_train, y_train, x_test, y_test

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
        x_train, y_train, x_test, y_test = _make_splits_for_instance(inst)
        all_splits.append({
            "seed": s,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        })

    return all_splits

def compute_stats(arr):
    arr = np.asarray(arr)

    # mean
    mean = np.mean(arr)

    # sample standard deviation (ddof=1)
    std = np.std(arr, ddof=1)

    # VaR 90 (90th percentile)
    var90 = np.percentile(arr, 90)

    # CVaR 90 = mean of tail losses exceeding VaR90
    tail = arr[arr >= var90]
    cvar90 = tail.mean() if len(tail) > 0 else var90

    return mean, std, var90, cvar90


#######################################################################################################################################
##################################################### Shortest-path problem function ##################################################
#######################################################################################################################################

# Generate all feasible solution sets for shortest-path problem
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

def draw_solution(G, edges, vec, grid_width=5, cost=None):
    """
    Visualize a feasible solution vector on the grid graph
    using seaborn deep palette. Cost values appear BESIDE edges.
    """

    # seaborn colors
    deep = sns.color_palette("deep")
    path_color = deep[2]
    node_edge_color = deep[0]
    other_edge_color = "lightgray"

    # grid layout
    pos = {node: (node % grid_width, -(node // grid_width)) for node in G.nodes}

    plt.figure(figsize=(5, 5), facecolor="white")

    # nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=520,
        node_color="white",
        edgecolors=node_edge_color,
        linewidths=1.8
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight="bold",
        font_color=node_edge_color
    )

    # classify edges
    main_edges = []
    other_edges = []
    for i, e in enumerate(edges):
        if vec[i] == 1:
            main_edges.append(e)
        else:
            other_edges.append(e)

    # background edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=other_edges,
        edge_color=other_edge_color,
        width=1.2,
        arrowsize=14,
        alpha=0.7
    )

    # highlighted edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=main_edges,
        edge_color=path_color,
        width=4.0,
        arrowsize=26
    )

    # ------------- cost labels BESIDE edges -------------
    if cost is not None:
        for i, e in enumerate(edges):
            u, v = e
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # mid point
            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2

            # offset vector depends on edge orientation
            if x1 == x2:
                # vertical edge → shift horizontally
                dx, dy = 0.25, 0
            else:
                # horizontal edge → shift vertically
                dx, dy = 0, 0.15

            # chosen edge → bold, colored text
            if vec[i] == 1:
                text_color = path_color
                font_weight = "bold"
            else:
                text_color = "gray"
                font_weight = "normal"

            plt.text(
                xm + dx, ym + dy,
                f"{cost[i]:.2f}",
                fontsize=9,
                fontweight=font_weight,
                color=text_color,
                ha="center",
                va="center"
            )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

##################################################################################
################################ LST Optimization ################################
##################################################################################

class LSTECPParam:
    """
    ECP ERM with exponential cone.
    Model is built ONCE with parameters for X and pi(C),
    so we can reuse it for many datasets (x_n, c_n).

    Problem (same as your LaTeX, in compact form):

        min_{B, t_n, v_{n,z'}}  (1/(N|Z|)) * sum_n t_n
                               + (1/(N|Z|)) * sum_n sum_z pi_{n,z} (B^T x_n)^T z

        s.t.   (v_{n,z'}, eta, -(B^T x_n)^T z' - t_n) ∈ K_exp,   ∀n,z'
               sum_{z'} v_{n,z'} ≤ eta,                          ∀n

    Here:
        - Z, eta fixed
        - X, C change across runs
        - pi_{n,z} is computed from C, Z, eta and passed as a parameter.
    """

    def __init__(self, N, d_x, d_c, Z, eta: float):
        """
        Parameters
        ----------
        N : int
            Number of samples.
        d_x : int
            Dimension of x_n.
        d_c : int
            Dimension of c_n (and z).
        Z : array, shape (K, d_c)
            All feasible z ∈ Z as rows.
        eta : float
            Temperature / scaling parameter.
        """
        self.N = int(N)
        self.d_x = int(d_x)
        self.d_c = int(d_c)
        self.Z = np.asarray(Z, dtype=float)          # (K, d_c)
        self.eta = float(eta)

        self.K, d_c2 = self.Z.shape
        if d_c2 != self.d_c:
            raise ValueError("Z must have shape (K, d_c) with d_c matching provided d_c.")

        # Build the MOSEK model with Parameters X and pi
        self._build_model()

    # --------------------------------------------------------
    # INTERNAL: build MOSEK model with Parameters
    # --------------------------------------------------------
    def _build_model(self):
        M = Model("lst_ecp_param")
        self.M = M

        # Parameters
        self.Xpar = M.parameter("X", [self.N, self.d_x])      # (N, d_x)
        self.pipar = M.parameter("pi", [self.N, self.K])      # (N, K)

        # Variables
        self.B = M.variable("B", [self.d_x, self.d_c], Domain.unbounded())
        self.t = M.variable("t", self.N, Domain.unbounded())
        self.v = M.variable("v", [self.N, self.K], Domain.unbounded())

        # Expressions
        XB = Expr.mul(self.Xpar, self.B)                      # (N, d_c)
        Zt = Matrix.dense(self.Z.T)                           # (d_c, K)
        self.Phi = Expr.mul(XB, Zt)                           # (N, K)

        # u_{n,k} = -(Phi_{n,k} + t_n)
        T_mat = Expr.repeat(self.t, self.K, 1)                # (N, K)
        U_mat = Expr.neg(Expr.add(self.Phi, T_mat))           # (N, K)

        # Flatten for exponential cone
        v_flat = Expr.reshape(self.v, self.N * self.K, 1)     # (NK,1)
        U_flat = Expr.reshape(U_mat, self.N * self.K, 1)      # (NK,1)

        eta_vec = np.full((self.N * self.K, 1), self.eta, dtype=float)
        eta_mat = Matrix.dense(eta_vec)
        eta_expr = Expr.constTerm(eta_mat)                    # (NK,1) as Expression

        exp_triplets = Expr.hstack(v_flat, eta_expr, U_flat)  # (NK,3)

        # (v_{n,k}, eta, u_{n,k}) ∈ K_exp
        M.constraint("exp_cones", exp_triplets, Domain.inPExpCone())

        # sum_k v_{n,k} ≤ eta, ∀n
        sum_v_over_k = Expr.sum(self.v, 1)
        M.constraint(
            "sum_v_le_eta",
            sum_v_over_k,
            Domain.lessThan(np.full(self.N, self.eta))
        )

        # Objective
        term1 = Expr.sum(self.t)
        term2 = Expr.sum(Expr.mulElm(self.pipar, self.Phi))

        scale = 1.0 / (self.N * self.K)
        obj = Expr.mul(scale, Expr.add(term1, term2))

        M.objective("erm_ecp_param_obj", ObjectiveSense.Minimize, obj)


    # --------------------------------------------------------
    # INTERNAL: compute pi(C) from C, Z, eta
    # --------------------------------------------------------
    def _compute_pi_from_C(self, C):
        """
        Given cost matrix C ∈ R^{N × d_c} (rows: c_n^T),
        compute pi_{n,k} = softmax_z( -c_n^T z_k / eta ).

        Returns
        -------
        pi : ndarray, shape (N, K)
        """
        C = np.asarray(C, dtype=float)
        if C.shape != (self.N, self.d_c):
            raise ValueError(f"C must have shape ({self.N}, {self.d_c})")

        # Scores S_{n,k} = -c_n^T z_k / eta
        S = - (C @ self.Z.T) / self.eta      # (N, K)

        # Softmax along k
        S_max = S.max(axis=1, keepdims=True)
        expS = np.exp(S - S_max)
        denom = expS.sum(axis=1, keepdims=True)
        pi = expS / denom
        return pi

    # --------------------------------------------------------
    # PUBLIC: solve for a given dataset (X, C)
    # --------------------------------------------------------
    def solve(self, X, C, verbose: bool = False):
        """
        Solve the ECP problem for a given dataset (X, C).

        Parameters
        ----------
        X : array, shape (N, d_x)
            Feature matrix (rows = x_n^T).
        C : array, shape (N, d_c)
            Cost matrix (rows = c_n^T).
        verbose : bool
            If True, show MOSEK log.

        Returns
        -------
        result : dict with keys:
            "B"   : optimal B (d_x, d_c)
            "t"   : optimal t_n (N,)
            "v"   : optimal v_{n,z'} (N, K)
            "obj" : optimal objective value
        """
        X = np.asarray(X, dtype=float)
        if X.shape != (self.N, self.d_x):
            raise ValueError(f"X must have shape ({self.N}, {self.d_x})")

        # Set parameter X
        self.Xpar.setValue(X)

        # Compute pi from C and set parameter
        pi_val = self._compute_pi_from_C(C)
        self.pipar.setValue(pi_val)

        # Logging control
        if not verbose:
            self.M.setLogHandler(None)

        # Solve
        self.M.solve()

        # Extract solution
        B_val = np.array(self.B.level()).reshape(self.d_x, self.d_c)
        t_val = np.array(self.t.level())
        v_val = np.array(self.v.level()).reshape(self.N, self.K)
        obj_val = self.M.primalObjValue()

        return {
            "B": B_val,
            "t": t_val,
            "v": v_val,
            "obj": obj_val,
        }
    
    def evaluate_regret(self, X_test, C_test, B=None, return_per_sample=False):
        """
        Evaluate the prediction regret on a test dataset (X_test, C_test) using 
        vectorized operations.

        Parameters
        ----------
        X_test : array, shape (N_test, d_x)
        C_test : array, shape (N_test, d_c)
        B : array or None, optional
        return_per_sample : bool, optional

        Returns
        -------
        result : dict
        """
        # 1. Setup and Validations
        X_test = np.asarray(X_test, dtype=float)
        C_test = np.asarray(C_test, dtype=float)
        
        N_test, d_x_test = X_test.shape
        N_ctest, d_c_test = C_test.shape

        if d_x_test != self.d_x:
            raise ValueError(f"X_test shape mismatch: expected dim {self.d_x}, got {d_x_test}")
        if d_c_test != self.d_c:
            raise ValueError(f"C_test shape mismatch: expected dim {self.d_c}, got {d_c_test}")
        if N_test != N_ctest:
            raise ValueError("X_test and C_test must have the same number of rows.")

        # Get B (from parameter or current model)
        if B is None:
            B = np.array(self.B.level()).reshape(self.d_x, self.d_c)
        else:
            B = np.asarray(B, dtype=float).reshape(self.d_x, self.d_c)

        # ---------------------------------------------------------
        # 2. Vectorized Calculation
        # ---------------------------------------------------------

        # A. Predict costs for ALL samples at once
        # Shape: (N, d_x) @ (d_x, d_c) -> (N, d_c)
        C_hat = X_test @ B

        # B. Calculate objective values for ALL feasible z on predicted costs
        # Z is (K, d_c). We want (N, K).
        # Shape: (N, d_c) @ (d_c, K) -> (N, K)
        # obj_pred[i, k] is the predicted cost of solution k for sample i
        obj_pred = C_hat @ self.Z.T

        # C. Find the index of the predicted optimal solution for each sample
        # Shape: (N,)
        idx_hat = np.argmin(obj_pred, axis=1)

        # D. Calculate objective values for ALL feasible z on TRUE costs
        # Shape: (N, d_c) @ (d_c, K) -> (N, K)
        obj_true = C_test @ self.Z.T

        # E. Extract the true costs
        # 1. The true cost of the decision we PREDICTED (using advanced indexing)
        #    obj_true[row_indices, col_indices]
        chosen_costs = obj_true[np.arange(N_test), idx_hat]

        # 2. The true cost of the OPTIMAL decision (Oracle)
        optimal_costs = obj_true.min(axis=1)

        # F. Compute Regret
        regrets = chosen_costs - optimal_costs

        # ---------------------------------------------------------
        # 3. Result formatting
        # ---------------------------------------------------------
        avg_regret = float(regrets.mean())
        
        result = {"avg_regret": avg_regret}
        if return_per_sample:
            result["regret_per_sample"] = regrets

        return result