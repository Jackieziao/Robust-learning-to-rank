
import os
import pickle
import math
import random
import shutil
import gc
import psutil
import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import gurobipy as gp
import matplotlib.ticker as mtick

from pathlib import Path
from mosek.fusion import *
from mosek.fusion import Model, Variable, Matrix, Expr, Domain, ObjectiveSense
from tqdm import tqdm
from gurobipy import GRB
from unittest import case
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

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
        if verbose:
            self.M.setLogHandler(sys.stdout)
        else:
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
        # regrets = chosen_costs - optimal_costs
        regrets = (chosen_costs - optimal_costs)/optimal_costs

        # ---------------------------------------------------------
        # 3. Result formatting
        # ---------------------------------------------------------
        avg_regret = float(regrets.mean())
        
        result = {"avg_regret": avg_regret}
        if return_per_sample:
            result["regret_per_sample"] = regrets

        return result

###################################################################################
################################ SPO+ Optimization ################################
###################################################################################

def build_incidence_matrix_and_b(edges, G):
    """
    Build node–edge incidence matrix A and vector b:

        min c^T z
        s.t. A^T z = b, z >= 0

    A[v, e] = +1 if e leaves v
            = -1 if e enters v
            =  0 otherwise
    b_s = 1, b_t = -1, others 0.
    """
    nodes = list(G.nodes())
    node_index = {v: i for i, v in enumerate(nodes)}

    m = len(edges)   # number of edges
    n = len(nodes)   # number of nodes

    A = np.zeros((n, m), dtype=float)
    for e_idx, (u, v) in enumerate(edges):
        A[node_index[u], e_idx] = +1.0  # leaves u
        A[node_index[v], e_idx] = -1.0  # enters v

    s = min(nodes)
    t = max(nodes)
    b = np.zeros(n, dtype=float)
    b[node_index[s]] = +1.0
    b[node_index[t]] = -1.0

    return A, b

def compute_z_star_single(c, Z):
    """
    Compute z*(c) among enumerated feasible paths Z.
    Z : (K, m) matrix of 0/1 edges for each path.
    c : (m,) cost vector
    """
    costs = Z @ c
    k = np.argmin(costs)
    return Z[k]

def compute_z_star_batch(C, Z):
    """
    C : (N, m)
    Z : (K, m)
    returns Zstar : (N, m)
    """
    N = C.shape[0]
    Zstar = np.zeros_like(C)

    for i in range(N):
        costs = Z @ C[i]
        k = np.argmin(costs)
        Zstar[i] = Z[k]

    return Zstar

class SPOShortestPathParam:
    r"""
    Parametric SPO+ERM model for the shortest-path-style dual problem:

        Given:
            - A ∈ R^{d_c × d_p}
            - b ∈ R^{d_p}
            - dataset {x_n, c_n, z^*(c_n)}_{n=1}^N

        Solve:

            min_{B, p_n}  (1/N) ∑_{n=1}^N [
                 b^T p_n
               + 2 (B^T x_n)^T z^*(c_n)
               - c_n^T z^*(c_n)
            ]

            s.t.  A p_n ≥ c_n - 2 B^T x_n,   ∀n
                  p_n ∈ R^{d_p}
                  B   ∈ R^{d_x × d_c}

        We build the MOSEK model ONCE, with Parameters for X, C, Zstar,
        so we can reuse it for many datasets.
    """

    def __init__(self, N, d_x, d_c, d_p, A, b):
        """
        Parameters
        ----------
        N : int
            Number of samples.
        d_x : int
            Dimension of x_n.
        d_c : int
            Dimension of c_n and z^*(c_n).
        d_p : int
            Dimension of p_n.
        A : array-like, shape (d_c, d_p)
            Matrix in constraints A p_n ≥ c_n - 2 B^T x_n.
        b : array-like, shape (d_p,)
            Vector in the linear term b^T p_n.
        """
        self.N = int(N)
        self.d_x = int(d_x)
        self.d_c = int(d_c)
        self.d_p = int(d_p)

        # Check and store A
        A = np.asarray(A, dtype=float)
        if A.shape != (self.d_c, self.d_p):
            raise ValueError(f"A must have shape ({self.d_c}, {self.d_p})")
        self.A = A

        # Check and store b
        b = np.asarray(b, dtype=float).reshape(-1)
        if b.shape[0] != self.d_p:
            raise ValueError(f"b must have length d_p = {self.d_p}")
        self.b = b

        # Build MOSEK model with Parameters X, C, Zstar
        self._build_model()

    # ------------------------------------------------------------------
    # INTERNAL: build MOSEK model with Parameters
    # ------------------------------------------------------------------
    def _build_model(self):
        M = Model("spo_sp_param")
        self.M = M

        # ---------------- Parameters ----------------
        # X: (N, d_x)
        self.Xpar = M.parameter("X", [self.N, self.d_x])
        # C: (N, d_c)
        self.Cpar = M.parameter("C", [self.N, self.d_c])
        # Zstar: (N, d_c) where each row is z^*(c_n)^T
        self.Zstarpar = M.parameter("Zstar", [self.N, self.d_c])

        # ---------------- Variables -----------------
        # B: (d_x, d_c)
        self.B = M.variable("B", [self.d_x, self.d_c], Domain.unbounded())
        # p_n: (N, d_p)
        self.P = M.variable("p", [self.N, self.d_p], Domain.unbounded())

        # ---------------- Expressions ----------------
        # XB = X B  (N × d_x) @ (d_x × d_c) -> (N × d_c)
        # Each row is x_n^T B = (B^T x_n)^T.
        self.XB = Expr.mul(self.Xpar, self.B)  # shape (N, d_c)

        # For constraints: A p_n ≥ c_n - 2 B^T x_n
        # Using P A^T: (N × d_p) @ (d_p × d_c) -> (N × d_c)
        A_T = Matrix.dense(self.A.T)
        AP = Expr.mul(self.P, A_T)  # shape (N, d_c)

        # Constraint: A p_n - (c_n - 2 B^T x_n) ≥ 0
        # => AP - C + 2 XB ≥ 0
        dual_residual = Expr.add(
            Expr.sub(AP, self.Cpar),          # AP - C
            Expr.mul(2.0, self.XB)           # + 2 XB
        )
        M.constraint("dual_feas", dual_residual, Domain.greaterThan(0.0))

        # ---------------- Objective ------------------
        # term1 = ∑_n b^T p_n
        # replicate b across rows
        b_row = self.b.reshape(1, self.d_p)           # (1, d_p)
        b_mat = np.tile(b_row, (self.N, 1))           # (N, d_p)
        b_mat_dense = Matrix.dense(b_mat)
        term1 = Expr.sum(Expr.mulElm(self.P, b_mat_dense))

        # term2 = ∑_n (B^T x_n)^T z^*(c_n) = ∑_n ∑_k (XB_{n,k} * Zstar_{n,k})
        XB_times_Z = Expr.mulElm(self.XB, self.Zstarpar)  # (N, d_c)
        term2_per_n = Expr.sum(XB_times_Z, 1)             # (N,)
        term2 = Expr.sum(term2_per_n)

        # term3 = ∑_n c_n^T z^*(c_n) = ∑_n ∑_k C_{n,k} * Zstar_{n,k}
        CZ = Expr.mulElm(self.Cpar, self.Zstarpar)        # (N, d_c)
        term3_per_n = Expr.sum(CZ, 1)                     # (N,)
        term3 = Expr.sum(term3_per_n)

        # Full objective: (1/N) [ term1 + 2 term2 - term3 ]
        obj_expr = Expr.sub(
            Expr.add(term1, Expr.mul(2.0, term2)),
            term3
        )
        obj = Expr.mul(1.0 / self.N, obj_expr)

        M.objective("spo_sp_obj", ObjectiveSense.Minimize, obj)

    # ------------------------------------------------------------------
    # PUBLIC: solve for a given dataset (X, C, Zstar)
    # ------------------------------------------------------------------
    def solve(self, X, C, Zstar, verbose: bool = False):
        """
        Solve the SPO+SP problem for a given dataset (X, C, Zstar).

        Parameters
        ----------
        X : array, shape (N, d_x)
            Feature matrix (rows = x_n^T).
        C : array, shape (N, d_c)
            Cost matrix (rows = c_n^T).
        Zstar : array, shape (N, d_c)
            Each row is z^*(c_n)^T, i.e., the optimal shortest-path solution
            (or more generally the LP solution) for cost c_n.
        verbose : bool
            If True, show MOSEK log.

        Returns
        -------
        result : dict with keys:
            "B"   : optimal B, shape (d_x, d_c)
            "P"   : optimal p_n, shape (N, d_p)
            "obj" : optimal objective value
        """
        X = np.asarray(X, dtype=float)
        C = np.asarray(C, dtype=float)
        Zstar = np.asarray(Zstar, dtype=float)

        # Shape checks
        if X.shape != (self.N, self.d_x):
            raise ValueError(f"X must have shape ({self.N}, {self.d_x})")
        if C.shape != (self.N, self.d_c):
            raise ValueError(f"C must have shape ({self.N}, {self.d_c})")
        if Zstar.shape != (self.N, self.d_c):
            raise ValueError(f"Zstar must have shape ({self.N}, {self.d_c})")

        # Set parameter values
        self.Xpar.setValue(X)
        self.Cpar.setValue(C)
        self.Zstarpar.setValue(Zstar)

        # Logging
        if verbose:
            self.M.setLogHandler(sys.stdout)
        else:
            self.M.setLogHandler(None)

        # Solve
        self.M.solve()

        # Extract solution
        B_val = np.array(self.B.level()).reshape(self.d_x, self.d_c)
        P_val = np.array(self.P.level()).reshape(self.N, self.d_p)
        obj_val = self.M.primalObjValue()

        return {
            "B": B_val,
            "P": P_val,
            "obj": obj_val,
        }
    
    # ------------------------------------------------------------------
    # PUBLIC: evaluate regret on a test set
    # ------------------------------------------------------------------
    def evaluate_regret(self, X_test, C_test, Z, B=None, return_per_sample=False):
        """
        Evaluate prediction regret on a test dataset (X_test, C_test),
        given a discrete feasible set Z of s-t paths (0/1 edge indicators).

        Parameters
        ----------
        X_test : array, shape (N_test, d_x)
        C_test : array, shape (N_test, d_c)
        Z      : array, shape (K, d_c)
            All feasible solutions (paths) as rows.
        B : array or None, optional
            If None, use the current model solution self.B.
            Otherwise, use this B matrix (d_x, d_c) for evaluation.
        return_per_sample : bool, optional
            If True, also return regret per sample.

        Returns
        -------
        result : dict with keys:
            "avg_regret" : scalar
            "regret_per_sample" : (N_test,) array (if return_per_sample)
        """
        # 1. Setup and validations
        X_test = np.asarray(X_test, dtype=float)
        C_test = np.asarray(C_test, dtype=float)
        Z = np.asarray(Z, dtype=float)

        N_test, d_x_test = X_test.shape
        N_ctest, d_c_test = C_test.shape
        K, d_c_Z = Z.shape

        if d_x_test != self.d_x:
            raise ValueError(f"X_test shape mismatch: expected dim {self.d_x}, got {d_x_test}")
        if d_c_test != self.d_c:
            raise ValueError(f"C_test shape mismatch: expected dim {self.d_c}, got {d_c_test}")
        if d_c_Z != self.d_c:
            raise ValueError(f"Z dimension mismatch: expected {self.d_c}, got {d_c_Z}")
        if N_test != N_ctest:
            raise ValueError("X_test and C_test must have the same number of rows.")

        # Get B: from argument or current model solution
        if B is None:
            B = np.array(self.B.level()).reshape(self.d_x, self.d_c)
        else:
            B = np.asarray(B, dtype=float).reshape(self.d_x, self.d_c)

        # 2. Vectorized calculation

        # A. Predict costs for all samples
        #    C_hat = X_test B  (N_test, d_c)
        C_hat = X_test @ B

        # B. Objective values for all feasible z under predicted costs
        #    obj_pred[i, k] = (C_hat[i] dot Z[k])
        obj_pred = C_hat @ Z.T  # (N_test, K)

        # C. Index of predicted optimal solution for each sample
        idx_hat = np.argmin(obj_pred, axis=1)  # (N_test,)

        # D. True objective values for all feasible z
        obj_true = C_test @ Z.T  # (N_test, K)

        # E. True cost of the decision we predicted
        chosen_costs = obj_true[np.arange(N_test), idx_hat]

        # F. True cost of the optimal decision (oracle)
        optimal_costs = obj_true.min(axis=1)

        # G. Regret
        # regrets = chosen_costs - optimal_costs

        regrets = (chosen_costs - optimal_costs)/optimal_costs

        avg_regret = float(regrets.mean())

        result = {"avg_regret": avg_regret}
        if return_per_sample:
            result["regret_per_sample"] = regrets

        return result  

###################################################################################
################################ Pair Optimization ################################
###################################################################################

class PairwiseECPERM:
    """
    Pairwise ECP ERM with exponential cone (Optimized).

    Problem:
        min_{B, t, v}  (1/N) * sum_{n} t_n
        s.t.
           ((B^T x_n)^T (z' - z) - t_n,  eta,  v_{n,z,z'}) in K_exp
           sum_{(z,z') in O_n} v_{n,z,z'} <= eta
    """

    def __init__(self, X, C, Z, eta: float, tol: float = 1e-9):
        """
        Parameters
        ----------
        X : array, shape (N, d_x)
        C : array, shape (N, d_c)
        Z : array, shape (K, d_c)
        eta : float
            Exponential cone parameter.
        tol : float
            Tolerance for defining the set O_n.
        """
        self.X = np.asarray(X, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.eta = float(eta)
        self.tol = float(tol)

        self.N, self.d_x = self.X.shape
        N_c, self.d_c = self.C.shape
        self.K, d_c2 = self.Z.shape

        if N_c != self.N:
            raise ValueError("X and C must have the same number of rows.")
        if d_c2 != self.d_c:
            raise ValueError("Z dim must match C dim.")

        # 1. Build Data Matrices (Vectorized)
        self._build_O_n_and_linear_operators()

        # 2. Build Mosek Model (Sparse)
        self._build_model()

    def _build_O_n_and_linear_operators(self):
        """
        Vectorized construction of indices and the H matrix.
        """
        X, C, Z = self.X, self.C, self.Z
        tol = self.tol

        # --- 1. Identify all active pairs (n, k, l) ---
        # Compute costs C_n^T z_k for all n, k
        costs = C @ Z.T  # (N, K)

        # Broadcast comparison: costs[n, k] <= costs[n, l] + tol
        # valid_mask[n, k, l] is True if pair (z_k, z_l) is in O_n
        valid_mask = (costs[:, :, None] <= costs[:, None, :] + tol)

        # Extract indices (n, k, l) corresponding to True

        n_indices, k_indices, l_indices = np.where(valid_mask)
        # remove diagonal pairs (z, z)

        # mask_offdiag = (k_indices != l_indices)

        # n_indices = n_indices[mask_offdiag]
        # k_indices = k_indices[mask_offdiag]
        # l_indices = l_indices[mask_offdiag]

        Q = len(n_indices)
        self.Q = Q
        
        if Q == 0:
            raise ValueError("No valid pairs found for O_n.")

        # Store n_indices for building sparse matrices later
        self.n_q = n_indices

        # --- 2. Build H matrix (Vectorized) ---
        # H[q] = vec( x_n outer (z_l - z_k) )
        # We construct this directly using broadcasting without loops.

        X_sel = X[n_indices]               # Shape (Q, d_x)
        Z_diff = Z[k_indices] - Z[l_indices] # Shape (Q, d_c)

        # Outer product: (Q, d_x, 1) * (Q, 1, d_c) -> (Q, d_x, d_c)
        H_tensor = X_sel[:, :, None] * Z_diff[:, None, :]
        
        # Flatten to (Q, d_x * d_c)
        self.H = H_tensor.reshape(Q, -1)

    def _build_model(self):
        """
        Builds the MOSEK optimization model using sparse mapping matrices.
        """
        self.M = Model("pairwise_ecp_erm_fast")
        M = self.M

        N, d_x, d_c, Q = self.N, self.d_x, self.d_c, self.Q

        # --- Variables ---
        self.B = M.variable("B", [d_x, d_c], Domain.unbounded())
        self.t = M.variable("t", N, Domain.unbounded())
        self.v = M.variable("v", Q, Domain.unbounded())

        # --- Linear Terms ---

        # 1. Term H * vec(B)
        # H is dense (Q x dx*dc), but usually reasonable size compared to Tmap
        B_flat = Expr.reshape(self.B, d_x * d_c)
        H_mat = Matrix.dense(self.H)
        HB = Expr.mul(H_mat, B_flat)

        # 2. Term t_{n_q} using Sparse Matrix
        # FIX: Explicitly use int32 arrays for indices
        row_indices = np.arange(Q, dtype=np.int32)
        col_indices = self.n_q.astype(np.int32)
        vals = np.ones(Q, dtype=float)

        # Matrix.sparse(num_rows, num_cols, row_idx, col_idx, values)
        Tmap_mat = Matrix.sparse(Q, N, row_indices, col_indices, vals)
        Tt = Expr.mul(Tmap_mat, self.t)

        # u_q = (H B)_q - t_{n_q}
        U = Expr.sub(HB, Tt)

        # --- Constraints ---

        # 1. Exponential Cone: ((u), eta, v) \in K_exp
        U_col = Expr.reshape(U, [Q, 1])
        v_col = Expr.reshape(self.v, [Q, 1])
        
        eta_vec = np.full((Q, 1), self.eta, dtype=float)
        eta_mat = Matrix.dense(eta_vec)
        eta_expr = Expr.constTerm(eta_mat)

        exp_triplets = Expr.hstack(v_col, eta_expr, U_col)
        M.constraint("exp_cones", exp_triplets, Domain.inPExpCone())

        # 2. Sum Constraint: sum_{q in O_n} v_q <= eta
        # We equivalent to Tmap^T * v <= eta
        # FIX: Same int32 casting for the transpose matrix indices
        # Rows of Transpose = Cols of Original (n_q)
        # Cols of Transpose = Rows of Original (0..Q-1)
        Tmap_T_mat = Matrix.sparse(N, Q, col_indices, row_indices, vals)
        sum_v = Expr.mul(Tmap_T_mat, self.v)

        M.constraint("sum_v_le_eta", sum_v, Domain.lessThan(np.full(N, self.eta)))

        # --- Objective ---
        # min (1/N) * sum(t)

        t_sum = Expr.sum(self.t)               # scalar
        obj = Expr.mul(1.0 / N, t_sum)
        M.objective("obj", ObjectiveSense.Minimize, obj)

    def solve(self, verbose: bool = False):
        """
        Solves the model. Returns dictionary of results.
        """
        if verbose:
            self.M.setLogHandler(sys.stdout)
        else:
            self.M.setLogHandler(None)

        self.M.solve()

        status = self.M.getPrimalSolutionStatus()
        if status != SolutionStatus.Optimal:
            # You might want to handle NearOptimal or distinct cases
            if verbose:
                print(f"Warning: Optimization finished with status {status}")

        B_val = np.array(self.B.level()).reshape(self.d_x, self.d_c)
        t_val = np.array(self.t.level())
        # v_val = np.array(self.v.level()) # Often very large, commented out to save return memory
        obj_val = self.M.primalObjValue()

        return {
            "B": B_val,
            "t": t_val,
            "obj": obj_val
        }

    def evaluate_regret(self, X_test, C_test, B=None, return_per_sample=False):
        """
        Vectorized regret evaluation.
        """
        X_test = np.asarray(X_test, dtype=float)
        C_test = np.asarray(C_test, dtype=float)
        N_test = X_test.shape[0]

        if B is None:
            B = np.array(self.B.level()).reshape(self.d_x, self.d_c)
        else:
            B = np.asarray(B, dtype=float).reshape(self.d_x, self.d_c)

        # 1. Predict costs: C_hat = X * B
        C_hat = X_test @ B 

        # 2. Evaluate all z options against predicted costs
        # obj_pred[n, k] = c_hat_n^T z_k
        obj_pred = C_hat @ self.Z.T 

        # 3. Find predicted best decision index
        idx_hat = np.argmin(obj_pred, axis=1)

        # 4. Evaluate actual costs
        obj_true = C_test @ self.Z.T

        # 5. Calculate Regret
        # Cost of chosen decision - Cost of optimal decision (using true C)
        chosen_costs = obj_true[np.arange(N_test), idx_hat]
        optimal_costs = obj_true.min(axis=1)

        # regrets = chosen_costs - optimal_costs
        regrets = (chosen_costs - optimal_costs)/optimal_costs
        avg_regret = float(regrets.mean())

        result = {"avg_regret": avg_regret}
        if return_per_sample:
            result["regret_per_sample"] = regrets

        return result
    
###################################################################################
############################# Likelihood Optimization #############################
###################################################################################

class MLEECPParam:
    """
    Optimized Exponential-cone ListMLE-style ERM.
    
    Features:
    - Batched constraints to avoid stack overflow (ExpressionError).
    - Symbolic domains to avoid dense matrix allocation (ValueError).
    - Vectorized operations for speed.
    """

    def __init__(self, N, d_x, d_c, Z, eta: float):
        self.N = int(N)
        self.d_x = int(d_x)
        self.d_c = int(d_c)
        self.Z = np.asarray(Z, dtype=float)
        self.eta = float(eta)

        self.K, d_c2 = self.Z.shape
        if d_c2 != self.d_c:
            raise ValueError(f"Z shape mismatch. Expected (K, {self.d_c}), got {self.Z.shape}")

        self.D = self.d_x * self.d_c

        # -----------------------------------------
        # Precompute pair indices (i, k) with i <= k
        # -----------------------------------------
        pairs_i = []
        pairs_k = []
        for i in range(self.K):
            for k in range(i, self.K):
                pairs_i.append(i)
                pairs_k.append(k)

        self.pairs_i = np.array(pairs_i, dtype=int)
        self.pairs_k = np.array(pairs_k, dtype=int)
        self.L = len(self.pairs_i)

        # -----------------------------------------
        # Precompute index mappings
        # -----------------------------------------
        self.n_indices = np.repeat(np.arange(self.N), self.L)
        self.i_indices_tiled = np.tile(self.pairs_i, self.N)
        self.k_indices_tiled = np.tile(self.pairs_k, self.N)

        # Flat indices into S (N*K) and t (N*K)
        # Using int64 for safety during calculation, though MOSEK expects standard ints
        self.idx_S = (self.n_indices * self.K + self.k_indices_tiled).astype(np.int64)
        self.idx_t = (self.n_indices * self.K + self.i_indices_tiled).astype(np.int64)

        # Indices to scatter v into (N, K, K) output tensor
        self.tensor_indices = (
            self.n_indices * self.K * self.K
            + self.i_indices_tiled * self.K
            + self.k_indices_tiled
        )

        # -----------------------------------------
        # Prebuild sparse matrix for sum constraints
        # -----------------------------------------
        rows_block = self.pairs_i
        cols_block = np.arange(self.L)
        n_range = np.arange(self.N)

        # Broadcasting to create sparse coordinates
        rows_expanded = rows_block[None, :] + (n_range[:, None] * self.K)
        cols_expanded = cols_block[None, :] + (n_range[:, None] * self.L)

        self.sum_matrix = Matrix.sparse(
            self.N * self.K,
            self.N * self.L,
            rows_expanded.ravel().tolist(),
            cols_expanded.ravel().tolist(),
            np.ones(self.N * self.L),
        )

        self._build_model()

    def _build_model(self):
        M = Model("listmle_ecp_param")
        self.M = M

        # Parameter: A_par (N*K, D)
        self.A_par = M.parameter("A", [self.N * self.K, self.D])

        # Variables
        self.B_var = M.variable("B", self.D, Domain.unbounded())
        self.t_var = M.variable("t", self.N * self.K, Domain.unbounded())
        self.v_var = M.variable("v", self.N * self.L, Domain.greaterThan(0.0))

        # S = A * B
        S = Expr.mul(self.A_par, self.B_var)

        # -----------------------------------------------------------
        # BATCHED Exponential Cone Constraints
        # Prevents "Attempted to allocate negative amount on work stack"
        # -----------------------------------------------------------
        total_cones = self.N * self.L
        batch_size = 50000  # Safe batch size for expression graph

        for start in range(0, total_cones, batch_size):
            end = min(start + batch_size, total_cones)
            current_size = end - start

            # 1. Slice 'v' variable
            v_batch = self.v_var.slice(start, end)

            # 2. Pick indices for S and t
            idx_S_batch = self.idx_S[start:end].tolist()
            idx_t_batch = self.idx_t[start:end].tolist()

            S_batch = Expr.pick(S, idx_S_batch)
            t_batch = Expr.pick(self.t_var, idx_t_batch)
            
            diff_batch = Expr.neg(Expr.add(S_batch, t_batch))

            # 3. Constant eta for this batch (Expr.constTerm is correct here for hstack)
            eta_batch = Expr.constTerm(current_size, self.eta)

            # 4. Stack and Constraint
            triplet = Expr.hstack(v_batch, eta_batch, diff_batch)
            M.constraint(triplet, Domain.inPExpCone())

        # -----------------------------------------------------------
        # Sum constraints: SumMat * v <= eta
        # -----------------------------------------------------------
        # Sparse multiplication
        sum_v = Expr.mul(self.sum_matrix, self.v_var)
        
        # CORRECTED: Use Domain.lessThan(scalar, size)
        # This defines a domain where every component is <= self.eta
        M.constraint(
            "sum_v_le_eta",
            sum_v,
            Domain.lessThan(self.eta) 
        )

        # -----------------------------------------------------------
        # Objective
        # -----------------------------------------------------------
        sum_t = Expr.sum(self.t_var)
        sum_S = Expr.sum(S)
        obj_expr = Expr.mul(1.0 / self.N, Expr.add(sum_t, sum_S))
        M.objective("obj", ObjectiveSense.Minimize, obj_expr)

    def _compute_phi_from_C(self, C):
        C = np.asarray(C, dtype=float)
        costs = C @ self.Z.T
        phi = np.argsort(costs, axis=1, kind="mergesort")
        return phi

    def solve(self, X, C, verbose: bool = True, return_v_tensor: bool = False):
        X = np.asarray(X, dtype=float)
        C = np.asarray(C, dtype=float)

        if X.shape[0] != self.N:
            raise ValueError(f"X batch size mismatch. Expected {self.N}, got {X.shape[0]}")

        # 1. Compute rankings
        phi = self._compute_phi_from_C(C)

        # 2. Build A (N*K, D)
        Z_sorted = self.Z[phi] # (N, K, d_c)
        tensor_A = X[:, np.newaxis, :, np.newaxis] * Z_sorted[:, :, np.newaxis, :]
        A_np = tensor_A.reshape(self.N * self.K, self.D)

        # 3. Update Parameter
        self.A_par.setValue(A_np)

        # 4. Solve
        if not verbose:
            self.M.setLogHandler(None)
        else:
            self.M.setLogHandler(sys.stdout)

        self.M.solve()

        # 5. Extract
        B_val = np.array(self.B_var.level()).reshape(self.d_x, self.d_c)
        t_val = np.array(self.t_var.level()).reshape(self.N, self.K)
        v_flat = np.array(self.v_var.level())
        obj_val = self.M.primalObjValue()

        if return_v_tensor:
            v_val = np.zeros((self.N, self.K, self.K))
            v_val.flat[self.tensor_indices] = v_flat
            v_out = v_val
        else:
            v_out = v_flat

        return {
            "B": B_val,
            "t": t_val,
            "v": v_out,
            "obj": obj_val,
        }
        
    def evaluate_regret(self, X_test, C_test, B=None, return_per_sample=False):
        X_test = np.asarray(X_test, dtype=float)
        C_test = np.asarray(C_test, dtype=float)
        N_test = X_test.shape[0]

        if B is None:
            B = np.array(self.B_var.level()).reshape(self.d_x, self.d_c)
        else:
            B = np.asarray(B).reshape(self.d_x, self.d_c)

        C_hat = X_test @ B
        obj_pred = C_hat @ self.Z.T
        idx_hat = np.argmin(obj_pred, axis=1)
        obj_true = C_test @ self.Z.T
        
        chosen_costs = obj_true[np.arange(N_test), idx_hat]
        optimal_costs = obj_true.min(axis=1)
        
        # regrets = chosen_costs - optimal_costs
        regrets = (chosen_costs - optimal_costs)/optimal_costs

        avg_regret = float(regrets.mean())
        
        result = {"avg_regret": avg_regret}
        if return_per_sample:
            result["regret_per_sample"] = regrets
        return result

##################################################################################
################################ DIO Optimization ################################
##################################################################################

class DIOECPParam:
    """
    Implements the DIO exponential-cone program:

        minimize   (1/N) * sum_n t_n

        s.t.       ( (B^T x_n)^T (z*(c_n) - z_k) - t_n,  η,  v_{n,k} ) ∈ K_exp
                   sum_k v_{n,k} ≤ η

        where z*(c_n) is passed as a PARAMETER (computed from C and Z).
    """

    # ----------------------------------------------------------------------
    def __init__(self, N, d_x, d_c, Z, eta: float):
        self.N = int(N)
        self.d_x = int(d_x)
        self.d_c = int(d_c)

        self.Z = np.asarray(Z, float)          # (K, d_c)
        self.K = self.Z.shape[0]
        self.eta = float(eta)

        self._build_model()

    # ----------------------------------------------------------------------
    def _build_model(self):
        M = Model("dio_ecp_param")
        self.M = M

        # --------------------------
        # Parameters
        # --------------------------
        self.Xpar = M.parameter("X", [self.N, self.d_x])          # (N, d_x)
        self.Zstar_par = M.parameter("Zstar", [self.N, self.d_c]) # (N, d_c)

        # --------------------------
        # Variables
        # --------------------------
        self.B = M.variable("B", [self.d_x, self.d_c], Domain.unbounded())
        self.t = M.variable("t", self.N, Domain.unbounded())          # t_n ∈ R
        self.v = M.variable("v", [self.N, self.K], Domain.unbounded())  # v_{n,k}

        # --------------------------
        # Expressions
        # --------------------------
        # XB = X B → (N,d_c)
        XB = Expr.mul(self.Xpar, self.B)   # (N, d_c) — predicted cost vectors

        # Φ = XB Z^T → (N,K), each entry is (B^T x_n)^T z_k
        Zmat = Matrix.dense(self.Z.T)      # (d_c, K)
        Phi = Expr.mul(XB, Zmat)           # (N, K)

        # s_n = (B^T x_n)^T z*(c_n)
        prod = Expr.mulElm(XB, self.Zstar_par)   # (N,d_c)
        s = Expr.sum(prod, 1)                     # (N,)

        # u_{n,k} = (B^T x_n)^T (z*(c_n) - z_k) = s_n - Φ_{n,k}
        s_expanded = Expr.repeat(s, self.K, 1)    # (N,K)
        u = Expr.sub(s_expanded, Phi)             # (N,K)

        # x0_{n,k} = u_{n,k} - t_n
        t_expanded = Expr.repeat(self.t, self.K, 1)   # (N,K)
        x0 = Expr.sub(u, t_expanded)                  # (N,K)

        # Vectorize for the cone: (x0_vec, eta_vec, v_vec) ∈ K_exp
        x0_vec = Expr.reshape(x0, self.N * self.K, 1)         # (NK,1)
        v_vec = Expr.reshape(self.v, self.N * self.K, 1)      # (NK,1)

        eta_vec = Matrix.dense(np.full((self.N * self.K, 1), self.eta))
        eta_expr = Expr.constTerm(eta_vec)

        # Exponential cone: (x0, x1, x2) = (u - t, eta, v)
        triple = Expr.hstack(v_vec, eta_expr, x0_vec)
        M.constraint("exp_cone", triple, Domain.inPExpCone())

        # sum_k v_{n,k} ≤ η
        M.constraint(
            "sum_v",
            Expr.sum(self.v, 1),
            Domain.lessThan(np.full(self.N, self.eta))
        )

        # --------------------------
        # Objective: (1/N) * sum_n t_n
        # --------------------------
        obj = Expr.mul(1.0 / self.N, Expr.sum(self.t))
        M.objective("dio_obj", ObjectiveSense.Minimize, obj)

    # ----------------------------------------------------------------------
    # Compute oracle Zstar(c)
    # ----------------------------------------------------------------------
    def _compute_Zstar_from_C(self, C):
        """
        Compute z*(c_n) = argmin_z c_n^T z
        Return shape (N, d_c).
        """
        C = np.asarray(C, float)            # (N, d_c)
        costs = C @ self.Z.T               # (N, K)
        idx = costs.argmin(axis=1)         # (N,)
        Zstar = self.Z[idx]                # (N, d_c)
        return Zstar

    # ----------------------------------------------------------------------
    # Solve model
    # ----------------------------------------------------------------------
    def solve(self, X, C, verbose=False):
        X = np.asarray(X, float)
        C = np.asarray(C, float)

        # Set parameters
        self.Xpar.setValue(X)
        self.Zstar_par.setValue(self._compute_Zstar_from_C(C))

        if not verbose:
            self.M.setLogHandler(None)

        self.M.solve()

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

    # ----------------------------------------------------------------------
    # REGRET EVALUATION (unchanged)
    # ----------------------------------------------------------------------
    def evaluate_regret(self, X_test, C_test, B=None, return_per_sample=False):
        """
        regret = c^T z_pred - c^T z*
        """
        X_test = np.asarray(X_test, float)
        C_test = np.asarray(C_test, float)

        N_test = X_test.shape[0]

        # Load B
        if B is None:
            B = np.array(self.B.level()).reshape(self.d_x, self.d_c)
        else:
            B = np.asarray(B, float).reshape(self.d_x, self.d_c)

        # Predicted cost vectors
        C_hat = X_test @ B                        # (N,d)

        # Evaluate predicted objective over all feasible z
        pred_costs = C_hat @ self.Z.T             # (N,K)
        idx_pred = np.argmin(pred_costs, axis=1)  # predicted z

        # True evaluation
        true_costs = C_test @ self.Z.T            # (N,K)
        oracle_costs = true_costs.min(axis=1)
        chosen_true_costs = true_costs[np.arange(N_test), idx_pred]

        # regrets = chosen_true_costs - oracle_costs
        regrets = (chosen_true_costs - oracle_costs)/oracle_costs
        avg_regret = float(regrets.mean())

        result = {"avg_regret": avg_regret}
        if return_per_sample:
            result["regret_per_sample"] = regrets

        return result