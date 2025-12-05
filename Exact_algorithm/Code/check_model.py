from function_lib import *

# Import your model

class PairwiseECPERM:
    """
    Pairwise ECP ERM with exponential cone (Optimized).

    Problem:
        min_{B, t, v}  (1/N) * sum_{n} t_n
        s.t.
           ((B^T x_n)^T (z' - z) - t_n,  eta,  v_{n,z,z'}) in K_exp
           sum_{(z,z') in O_n} v_{n,z,z'} <= eta
    """

    def __init__(self, X, C, Z, eta: float, tol: float = 1e-9, lambda_reg: float = 1e-4):
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
        self.lambda_reg = float(lambda_reg)

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

        mask_offdiag = (k_indices != l_indices)

        n_indices = n_indices[mask_offdiag]
        k_indices = k_indices[mask_offdiag]
        l_indices = l_indices[mask_offdiag]

        
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
        
        # --- Ridge regularization on B via rotated quadratic cone ---

        # Flatten B to a single vector
        B_vec = Expr.reshape(self.B, d_x * d_c)

        # Scalar variable r >= 0 with r >= ||B_vec||_2^2 enforced by rotated QC
        self.r = M.variable("r", 1, Domain.greaterThan(0.0))

        # Build cone vector [r, 0.5, B_vec] \in QR^{2 + d_x*d_c}
        cone_expr = Expr.vstack(self.r, Expr.constTerm(self.eta), B_vec)
        M.constraint("ridge_qcone", cone_expr, Domain.inRotatedQCone())

        # --- Objective: (1/N) * sum(t) + lambda_reg * r ---

        t_sum = Expr.sum(self.t)               # scalar
        ridge_term = Expr.mul(self.lambda_reg, Expr.sum(self.r))  # r is 1-dim

        obj = Expr.add(
            Expr.mul(1.0 / N, t_sum),
            ridge_term
        )
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

        regrets = chosen_costs - optimal_costs
        avg_regret = float(regrets.mean())

        result = {"avg_regret": avg_regret}
        if return_per_sample:
            result["regret_per_sample"] = regrets

        return result







#########################################################################################################
##################################### Testing the implementation  #######################################
#########################################################################################################
def tiny_integer_test():
    random.seed(0)
    np.random.seed(0)

    # ------------------------------------------------------------
    # 1. Feasible set Z for 2×2 toy shortest-path problem
    # ------------------------------------------------------------
    # We define 4 nodes: [s, right, down, t]
    #
    # Path A: s → right → t  = [1, 1, 0, 1]
    # Path B: s → down  → t  = [1, 0, 1, 1]
    #
    Z = np.array([
        [1, 1, 0, 1],   # Path A
        [1, 0, 1, 1],   # Path B
    ], dtype=int)

    # ------------------------------------------------------------
    # 2. Integer X (features) for N=2 samples
    # ------------------------------------------------------------
    X = np.array([
        [1, 0, -1],    # sample 1
        [2, 1,  0],    # sample 2
    ], dtype=int)

    # ------------------------------------------------------------
    # 3. Integer C (true costs)
    # ------------------------------------------------------------
    C = np.array([
        [3, 1, 4, 2],   # sample 1
        [2, 0, 5, 3],   # sample 2
    ], dtype=int)

    print("X:\n", X)
    print("C:\n", C)
    print("Z:\n", Z)

    # ------------------------------------------------------------
    # 4. Build solver with eta = 0.5
    # ------------------------------------------------------------
    eta = 0.5
    solver = PairwiseECPERM(X=X, C=C, Z=Z, eta=eta)

    # ------------------------------------------------------------
    # 5. Dump MOSEK model to inspect constraints / cones
    # ------------------------------------------------------------
    solver.M.writeTask("tiny_integer_pairwise_ecp.ptf")
    print("MOSEK task written to tiny_integer_pairwise_ecp.ptf")

    try:
        print("Number of variables:", solver.M.getNvar())
        print("Number of constraints:", solver.M.getNcon())
    except:
        pass

    # ------------------------------------------------------------
    # 6. Solve the tiny problem
    # ------------------------------------------------------------
    res = solver.solve(verbose=True)
    print("Result keys:", list(res.keys()))
    print("Learned B =\n", res["B"])

    # ------------------------------------------------------------
    # 7. Evaluate regret on same samples (sanity check)
    # ------------------------------------------------------------
    eval_res = solver.evaluate_regret(X, C, return_per_sample=False)
    print("Avg regret =", eval_res["avg_regret"])

    # ------------------------------------------------------------
    # 8. Cleanup
    # ------------------------------------------------------------
    solver.M.dispose()
    del solver
    gc.collect()


if __name__ == "__main__":
    tiny_integer_test()

