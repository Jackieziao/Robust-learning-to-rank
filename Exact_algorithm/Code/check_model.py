from function_lib import *

#########################################################################################################
##################################### Testing the implementation  #######################################
#########################################################################################################
def tiny_lst_ecp_test(eta=0.5):
    """
    Tiny sanity test for LSTECPParam on a 2x2 grid shortest-path problem.

    - Grid: 2x2, moves right/down only → exactly 2 s->t paths.
    - Z: all feasible s->t paths over 4 edges.
    - N = 2 samples, d_x = 3.
    """
    print("==== Tiny LSTECPParam 2x2 SP test ====")
    random.seed(0)
    np.random.seed(0)

    # ------------------------------------------------------------
    # 1) Feasible paths Z from 2x2 grid
    # ------------------------------------------------------------
    sols, edges, G = generate_all_feasible_solutions(grid_width=2)
    Z = np.vstack(sols).astype(float)   # (K=2, d_c=4)
    K, d_c = Z.shape

    print("Edges (order):", edges)
    print("Z (feasible s->t paths):\n", Z)
    print(f"K = {K}, d_c = {d_c}")

    # ------------------------------------------------------------
    # 2) Tiny dataset: N = 2 samples, d_x = 3
    # ------------------------------------------------------------
    X = np.array([
        [1.0, 0.0, -1.0],   # sample 0
        [2.0, 1.0,  0.0],   # sample 1
    ], dtype=float)
    N, d_x = X.shape

    # True costs C over the 4 edges (must match 'edges' order)
    C = np.array([
        [3.0, 1.0, 4.0, 2.0],   # sample 0
        [2.0, 0.0, 5.0, 3.0],   # sample 1
    ], dtype=float)

    print("X:\n", X)
    print("C:\n", C)

    # ------------------------------------------------------------
    # 3) Build and solve LSTECPParam
    # ------------------------------------------------------------
    solver = LSTECPParam(
        N=N,
        d_x=d_x,
        d_c=d_c,
        Z=Z,
        eta=eta
    )

    # Solve ERM
    res = solver.solve(X, C, verbose=True)

    print("\n--- Solve result ---")
    print("Keys:", list(res.keys()))
    print("Learned B (d_x x d_c):\n", res["B"])
    print("t (N,):\n", res["t"])
    print("v (N x K):\n", res["v"])
    print("Objective value:", res["obj"])

    # ------------------------------------------------------------
    # 4) Evaluate regret on the same tiny dataset (sanity check)
    # ------------------------------------------------------------
    eval_res = solver.evaluate_regret(X_test=X, C_test=C, return_per_sample=True)
    print("\n--- Evaluation ---")
    print("Average relative regret:", eval_res["avg_regret"])
    print("Regret per sample:", eval_res["regret_per_sample"])

    # ------------------------------------------------------------
    # 5) Optionally inspect pi(C) to see the ListMLE weights
    # ------------------------------------------------------------
    pi_val = solver._compute_pi_from_C(C)
    print("\npi(C) (softmax weights over paths):\n", pi_val)

    # ------------------------------------------------------------
    # 6) Dump MOSEK task to inspect the conic model
    # ------------------------------------------------------------
    solver.M.writeTask("tiny_lst_ecp_2x2_sp.ptf")
    print("\nMOSEK task written to tiny_lst_ecp_2x2_sp.ptf")

    # Cleanup
    solver.M.dispose()
    del solver
    gc.collect()
    print("==== Tiny LSTECPParam 2x2 SP test done ====")

def tiny_pair_test(eta=0.5):

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
    # 4. Build solver
    # ------------------------------------------------------------
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

def tiny_dio_ecp_test(eta=0.5):
    """
    Tiny sanity test for DIOECPParam on a 2x2 grid shortest-path problem.

    - Grid: 2x2, moves right/down only → exactly 2 s->t paths.
    - Z: all feasible s->t paths over 4 edges.
    - N = 2 samples, d_x = 3.
    """
    print("==== Tiny DIOECPParam 2x2 SP test ====")
    random.seed(0)
    np.random.seed(0)

    # ------------------------------------------------------------
    # 1) Feasible paths Z from 2x2 grid
    # ------------------------------------------------------------
    sols, edges, G = generate_all_feasible_solutions(grid_width=2)
    Z = np.vstack(sols).astype(float)   # (K=2, d_c=4)
    K, d_c = Z.shape

    print("Edges (order):", edges)
    print("Z (feasible s->t paths):\n", Z)
    print(f"K = {K}, d_c = {d_c}")

    # ------------------------------------------------------------
    # 2) Tiny dataset: N = 2 samples, d_x = 3
    # ------------------------------------------------------------
    X = np.array([
        [1.0, 0.0, -1.0],   # sample 0
        [2.0, 1.0,  0.0],   # sample 1
    ], dtype=float)
    N, d_x = X.shape

    # True costs C over the 4 edges (must match 'edges' order)
    C = np.array([
        [3.0, 1.0, 4.0, 2.0],   # sample 0
        [2.0, 0.0, 5.0, 3.0],   # sample 1
    ], dtype=float)

    print("X:\n", X)
    print("C:\n", C)

    # ------------------------------------------------------------
    # 3) Build and solve DIOECPParam
    # ------------------------------------------------------------
    solver = DIOECPParam(N=N, d_x=d_x, d_c=d_c, Z=Z, eta=eta)

    res = solver.solve(X, C, verbose=True)

    print("\n--- Solve result ---")
    print("Keys:", list(res.keys()))
    print("Learned B (d_x x d_c):\n", res["B"])
    print("t (N,):\n", res["t"])
    print("v (N x K):\n", res["v"])
    print("Objective value:", res["obj"])

    # ------------------------------------------------------------
    # 4) Evaluate regret on the same tiny dataset (sanity check)
    # ------------------------------------------------------------
    eval_res = solver.evaluate_regret(X_test=X, C_test=C, return_per_sample=True)
    print("\n--- Evaluation ---")
    print("Average relative regret:", eval_res["avg_regret"])
    print("Regret per sample:", eval_res["regret_per_sample"])

    # ------------------------------------------------------------
    # 5) Optionally dump MOSEK task to inspect the model
    # ------------------------------------------------------------
    solver.M.writeTask("tiny_dio_ecp_2x2_sp.ptf")
    print("\nMOSEK task written to tiny_dio_ecp_2x2_sp.ptf")

    # Cleanup
    solver.M.dispose()
    del solver
    gc.collect()
    print("==== Tiny DIOECPParam 2x2 SP test done ====")

def tiny_mle_ecp_test(eta=0.5):
    """
    Tiny sanity test for MLEECPParam on a 2x2 grid shortest-path problem.

    - Grid: 2x2, moves right/down only => exactly 2 s->t paths.
    - Z: all feasible s->t paths (K=2, d_c=4 edges).
    - N = 2 samples, d_x = 3 features.
    """
    print("==== Tiny MLEECPParam 2x2 SP test ====")
    random.seed(0)
    np.random.seed(0)

    # ---------------------------------------------
    # 1) Feasible paths Z from 2x2 grid
    # ---------------------------------------------
    sols, edges, G = generate_all_feasible_solutions(grid_width=2)
    Z = np.vstack(sols).astype(float)   # (K=2, d_c=4)
    K, d_c = Z.shape

    print("Edges (order):", edges)
    print("Z (feasible s->t paths):\n", Z)
    print(f"K = {K}, d_c = {d_c}")

    # ---------------------------------------------
    # 2) Tiny dataset: N = 2 samples, d_x = 3
    # ---------------------------------------------
    X = np.array([
        [1.0, 0.0, -1.0],   # sample 0
        [2.0, 1.0,  0.0],   # sample 1
    ], dtype=float)
    N, d_x = X.shape

    # True costs C over the 4 edges (must match 'edges' order)
    C = np.array([
        [3.0, 1.0, 4.0, 2.0],   # sample 0
        [2.0, 0.0, 5.0, 3.0],   # sample 1
    ], dtype=float)

    print("X:\n", X)
    print("C:\n", C)

    # ---------------------------------------------
    # 3) Build and solve MLEECPParam
    # ---------------------------------------------
    solver = MLEECPParam(N=N, d_x=d_x, d_c=d_c, Z=Z, eta=eta)

    # Solve ERM
    res = solver.solve(X, C, verbose=True, return_v_tensor=False)

    print("\n--- Solve result ---")
    print("Keys:", list(res.keys()))
    print("Learned B (d_x x d_c):\n", res["B"])
    print("t (N x K):\n", res["t"])
    print("Objective value:", res["obj"])

    # ---------------------------------------------
    # 4) Evaluate regret on the same tiny dataset
    # ---------------------------------------------
    eval_res = solver.evaluate_regret(X_test=X, C_test=C, return_per_sample=True)
    print("\n--- Evaluation ---")
    print("Average relative regret:", eval_res["avg_regret"])
    print("Regret per sample:", eval_res["regret_per_sample"])

    # ---------------------------------------------
    # 5) Optionally dump MOSEK task to inspect
    # ---------------------------------------------
    solver.M.writeTask("tiny_mle_ecp_2x2_sp.ptf")
    print("\nMOSEK task written to tiny_mle_ecp_2x2_sp.ptf")

    # Cleanup
    solver.M.dispose()
    del solver
    gc.collect()
    print("==== Tiny MLEECPParam 2x2 SP test done ====")

def tiny_spo_test():
    random.seed(0)
    np.random.seed(0)

    # Integer X and C for N=2 samples
    X = np.array([
        [1, 0, -1],
        [2, 1,  0],
    ], dtype=int)

    C = np.array([
        [3, 1, 4, 2],
        [2, 0, 5, 3],
    ], dtype=int)

    print("X:\n", X)
    print("C:\n", C)

    # Build feasible paths Z from a 2x2 grid
    sols, edges, G = generate_all_feasible_solutions(grid_width=2)
    Z = np.vstack(sols)  # (K, m), m = number of edges = d_c

    # Build incidence matrix and RHS
    A_inc, b = build_incidence_matrix_and_b(edges, G)

    # FIX 1: transpose A to get shape (d_c, d_p) = (n_edges, n_nodes)
    A = A_inc.T

    d_c = Z.shape[1]      # number of edges
    d_p = A.shape[1]      # number of nodes

    solver = SPOShortestPathParam(
        N=X.shape[0],
        d_x=X.shape[1],
        d_c=d_c,
        d_p=d_p,
        A=A,
        b=b
    )

    # Optional: inspect MOSEK task
    solver.M.writeTask("tiny_integer_check.ptf")
    print("MOSEK task written to tiny_integer_check.ptf")

    # Compute z_star for training data
    Zstar_train = compute_z_star_batch(C, Z)

    # Solve training problem
    res = solver.solve(X, C, Zstar_train, verbose=False)
    print("Result keys:", list(res.keys()))
    print("Learned B =\n", res["B"])

    # FIX 2: pass Z into evaluate_regret
    eval_res = solver.evaluate_regret(X, C, Z, return_per_sample=False)
    print("Avg regret =", eval_res["avg_regret"])

    solver.M.dispose()
    del solver
    gc.collect()

if __name__ == "__main__":
    FLAG = 2
    eta=0.0001
    match FLAG:
        case 1:
            tiny_dio_ecp_test(eta)
        case 2:
            tiny_lst_ecp_test(eta)
        case 3:
            tiny_mle_ecp_test(eta)
        case 4:
            tiny_pair_test(eta)
        case 5:
            tiny_spo_test()
        case _:
            print("Unknown FLAG value")