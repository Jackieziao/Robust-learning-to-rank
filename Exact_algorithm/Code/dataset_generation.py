from function_lib import *  # assuming this defines change_to_parent_dir_once, etc.

def bernoulli(p: float) -> int:
    """Sample a Bernoulli(p) random variable."""
    return 1 if random.random() <= p else 0


def generate_instances(num_instances,
                       seed,
                       input_dim=5,
                       deg=2,
                       mult_noise=0.5,
                       grid_width=5):
    """
    Generate shortest-path data for a SINGLE seed.

    Parameters
    ----------
    num_instances : int
        Number of instances to generate for this seed.
    seed : int
        Random seed.
    input_dim : int
        Number of features per instance.
    deg : int
        Degree of polynomial misspecification.
    mult_noise : float
        Half-width of multiplicative noise.
    grid_width : int
        Grid width (grid_width x grid_width).

    Returns
    -------
    X : np.ndarray, shape (num_instances, input_dim)
        Feature matrix.
    Y : np.ndarray, shape (num_instances, d)
        Edge-cost matrix, where d is the number of edges in the grid.
    """
    # number of directed edges in an n x n grid (right + down)
    d = 2 * grid_width * (grid_width - 1)

    # Build edge list E in a fixed order
    V = range(grid_width ** 2)
    E = []
    for j in V:
        if (j + 1) % grid_width != 0:
            E.append((j, j + 1))
        if j + grid_width < grid_width ** 2:
            E.append((j, j + grid_width))
    assert len(E) == d

    # Fix randomness for this seed
    random.seed(seed)
    np.random.seed(seed)

    # One Bernoulli parameter matrix for all instances under this seed
    B = np.array([[bernoulli(0.5) for _ in range(input_dim)] for _ in range(d)])

    X_list = []
    Y_list = []

    for _ in range(num_instances):
        # features x ~ N(0,1)
        x = np.array([random.gauss(0, 1) for _ in range(input_dim)], dtype=np.float32)
        Bx = B @ x  # shape (d,)

        # edge costs in the fixed order of E
        c_vec = []
        for j in range(d):
            pred = Bx[j]
            cost = (1 + (pred / math.sqrt(input_dim) + 3) ** deg) * \
                   random.uniform(1 - mult_noise, 1 + mult_noise)
            c_vec.append(cost)

        X_list.append(x)
        Y_list.append(c_vec)

    X = np.array(X_list, dtype=np.float32)  # (num_instances, input_dim)
    Y = np.array(Y_list, dtype=np.float32)  # (num_instances, d)

    return X, Y


if __name__ == '__main__':
    # n_simulation = 50          # number of seeds (= number of instances in the pickle)
    # degs = [1, 2, 4, 6, 8]
    # input_dims = [5]
    # ns = [100, 1000, 5000]     # num_instances per seed
    # mult_noises = [0, 0.5]
    # grid_widths = [5]

    n_simulation = 30          # number of seeds (= number of instances in the pickle)
    degs = [1, 2, 4, 6, 8]
    input_dims = [3]
    ns = [100]     # num_instances per seed
    mult_noises = [0, 0.5]
    grid_widths = [3]

    SETTINGS = [(n, input_dim, mult_noise, deg, grid_width)
                for n in ns
                for input_dim in input_dims
                for mult_noise in mult_noises
                for deg in degs
                for grid_width in grid_widths]

    base_path = change_to_py_file_dir(True)
    print("Start to generate datasets for shortest path problem (pkl only).")

    out_dir = os.path.join(base_path, "Data", "Shortest_path")
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for (n, input_dim, mult_noise, deg, grid_width) in SETTINGS:
        print(f"  Setting: n={n}, p={input_dim}, noise={mult_noise}, "
              f"deg={deg}, grid_width={grid_width}")

        # For this setting, we will collect 50 instances (one per seed)
        instances = []  # length = n_simulation

        for seed in range(n_simulation):
            X_seed, Y_seed = generate_instances(
                num_instances=n,
                seed=seed,
                input_dim=input_dim,
                mult_noise=mult_noise,
                deg=deg,
                grid_width=grid_width
            )
            instances.append({
                "seed": seed,
                "x": X_seed,   # shape (n, input_dim)
                "y": Y_seed    # shape (n, d)
            })

        # One pickle per setting, containing 50 instances (seeds)
        pkl_path = (
            out_dir +
            f"/shortest_path_instances_n_{n}"
            f"_input_dim_{input_dim}"
            f"_deg_{deg}"
            f"_noise_{mult_noise}"
            f"_grid_width_{grid_width}.pkl"
        )

        with open(pkl_path, "wb") as f:
            pickle.dump(instances, f)

    print("Finish generating datasets for shortest path problem.")
