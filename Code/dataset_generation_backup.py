"""
Data generation of shortest path instances as it is done in the article
Smart “predict, then optimize” (2021) [1]. The methodology is as follows:

 1. Real model parameters are simulated as a Bernoulli (probability = 0.5)

 2. Real cost per edge are simulated with the formula
     c_ij = (1 + 1/math.sqrt(p)*(real+3)**deg )*random.uniform(1-noise, 1+noise)

     where p is the number of features of the model, real is the simulated real cost,
     deg controls the misspecification of the linear model by creating a polynomial of
     higher degree, noise is the half-width of the perturbation.


Function generate_instance has two parameters:
    K: the amount of instances to generate.
    p: the number of features to generate per instance.
    deg: controls the amount of model misspecification
    noise: random perturbation of the real cost

Function compute_shortest_path(data_file): Solve all the instances in data_file
and store it in a file with a prefix "sol_"

[1] Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.
"""
from function_lib import *

def bernoulli(p):
    if random.random() <= p:
        return 1
    else:
        return 0

def generate_instances(base_dict, num_instances, seed, input_dim=5, deg=2, mult_noise=0.5, grid_width=5):
    """
    Generates shortest path instances, wherein each instance consists of a feature vector, a vector of ground-truth
    edge costs, and a solution bitvector. Each of these parts is saved in a separate CSV file.
    :param num_instances: The number of instances to generate
    :param input_dim: The number of features
    :param deg: The degree of the ground-truth polynomial relationship between features and costs
    :param mult_noise: The half width of the noise factor
    """
    file_output_targets = base_dict + f"/targets_n_{num_instances}_input_dim_{input_dim}" \
                          f"_mult_noise_{mult_noise}_deg_{deg}_grid_width_{grid_width}.csv"
    file_output_features = base_dict + f"/features_n_{num_instances}_input_dim_{input_dim}" \
                           f"_mult_noise_{mult_noise}_deg_{deg}_grid_width_{grid_width}.csv"

    d = 2 * grid_width * (grid_width - 1)

    # Defining Payoffs matrices
    V = range(grid_width**2)
    E = []

    for j in V:
        if (j + 1) % grid_width != 0:
            E.append((j, j + 1))
        if j + grid_width < grid_width**2:
            E.append((j, j + grid_width))

    c = {}

    if not os.path.exists(base_dict):
        os.makedirs(base_dict)
        
    ff_targets = open(file_output_targets, 'w')
    ff_features = open(file_output_features, 'w')

    ff_targets.write('data,node_init,node_term,c,\n')
    string = ['at{0}'.format(i + 1) for i in range(input_dim)]
    att = ','.join(string)
    ff_features.write("data,"+ att + '\n')

    # Create one model for the entire instance
    random.seed(seed)
    B = np.array([[bernoulli(0.5) for k in range(input_dim)] for e in range(d)])

    for i in range(num_instances):
        x = np.array([round(random.gauss(0, 1), 3) for _ in range(input_dim)])
        B_matmul_x = np.matmul(B, x)
        for j in range(len(E)):
            (u, v) = E[j]

            # Generate the true model
            pred = B_matmul_x[j]
            c[u, v] = round((1 + (pred / math.sqrt(input_dim) + 3) ** deg) * random.uniform(1 - mult_noise, 1 + mult_noise), 5)
            ff_targets.write('{0},{1},{2},{3}\n'.format(i, u, v, c[u, v]))
        attributes_string = ','.join(str(x[i]) for i in range(len(x)))
        ff_features.write(str(i) + "," + attributes_string + "\n")
    ff_targets.close()

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

def read_shortest_path_data(base_dict, n, input_dim, noise, deg, num_node):
    '''
    Read data and using dataloader to generate the train, valid, test dataset
    '''
    try:
        with open(base_dict + f"/Data/Shortest_path/shortest_path_data_input_dim_{input_dim}_deg_{deg}_noise_{noise}_grid_width_{num_node}.pkl", "rb") as f:
            y_train, x_test, y_test, train_dl, valid_dl, test_dl = pickle.load(f)
    except:
        df_features = pd.read_csv(base_dict+f"/Data/Shortest_path/Raw_data/features_n_{n}_input_dim_{input_dim}"
                          f"_mult_noise_{noise}_deg_{deg}_grid_width_{num_node}.csv")
        df_targets = pd.read_csv(base_dict+f"/Data/Shortest_path/Raw_data/targets_n_{n}_input_dim_{input_dim}"
                                f"_mult_noise_{noise}_deg_{deg}_grid_width_{num_node}.csv")

        # Converting data to NumPy arrays
        x = df_features.iloc[:, 1:].values.astype(np.float32)
        output_dim = num_node * num_node - num_node
        y = df_targets.iloc[:, 3].values.reshape(-1, output_dim).astype(np.float32)

        # Split datasets into training, validation and test sets
        n_samples = len(x)
        n_training = n_samples * 8 // 10
        n_validation = n_samples * 1 // 10
        x_train, y_train = x[:n_training], y[:n_training]
        x_valid, y_valid = x[n_training:n_training + n_validation], y[n_training:n_training + n_validation]
        x_test, y_test = x[n_training + n_validation:], y[n_training + n_validation:]

        train_dl = DataLoader(DataWrapper(x_train, y_train), batch_size=32)
        valid_dl = DataLoader(DataWrapper(x_valid, y_valid), batch_size=32)
        test_dl = DataLoader(DataWrapper(x_test, y_test), batch_size=32)

        with open(base_dict + f"/Data/Shortest_path/shortest_path_data_input_dim_{input_dim}_deg_{deg}_noise_{noise}_grid_width_{num_node}.pkl", 'wb') as f:
            pickle.dump([y_train, x_test, y_test, train_dl, valid_dl, test_dl], f)
    
    return y_train, x_test, y_test, train_dl, valid_dl, test_dl

if __name__ == '__main__':
    n_simulation = 50
    degs = [1, 2, 4, 6, 8]
    input_dims = [5]
    ns = [100, 1000, 5000]
    mult_noises = [0, 0.5]
    grid_widths = [5, 10, 15]

    SETTINGS = [(n, input_dim, mult_noise, deg, grid_width) for n in ns for input_dim in input_dims
                for mult_noise in mult_noises for deg in degs for grid_width in grid_widths]
    base_path = change_to_parent_dir_once(True)
    print("Start to generate datasets for shortest path problem.")
    base_dict = base_path + '/Data/Shortest_path/Raw_data/'

    for (n, input_dim, mult_noise, deg, grid_width) in SETTINGS:
        for seed in range(n_simulation):
            generate_instances(base_dict, n, seed, input_dim=input_dim, mult_noise=mult_noise, deg=deg, grid_width=grid_width)

    print("Finish generating datasets for shortest path problem.")