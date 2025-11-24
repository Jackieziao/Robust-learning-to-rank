
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