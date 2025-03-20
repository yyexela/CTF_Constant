import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import torch.utils.data as data

def load_dataset_raw(dataset, matrix_id):
    """
    Load original unprocessed dataset (both training and testing matrices)
    """
    if dataset in ["ODE_Lorenz", "PDE_KS"]:
        train_mat = loadmat(f'../../Datasets/{dataset}/X{matrix_id}train.mat')
        train_mat = train_mat[list(train_mat.keys())[-1]]
        test_mat = loadmat(f'../../Datasets/{dataset}/X{matrix_id}test.mat')
        test_mat = test_mat[list(test_mat.keys())[-1]]

        train_mat = torch.Tensor(train_mat.astype(np.float32))
        test_mat = torch.Tensor(test_mat.astype(np.float32))
    else:
        raise Exception(f"Timeseries dataset {dataset} not found")
    return train_mat, test_mat

# SCORING FOR LEAST-SQUARE RECONSTRUCTION
def scoring2(truth, prediction):
    '''produce reconstruction fit score.'''
    Est = np.linalg.norm(truth-prediction,2)/np.linalg.norm(truth,2)
    E1 = 100*(1-Est)
    return E1
