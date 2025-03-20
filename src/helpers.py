import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat
import torch.utils.data as data

def print_dictionary(hp_dict: dict[str, str], pre_text: str = None, post_text: str = None) -> None:
    """
    Print given dictionary

    `hp_dict`: dictionary dictionary to print key and values for  
    `pre-text`: text to print before dictionary  
    `post-text`: text to print after dictionary  

    Returns: `None`
    """
    if pre_text is not None:
        print(pre_text)
    for key in hp_dict.keys():
        print(f"{key}: {hp_dict[key]}")
    if post_text is not None:
        print(post_text)

    return None

def load_dataset_raw(dataset, matrix_id):
    """
    Load original unprocessed dataset (both training and testing matrices)  

    `dataset`: Name of dataset (either \"ODE_Lorenz\", or \"PDE_KS\")  
    `matrix_id`: An integer as a string, from 1 to 10 inclusive  

    Returns: (`train_mat`, `test_mat`)  
    The training and testing matrices
    """
    if dataset in ["ODE_Lorenz", "PDE_KS"]:
        train_mat = loadmat(str(Path(os.path.abspath(__file__)).parent.parent.parent.parent.absolute() / "Datasets" / dataset / f"X{matrix_id}train.mat"))
        train_mat = train_mat[list(train_mat.keys())[-1]]
        test_mat = loadmat(str(Path(os.path.abspath(__file__)).parent.parent.parent.parent.absolute() / "Datasets" / dataset / f"X{matrix_id}test.mat"))
        test_mat = test_mat[list(test_mat.keys())[-1]]

        train_mat = torch.Tensor(train_mat.astype(np.float32))
        test_mat = torch.Tensor(test_mat.astype(np.float32))
    else:
        raise Exception(f"Timeseries dataset {dataset} not found")
    return train_mat, test_mat

def scoring2(truth, prediction):
    """
    Scoring for least-square reconstruction  

    `truth`: Ground-truth matrix  
    `prediction`: Predicted matrix  

    Returns:  
    Floating point value as a score  
    """
    Est = np.linalg.norm(truth-prediction,2)/np.linalg.norm(truth,2)
    E1 = 100*(1-Est)
    return E1

def loguniform(low=0, high=1, size=None):
    """
    Generate values from a log-uniform distribution  

    `low`: float or array_like of floats, optional  
    Lower boundary of the output interval. All values generated will be greater than or equal to low. The default value is 0.  

    `high`: float or array_like of floats  
    Upper boundary of the output interval. All values generated will be less than or equal to high. The high limit may be included in the returned array of floats due to floating-point rounding in the equation low + (high-low) * random_sample(). The default value is 1.0.  

    `size`: int or tuple of ints, optional  
    Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if low and high are both scalars. Otherwise, np.broadcast(low, high).size samples are drawn.  

    Returns:  
    `out`: ndarray or scalar  
    Drawn samples from the parameterized log-uniform distribution.  
    """
    if low <= 0.0 : raise Exception("Lower bound has to be > 0.0")
    values = np.exp(np.random.uniform(np.log(low), np.log(high), size))
    return values
