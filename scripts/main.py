# Standard imports
import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path

# Import files from "src" by adding the folder to Python Path
pkg_path = str(Path(os.path.abspath(__file__)).parent.parent.absolute() / "src")
sys.path.insert(0, pkg_path)
import helpers

def main(args=None):
    # Load training and testing matrix
    train_mat, test_mat = helpers.load_dataset_raw(args.dataset, args.matrix_id)

    # Generate predictions based on search type
    if args.search_type == "avg":
        # Predicted matrix is average of training matrix
        pred_mat = np.average(train_mat, axis=1)
        pred_mat = np.reshape(pred_mat, (-1,1))
        pred_mat = np.tile(pred_mat, (1,test_mat.shape[1]))
    elif args.search_type == "zero":
        # Predicted matrix is just zeros
        pred_mat = np.zeros_like(test_mat)
    elif args.search_type == "constant":
        # Predicted matrix is just a constant
        pred_mat = np.ones_like(test_mat)*args.constant
    elif args.search_type == "random":
        # Predicted matrix is what evaluates best on testing set from random search

        # Check input
        if args.random_lower_bound is None:
            raise Exception(f"--random_lower_bound cannot be None.")
        if args.random_upper_bound is None:
            raise Exception(f"--random_upper_bound cannot be None.")
        if args.random_n_values is None:
            raise Exception(f"--random_n_values cannot be None.")
        if args.random_lower_bound > args.random_upper_bound:
            raise Exception(f"--random_lower_bound ({args.random_lower_bound}) cannot be larger than --random_upper_bound ({args.random_upper_bound}).")
        if args.random_n_values < 1:
            raise Exception(f"--random_n_values must be at least 1 ({args.random_n_values}).")

        # Set up hyperparameters
        if args.random_distribution == "uniform":
            hyperparams = np.random.uniform(args.random_lower_bound,
                                      args.random_upper_bound,
                                      args.random_n_values)
        elif args.random_distribution == "log":
            hyperparams = helpers.loguniform(args.random_lower_bound,
                                      args.random_upper_bound,
                                      args.random_n_values)
        else:
            raise Exception(f"Hyperparameter search option {args.random_distribution} is not valid. Use one of \"uniform\" or \"log\".")
        print("Searching over hyperparameter space:")
        print(hyperparams)

        # Do grid search, keep track of maximum score and the associated hyperparameter
        max_hyperparameter = None
        max_score = -np.inf
        for hyperparam in hyperparams:
            pred_mat = np.ones_like(test_mat)*hyperparam
            score = helpers.scoring2(test_mat, pred_mat)
            if max_score < score:
                max_score = score
                max_hyperparameter = hyperparam
            
        # Use best scoring value
        print(f"Best hyperparameter was {max_hyperparameter}")
        print()
        pred_mat = np.ones_like(test_mat)*max_hyperparameter
    else:
        raise Exception(f"Search type {args.search_type} is not valid. Use one of \"avg\", \"zero\", \"constant\", or \"random\".")

    # Evaluate predicted matrix
    score = helpers.scoring2(test_mat, pred_mat)
    print(f"Final score using method \"{args.search_type}\" is {score:0.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Config file, if used populate the below arguments automatically
    parser.add_argument("--config", type=str, default=None, help="YAML configuration file.")

    # Dataset selection
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to use, either \"ODE_Lorenz\" or \"PDE_KS\".")
    parser.add_argument('--matrix_id', type=str, default=None, help="Integer value for matrix to use from the provided dataset, a value from 1 to 10 inclusive.")

    # Search type
    parser.add_argument('--search_type', type=str, default=None, help="Which type of model to use, choose from \"avg\", \"zero\", \"constant\", or \"random\"")

    # Random search options
    parser.add_argument('--random_lower_bound', type=float, default=None, help="When \"--search_type\" is \"rand\", provide the lower bound to search over.")
    parser.add_argument('--random_upper_bound', type=float, default=None, help="When \"--search_type\" is \"rand\", provide the upper bound to search over.")
    parser.add_argument('--random_n_values', type=int, default=None, help="When \"--search_type\" is \"rand\", provide the number of hyperparameters to search over.")
    parser.add_argument('--random_distribution', type=str, default=None, help="When \"--search_type\" is \"rand\", specify if we're searching over \"uniform\" or \"log\" distribution.")
    parser.add_argument('--random_seed', type=float, default=None, help="When \"--search_type\" is \"rand\", specify the seed for numpy for reproducability.")
    args = parser.parse_args()

    # Try opening config
    try:
        with open(args.config, 'r') as file:
            data = yaml.safe_load(file)
        print(f"Using provided config file ({args.config}).")

        # Update `args` with yaml configuration
        for key, value in data.items():
            setattr(args, key, value)
    except FileNotFoundError:
        print(f"The file {args.config} was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing the YAML file: {e}")

    # Print configuration
    helpers.print_dictionary(vars(args), "Configuration values:", "")

    # Set seed
    np.random.seed(args.random_seed)

    main(args)
