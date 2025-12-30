"""
AntiFraud - Main Training Entry Point
=====================================

This is the main entry point for training fraud detection models.
It supports three graph neural network models:

    1. STAGN (Spatial-Temporal Attention Graph Network)
       - Uses 2D feature matrices (5 features × 8 time windows)
       - Combines temporal attention + CNN + GCN
       - Command: python main.py --method stagn
    
    2. GTAN (Graph Temporal Attention Network)
       - Uses 127 temporal aggregation features
       - Graph Transformer with multi-head attention
       - Command: python main.py --method gtan
    
    3. RGTAN (Risk-aware GTAN)
       - Extends GTAN with neighbor risk statistics
       - Uses 133 features (127 + 6 neighbor stats)
       - Command: python main.py --method rgtan

Usage:
    python main.py --method [stagn|gtan|rgtan]

Configuration:
    Model hyperparameters are stored in config/*.yaml files.
    Each model has its own configuration file.

Data:
    All models use the S-FFSD (Simulated Financial Fraud Semi-supervised Dataset).
    Raw data: data/S-FFSD.csv
    Processed features are cached for faster loading.
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
from scipy.io import loadmat
import yaml

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments and load configuration
    
    The --method argument determines which model to train and
    which configuration file to load.
    
    Returns:
        dict: Configuration dictionary with all hyperparameters
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve'
    )
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']

    # ============ LOAD MODEL-SPECIFIC CONFIGURATION ============
    # Each model has its own YAML configuration file
    if method in ['mcnn']:
        yaml_file = "config/mcnn_cfg.yaml"
    elif method in ['stan']:
        yaml_file = "config/stan_cfg.yaml"
    elif method in ['stan_2d']:
        yaml_file = "config/stan_2d_cfg.yaml"
    elif method in ['stagn']:
        yaml_file = "config/stagn_cfg.yaml"
    elif method in ['gtan']:
        yaml_file = "config/gtan_cfg.yaml"
    elif method in ['rgtan']:
        yaml_file = "config/rgtan_cfg.yaml"
    elif method in ['hogrl']:
        yaml_file = "config/hogrl_cfg.yaml"
    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    # Load YAML configuration
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    return args


def base_load_data(args: dict):
    """
    Load and preprocess data for base models (MCNN, STAN)
    
    This function:
    1. Reads the raw S-FFSD.csv file
    2. Generates 2D or 3D feature matrices using span_data functions
    3. Splits into train/test sets
    4. Saves to numpy files for faster loading in future runs
    
    Args:
        args (dict): Configuration dictionary with:
            - test_size: fraction for test set
            - method: model method name
            - trainfeature, testfeature, trainlabel, testlabel: file paths
    """
    # Load raw S-FFSD dataset
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    
    # ============ GENERATE FEATURES ============
    # STAN uses 3D features, others use 2D
    if args['method'] == 'stan':
        if os.path.exists("data/tel_3d.npy"):
            return  # Already cached
        features, labels = span_data_3d(feat_df)
    else:
        if os.path.exists("data/tel_2d.npy"):
            return  # Already cached
        features, labels = span_data_2d(feat_df)
    
    # ============ TRAIN/TEST SPLIT ============
    num_trans = len(feat_df)
    trf, tef, trl, tel = train_test_split(
        features, labels, 
        train_size=train_size, 
        stratify=labels,  # Maintain class distribution
        shuffle=True
    )
    
    # ============ SAVE TO FILES ============
    trf_file = args['trainfeature']
    tef_file = args['testfeature']
    trl_file = args['trainlabel']
    tel_file = args['testlabel']
    
    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return


def main(args):
    """
    Main training function - dispatches to appropriate model trainer
    
    This function routes to the correct model based on args['method']:
    
        STAGN: Spatial-Temporal Attention Graph Network
            - Loads data with load_stagn_data()
            - Creates Source→Target transaction graph
            - Trains with stagn_main()
        
        GTAN: Graph Temporal Attention Network
            - Loads data with load_gtan_data()
            - Creates similarity-based transaction graph
            - Uses K-fold cross-validation
        
        RGTAN: Risk-aware GTAN
            - Same as GTAN but includes neighbor risk features
            - Uses neighbor attention heads for risk aggregation
    
    Args:
        args (dict): Configuration dictionary from parse_args()
    """
    
    # ============ MCNN (CNN-based baseline) ============
    if args['method'] == 'mcnn':
        from methods.mcnn.mcnn_main import mcnn_main
        base_load_data(args)
        mcnn_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            lr=args['lr'],
            device=args['device']
        )
    
    # ============ STAN 2D (Attention-based) ============
    elif args['method'] == 'stan_2d':
        from methods.stan.stan_2d_main import stan_main
        base_load_data(args)
        stan_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            mode='2d',
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )
    
    # ============ STAN 3D (Attention-based) ============
    elif args['method'] == 'stan':
        from methods.stan.stan_main import stan_main
        base_load_data(args)
        stan_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            mode='3d',
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )

    # ============ STAGN (Graph + Attention + CNN) ============
    elif args['method'] == 'stagn':
        from methods.stagn.stagn_main import stagn_main, load_stagn_data
        
        # Load data and construct graph
        # Graph: Source → Target transaction edges with Amount+Location features
        features, labels, g = load_stagn_data(args)
        
        stagn_main(
            features,           # 2D feature matrices [N, 5, 8]
            labels,             # Binary labels
            args['test_size'],  # Test set ratio
            g,                  # Transaction graph
            mode='2d',
            epochs=args['epochs'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )
    
    # ============ GTAN (Graph Transformer) ============
    elif args['method'] == 'gtan':
        from methods.gtan.gtan_main import gtan_main, load_gtan_data
        
        # Load data and construct graph
        # Graph: edges between transactions sharing Source/Target/Location/Type
        feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
            args['dataset'], args['test_size']
        )
        
        gtan_main(
            feat_data,      # 127 temporal features
            g,              # Transaction similarity graph
            train_idx,      # Training indices
            test_idx,       # Test indices
            labels,         # Labels (0/1/2 for normal/fraud/unlabeled)
            args,           # Full config
            cat_features    # Categorical feature names
        )
    
    # ============ RGTAN (Risk-aware GTAN) ============
    elif args['method'] == 'rgtan':
        from methods.rgtan.rgtan_main import rgtan_main, loda_rgtan_data
        
        # Load data with additional neighbor risk features
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size']
        )
        
        rgtan_main(
            feat_data,          # 127 temporal features
            g,                  # Transaction similarity graph
            train_idx,          # Training indices
            test_idx,           # Test indices
            labels,             # Labels
            args,               # Full config
            cat_features,       # Categorical features
            neigh_features,     # 6 neighbor risk statistics
            nei_att_head=args['nei_att_heads'][args['dataset']]  # Attention heads for dataset
        )
    
    # ============ HOGRL (High-Order Graph RL) ============
    elif args['method'] == 'hogrl':
        from methods.hogrl.hogrl_main import hogrl_main
        hogrl_main(args)
    
    else:
        raise NotImplementedError(f"Unsupported method: {args['method']}")


if __name__ == "__main__":
    main(parse_args())
