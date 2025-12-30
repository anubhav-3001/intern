"""
STAGN Main Training Script
===========================

This file contains the main training and data loading functions for the STAGN model.
It handles:
1. Loading and preprocessing the S-FFSD dataset
2. Creating 2D feature matrices for temporal patterns
3. Constructing the Source→Target transaction graph
4. Training the STAGN model with class-weighted loss
5. Evaluating on test set and saving the best model

Key Functions:
    - load_stagn_data(): Loads dataset and creates graph
    - stagn_main(): Entry point for training
    - stagn_train_2d(): Main training loop
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from scipy.stats import zscore
from methods.stagn.stagn_2d import stagn_2d_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from feature_engineering.data_engineering import span_data_2d


def to_pred(logits: torch.Tensor) -> list:
    """
    Convert model logits to predicted class labels
    
    Args:
        logits (Tensor): Raw model outputs [batch, num_classes]
        
    Returns:
        list: Predicted class indices (0 or 1)
    """
    with torch.no_grad():
        # Apply softmax to get probabilities
        pred = F.softmax(logits, dim=1).cpu()
        # Take argmax to get predicted class
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()


def stagn_train_2d(
    features,
    labels,
    train_idx,
    test_idx,
    g,
    num_classes: int = 2,
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu",
    save_dir: str = "models"
):
    """
    Train the STAGN-2D model
    
    This function handles the complete training loop:
    1. Initialize model with 2D feature dimensions
    2. Compute class weights for imbalanced data
    3. Train with Adam optimizer and CrossEntropyLoss
    4. Track best model based on training loss
    5. Evaluate on test set and save best checkpoint
    
    Args:
        features (ndarray): 2D feature matrices [N, 5, 8]
        labels (ndarray): Ground truth labels [N]
        train_idx (list): Indices for training set
        test_idx (list): Indices for test set
        g (dgl.DGLGraph): Transaction graph
        num_classes (int): Number of classes (2: fraud/normal)
        epochs (int): Number of training epochs
        attention_hidden_dim (int): Hidden dimension for attention
        lr (float): Learning rate
        device (str): Device to train on
        save_dir (str): Directory to save checkpoints
    """
    # Move graph to device
    g = g.to(device)
    
    # ============ MODEL INITIALIZATION ============
    # Create STAGN-2D model with appropriate dimensions
    model = stagn_2d_model(
        time_windows_dim=features.shape[2],      # 8 time windows
        feat_dim=features.shape[1],               # 5 features
        num_classes=num_classes,                  # 2 classes
        attention_hidden_dim=attention_hidden_dim,
        g=g,
        device=device
    )
    model.to(device)

    # ============ DATA PREPARATION ============
    # Convert numpy arrays to tensors and move to device
    features = torch.from_numpy(features).to(device)
    # Transpose from [N, 5, 8] to [N, 8, 5] for attention
    features.transpose_(1, 2)
    labels = torch.from_numpy(labels).to(device)

    # ============ CLASS WEIGHTING ============
    # Compute inverse frequency weights to handle class imbalance
    # Fraud class is rare, so we weight its loss higher
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = (1 / counts) * len(labels) / len(unique_labels)

    # ============ OPTIMIZER & LOSS ============
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weights)  # Weighted loss

    # ============ TRAINING LOOP ============
    best_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass on all data (STAGN processes full batch)
        out = model(features, g)
        
        # Compute loss only on training indices
        loss = loss_func(out[train_idx], labels[train_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # ============ TRAINING METRICS ============
        pred = to_pred(out[train_idx])
        true = labels[train_idx].cpu().numpy()
        pred = np.array(pred)
        
        print(f"Epoch: {epoch}, loss: {loss:.4f}, "
              f"auc: {roc_auc_score(true, pred):.4f}, "
              f"F1: {f1_score(true, pred, average='macro'):.4f}, "
              f"AP: {average_precision_score(true, pred):.4f}")
        
        # Track best model based on lowest training loss
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            # Deep copy state dict to CPU
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # ============ TEST EVALUATION ============
    with torch.no_grad():
        out = model(features, g)
        pred = to_pred(out[test_idx])
        true = labels[test_idx].cpu().numpy()
        pred = np.array(pred)
        print(f"test set | "
              f"auc: {roc_auc_score(true, pred):.4f}, "
              f"F1: {f1_score(true, pred, average='macro'):.4f}, "
              f"AP: {average_precision_score(true, pred):.4f}")

    # ============ SAVE BEST MODEL ============
    if best_state is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(save_dir, "stagn_best.pth"))


def stagn_main(
    features,
    labels,
    test_ratio,
    g,
    mode: str = "2d",
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 0.003,
    device="cpu",
    save_dir: str = "models",
):
    """
    Main entry point for STAGN training
    
    Splits data into train/test and calls the appropriate training function.
    
    Args:
        features (ndarray): Feature matrices
        labels (ndarray): Ground truth labels
        test_ratio (float): Fraction of data for testing
        g (dgl.DGLGraph): Transaction graph
        mode (str): Training mode ('2d' for STAGN-2D)
        epochs (int): Number of training epochs
        attention_hidden_dim (int): Hidden dimension for attention
        lr (float): Learning rate
        device (str): Device to train on
        save_dir (str): Directory to save checkpoints
    """
    # Stratified split to maintain class distribution
    train_idx, test_idx = train_test_split(
        np.arange(features.shape[0]), 
        test_size=test_ratio, 
        stratify=labels
    )

    if mode == "2d":
        stagn_train_2d(
            features,
            labels,
            train_idx,
            test_idx,
            g,
            epochs=epochs,
            attention_hidden_dim=attention_hidden_dim,
            lr=lr,
            device=device,
            save_dir=save_dir
        )
    else:
        raise NotImplementedError("Not supported mode.")


def load_stagn_data(args: dict):
    """
    Load and preprocess S-FFSD dataset for STAGN model
    
    This function performs several preprocessing steps:
    1. Load raw S-FFSD.csv data
    2. Generate 2D feature matrices using span_data_2d()
       - Each transaction gets a (5, 8) matrix
       - 5 features: AvgAmount, TotalAmount, BiasAmount, Count, Entropy
       - 8 time windows: [1, 3, 5, 10, 20, 50, 100, 500]
    3. Build Source→Target transaction graph
       - Nodes: unique Source and Target account IDs
       - Edges: transactions (Source → Target)
       - Edge features: [z-score(Amount), one-hot(Location)]
    
    Args:
        args (dict): Configuration dictionary with 'test_size' key
        
    Returns:
        tuple: (features, labels, graph)
            - features: ndarray of shape (N, 5, 8)
            - labels: ndarray of shape (N,) with values 0 or 1
            - graph: DGLGraph with edge features
    """
    # ============ LOAD RAW DATA ============
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args.get('test_size', 0.2)
    
    # ============ GENERATE 2D FEATURES ============
    # Check if pre-computed features exist (for faster loading)
    if os.path.exists("data/features.npy"):
        features, labels = np.load("data/features.npy"), np.load("data/labels.npy")
    else:
        # Generate 2D feature matrices using span_data_2d
        # This creates (5, 8) matrices for each transaction
        features, labels = span_data_2d(feat_df)
        # Cache for future use
        np.save("data/features.npy", features)
        np.save("data/labels.npy", labels)

    # ============ FILTER UNLABELED DATA ============
    # Remove transactions with label=2 (unlabeled) for graph construction
    sampled_df = feat_df[feat_df['Labels'] != 2]
    sampled_df = sampled_df.reset_index(drop=True)

    # ============ BUILD TRANSACTION GRAPH ============
    # Encode Source and Target IDs as node indices
    all_nodes = pd.concat([sampled_df['Source'], sampled_df['Target']]).unique()
    encoder = LabelEncoder().fit(all_nodes)
    encoded_source = encoder.transform(sampled_df['Source'])
    encoded_tgt = encoder.transform(sampled_df['Target'])

    # ============ CREATE EDGE FEATURES ============
    # Edge features = [normalized Amount] + [one-hot Location]
    loc_enc = OneHotEncoder()
    # One-hot encode Location
    loc_feature = np.array(loc_enc.fit_transform(
        sampled_df['Location'].to_numpy()[:, np.newaxis]).todense())
    # Prepend z-score normalized Amount
    loc_feature = np.hstack(
        [zscore(sampled_df['Amount'].to_numpy())[:, np.newaxis], loc_feature])

    # ============ CREATE DGL GRAPH ============
    # Create directed graph: Source → Target
    g = dgl.DGLGraph()
    g.add_edges(
        encoded_source,      # Source nodes
        encoded_tgt,         # Target nodes
        data={"feat": torch.from_numpy(loc_feature).to(torch.float32)}  # Edge features
    )
    
    return features, labels, g
