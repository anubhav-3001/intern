"""
GTAN Label Propagation Attention (LPA) Helper Functions
=========================================================

This file contains helper functions for preparing mini-batch data
for GTAN training with Label Propagation Attention (LPA).

LPA is a semi-supervised learning technique where:
- Known labels are embedded and used to guide learning
- During training, seed node labels are masked as "unlabeled"
- The model learns to predict labels using both features and graph structure
"""

import copy


def load_lpa_subtensor(node_feat, work_node_feat, labels, seeds, input_nodes, device):
    """
    Load mini-batch data for GTAN with Label Propagation Attention
    
    This function prepares the data for training with LPA by:
    1. Extracting features for input nodes (seeds + neighbors)
    2. Extracting categorical features for input nodes
    3. Creating propagation labels where seed nodes are masked as unlabeled
    
    The key insight of LPA is that during training, we hide the true labels
    of the nodes we're trying to predict (seeds), forcing the model to
    leverage graph structure and neighbor information.
    
    Args:
        node_feat (Tensor): Full node feature matrix [N, in_feats]
        work_node_feat (dict): Categorical feature tensors {col: Tensor}
        labels (Tensor): Full label tensor [N]
        seeds (Tensor): Indices of nodes in this batch (prediction targets)
        input_nodes (Tensor): All nodes needed for computation (seeds + neighbors)
        device (str): Device to move tensors to
        
    Returns:
        tuple:
            - batch_inputs (Tensor): Features for input nodes [num_input, in_feats]
            - batch_work_inputs (dict): Categorical features for input nodes
            - batch_labels (Tensor): True labels for seed nodes (for loss)
            - propagate_labels (Tensor): Labels for input nodes with seeds masked
    
    Example:
        If input_nodes = [seed1, seed2, neighbor1, neighbor2, neighbor3]
        And labels = [0, 1, 1, 0, 1] (0=normal, 1=fraud)
        
        Then propagate_labels = [2, 2, 1, 0, 1]
        (Seeds are masked as 2=unlabeled, neighbors keep their labels)
    """
    # Get features for all input nodes (seeds + neighbors needed for message passing)
    batch_inputs = node_feat[input_nodes].to(device)
    
    # Get categorical features for input nodes
    # Exclude "Labels" as it's handled separately
    batch_work_inputs = {
        i: work_node_feat[i][input_nodes].to(device) 
        for i in work_node_feat if i not in {"Labels"}
    }
    
    # True labels for seeds (used for computing loss)
    batch_labels = labels[seeds].to(device)
    
    # Create propagation labels for semi-supervised learning
    # Deep copy to avoid modifying original labels
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]
    
    # IMPORTANT: Mask seed node labels as "unlabeled" (index 2)
    # This forces the model to learn from neighbors rather than memorizing labels
    # Only the first `len(seeds)` nodes are seeds (rest are neighbors)
    propagate_labels[:seeds.shape[0]] = 2  # 2 = unlabeled
    
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels.to(device)
