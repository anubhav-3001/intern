"""
RGTAN Label Propagation Attention (LPA) Helper Functions
=========================================================

This file contains helper functions for preparing mini-batch data
for RGTAN training with Label Propagation Attention (LPA).

Key Difference from GTAN's LPA:
    - RGTAN also processes neighbor risk statistics
    - Neighbor statistics are masked to prevent information leakage
    - 1-hop and 2-hop neighbor risk stats are zeroed out for seed nodes

The masking is critical because:
    - If a seed node is fraud, its neighbors' risk stats would leak this info
    - By zeroing neighbor stats for nodes close to seeds, we prevent leakage
"""

import copy


def load_lpa_subtensor(
    node_feat,              # Full node features (|all|, feat_dim)
    work_node_feat,         # Categorical features dict
    neigh_feat: dict,       # Neighbor risk statistics dict
    neigh_padding_dict: dict,  # Padding values for neighbor features
    labels,                 # Full labels (|all|,)
    seeds,                  # Batch seed indices (|batch|,)
    input_nodes,            # All input node indices (|batch_all|,)
    device,                 # Device to move tensors to
    blocks,                 # DGL blocks for mini-batch
):
    """
    Load mini-batch data for RGTAN with Label Propagation Attention
    
    This function extends GTAN's load_lpa_subtensor by:
    1. Also loading neighbor risk statistics
    2. Masking neighbor stats to prevent information leakage
    
    Information Leakage Problem:
        If node A is a fraud node, its neighbors will have high risk stats.
        During training, if we try to predict node A, these high risk stats
        would leak the answer. To prevent this:
        
        - Zero out 1-hop_riskstat for 1-hop neighbors of seeds
        - Zero out 2-hop_riskstat for 2-hop neighbors of seeds
    
    Args:
        node_feat (Tensor): Full node feature matrix [N, in_feats]
        work_node_feat (dict): Categorical feature tensors {col: Tensor}
        neigh_feat (dict): Neighbor risk statistics {stat_name: Tensor}
            Keys may include: '1hop_riskstat', '2hop_riskstat', 'degree', 
                              'pagerank', 'rev_pagerank', 'kcore'
        neigh_padding_dict (dict): Padding values when stats are unknown
        labels (Tensor): Full label tensor [N]
        seeds (Tensor): Indices of nodes being predicted in this batch
        input_nodes (Tensor): All nodes needed (seeds + k-hop neighbors)
        device (str): Device to move tensors to ('cpu' or 'cuda')
        blocks (list): DGL blocks representing the mini-batch subgraph
        
    Returns:
        tuple:
            - batch_inputs (Tensor): Features for input nodes [num_input, in_feats]
            - batch_work_inputs (dict): Categorical features for input nodes
            - batch_neighstat_inputs (dict): Neighbor stats for input nodes (masked)
            - batch_labels (Tensor): True labels for seed nodes
            - propagate_labels (Tensor): Labels with seeds masked as unlabeled
    
    Example of masking:
        If we're predicting nodes [A, B] (seeds) and the graph is:
            A ← C ← E
            B ← D ← F
        
        Then:
            - C, D are 1-hop neighbors: zero out their 1hop_riskstat
            - E, F are 2-hop neighbors: zero out their 2hop_riskstat
            
        This prevents the model from "cheating" by using neighbor risk stats
        that are computed from the very labels we're trying to predict.
    """
    
    # ============ MASKING NEIGHBOR RISK STATS ============
    # Prevent information leakage by zeroing stats that could reveal seed labels
    
    # Mask 1-hop neighbor risk statistics
    # blocks[-2].dstdata['_ID'] gives indices of 1-hop neighbors
    if "1hop_riskstat" in neigh_feat.keys() and len(blocks) >= 2:
        nei_hop1 = blocks[-2].dstdata['_ID']
        neigh_feat['1hop_riskstat'][nei_hop1] = 0
    
    # Mask 2-hop neighbor risk statistics
    # blocks[-3].dstdata['_ID'] gives indices of 2-hop neighbors
    if "2hop_riskstat" in neigh_feat.keys() and len(blocks) >= 3:
        nei_hop2 = blocks[-3].dstdata['_ID']
        neigh_feat['2hop_riskstat'][nei_hop2] = 0

    # ============ EXTRACT BATCH DATA ============
    
    # Get numerical features for all input nodes
    batch_inputs = node_feat[input_nodes].to(device)
    
    # Get categorical features for input nodes (exclude 'labels')
    batch_work_inputs = {
        i: work_node_feat[i][input_nodes].to(device) 
        for i in work_node_feat if i not in {"labels"}
    }

    # ============ EXTRACT NEIGHBOR STATISTICS ============
    batch_neighstat_inputs = None
    if neigh_feat:
        batch_neighstat_inputs = {
            col: neigh_feat[col][input_nodes].to(device) 
            for col in neigh_feat.keys()
        }

    # ============ LABEL PROPAGATION SETUP ============
    
    # True labels for seeds (for computing loss)
    batch_labels = labels[seeds].to(device)
    
    # Create labels for propagation (semi-supervised learning)
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]  # Labels for all input nodes
    
    # MASK SEED LABELS AS UNLABELED (index 2)
    # This is the core of LPA - we hide the labels we're trying to predict
    # The first `len(seeds)` entries in input_nodes are the seed nodes
    propagate_labels[:seeds.shape[0]] = 2  # 2 = unlabeled (padding_idx)
    
    return (
        batch_inputs,           # Node features [input_size, 127]
        batch_work_inputs,      # Categorical embeddings
        batch_neighstat_inputs, # Neighbor risk stats (masked) 
        batch_labels,           # True labels for loss
        propagate_labels.to(device)  # Masked labels for semi-supervised
    )
