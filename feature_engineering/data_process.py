"""
Data Processing - Graph Construction and Neighbor Features
============================================================

This file handles data processing for the three supported datasets:
    1. S-FFSD (Simulated Financial Fraud Semi-supervised Dataset)
    2. YelpChi (Yelp Review Fraud)
    3. Amazon (Amazon Review Fraud)

Key Functions:
    - featmap_gen(): Generate 127 temporal features for S-FFSD
    - sparse_to_adjlist(): Convert sparse matrix to adjacency list
    - k_neighs(): Find k-hop neighbors for a node
    - count_risk_neighs(): Count fraudulent neighbors
    - feat_map(): Generate neighbor risk statistics

Output Files Generated:
    - graph-S-FFSD.bin: DGL graph for S-FFSD
    - graph-yelp.bin: DGL graph for YelpChi
    - graph-amazon.bin: DGL graph for Amazon
    - *_neigh_feat.csv: Neighbor risk statistics for RGTAN

The neighbor risk features (used by RGTAN) include:
    - degree: Node in-degree
    - riskstat: Number of fraud neighbors
    - 1hop_degree, 2hop_degree: 1-hop and 2-hop degree stats
    - 1hop_riskstat, 2hop_riskstat: 1-hop and 2-hop risk stats
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Data directory path
DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")


def featmap_gen(tmp_df=None):
    """
    Generate 127 temporal features for S-FFSD dataset
    
    Creates aggregate statistics over 15 time windows for each transaction.
    
    Time Windows (in number of prior transactions):
        [2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000]
    
    Features Generated per Window (8 features × 15 windows = 120):
        1. trans_at_avg_T: Average transaction amount
        2. trans_at_totl_T: Total transaction amount
        3. trans_at_std_T: Standard deviation of amounts
        4. trans_at_bias_T: Current amount - average
        5. trans_at_num_T: Transaction count
        6. trans_target_num_T: Unique targets count
        7. trans_location_num_T: Unique locations count
        8. trans_type_num_T: Unique transaction types count
    
    Plus original 7 raw fields = 127 total features
    
    Args:
        tmp_df (DataFrame): Raw S-FFSD transaction data
        
    Returns:
        DataFrame: Feature-engineered data with 127 columns
    """
    # 15 time windows for feature aggregation
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        
        # Generate features for each time window
        for length, tname in zip(time_span, time_name):
            # Filter transactions within time window
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            
            # ============ AMOUNT STATISTICS ============
            new_df['trans_at_avg_{}'.format(tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(tname)] = temp_amt - correct_data['Amount'].mean()
            
            # ============ COUNT STATISTICS ============
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(correct_data.Type.unique())
            
        post_fe.append(new_df)
        
    return pd.DataFrame(post_fe)


def sparse_to_adjlist(sp_matrix, filename):
    """
    Convert sparse matrix to adjacency list format and save
    
    This converts scipy sparse matrices (from .mat files) to Python
    dict-based adjacency lists for efficient graph construction.
    
    Args:
        sp_matrix: Scipy sparse matrix (adjacency matrix)
        filename (str): Output pickle file path
    """
    # Add self-loops to adjacency matrix
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    
    # Create adjacency list (using set for O(1) lookup)
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    
    for index, node in enumerate(edges[0]):
        # Add bidirectional edges
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
        
    # Save to pickle file
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def set_seed(seed):
    """
    Set random seeds for reproducibility
    
    Sets seeds for: random, numpy, torch (CPU & CUDA)
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def MinMaxScaling(data):
    """
    Apply Min-Max scaling to normalize data to [0, 1]
    
    Args:
        data: Array-like data to scale
        
    Returns:
        Scaled data in range [0, 1]
    """
    mind, maxd = data.min(), data.max()
    return (data - mind) / (maxd - mind)


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1
) -> torch.Tensor:
    """
    Find k-hop neighbors of a node in the graph
    
    This function is used to compute neighbor risk statistics for RGTAN.
    It can filter to return only fraudulent (risk) neighbors.
    
    Args:
        graph (dgl.DGLGraph): The transaction graph
        center_idx (int): Index of the center node
        k (int): Number of hops (1 or 2)
        where (str): Direction - "in" for predecessors, "out" for successors
        choose_risk (bool): If True, only return fraud neighbors
        risk_label (int): Label value for fraud nodes (default: 1)
        
    Returns:
        Tensor: Indices of k-hop neighbors (optionally filtered by fraud label)
    """
    target_idxs: torch.Tensor
    
    if k == 1:
        # 1-hop neighbors
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        # 2-hop neighbors (excluding 1-hop and center)
        if where == "in":
            # Get 2-hop in-subgraph
            subg_in = dgl.khop_in_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            # Remove 1-hop neighbors (keep only 2-hop)
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_out.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    # Get labels of neighbors
    neigh_labels = graph.ndata['label'][neigh_idxs]
    
    # Filter to fraud neighbors if requested
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(
    graph: dgl.DGLGraph,
    risk_label: int = 1
) -> torch.Tensor:
    """
    Count number of fraudulent neighbors for each node
    
    This is used to compute the 'riskstat' feature - how many of a 
    node's neighbors are fraudulent.
    
    Args:
        graph (dgl.DGLGraph): The transaction graph
        risk_label (int): Label value for fraud nodes (default: 1)
        
    Returns:
        Tensor: Count of fraud neighbors for each node [num_nodes]
    """
    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)


def feat_map():
    """
    Generate neighbor feature map for all nodes
    
    For each node, computes:
        - 1hop_degree: Sum of degree for 1-hop neighbors
        - 2hop_degree: Sum of degree for 2-hop neighbors
        - 1hop_riskstat: Sum of risk stats for 1-hop neighbors
        - 2hop_riskstat: Sum of risk stats for 2-hop neighbors
    
    These features capture the neighborhood structure and fraud
    patterns around each node.
    
    Returns:
        tuple: (feature_tensor, feature_names)
            - feature_tensor: [num_nodes, 4]
            - feature_names: list of feature column names
    """
    tensor_list = []
    feat_names = []
    
    for idx in tqdm(range(graph.num_nodes())):
        # Get 1-hop and 2-hop neighbors
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        # Aggregate neighbor features
        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),  # 1hop_degree
            edge_feat[neighs_2_of_center, 0].sum().item(),  # 2hop_degree
            edge_feat[neighs_1_of_center, 1].sum().item(),  # 1hop_riskstat
            edge_feat[neighs_2_of_center, 1].sum().item(),  # 2hop_riskstat
        ])
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree",
                  "1hop_riskstat", "2hop_riskstat"]

    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names


if __name__ == "__main__":
    """
    Main preprocessing script
    
    This script processes all three datasets and generates:
        1. DGL graph files (.bin)
        2. Neighbor risk feature files (.csv)
    
    Run this script once before training GTAN/RGTAN models.
    """
    
    set_seed(42)

    # ================================================================
    # YELP DATASET PROCESSING
    # ================================================================
    # YelpChi contains restaurant/hotel review data with fraud labels
    # Graph relations: R-U-R (review-user-review), R-T-R (text), R-S-R (sentiment)
    
    print(f"processing YELP data...")
    yelp = loadmat(os.path.join(DATADIR, 'YelpChi.mat'))
    
    # Extract relation matrices
    net_rur = yelp['net_rur']  # Review-User-Review (same user posted)
    net_rtr = yelp['net_rtr']  # Review-Text-Review (similar text)
    net_rsr = yelp['net_rsr']  # Review-Sentiment-Review (same sentiment)
    yelp_homo = yelp['homo']   # Homogeneous combined graph

    # Save adjacency lists for each relation
    sparse_to_adjlist(net_rur, os.path.join(DATADIR, "yelp_rur_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(DATADIR, "yelp_rtr_adjlists.pickle"))
    sparse_to_adjlist(net_rsr, os.path.join(DATADIR, "yelp_rsr_adjlists.pickle"))
    sparse_to_adjlist(yelp_homo, os.path.join(DATADIR, "yelp_homo_adjlists.pickle"))

    # Create DGL graph
    data_file = yelp
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    
    with open(os.path.join(DATADIR, "yelp_homo_adjlists.pickle"), 'rb') as file:
        homo = pickle.load(file)
    file.close()
    
    # Convert adjacency list to edge arrays
    src = []
    tgt = []
    for i in homo:
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    src = np.array(src)
    tgt = np.array(tgt)
    
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-yelp.bin", [g])

    # ================================================================
    # AMAZON DATASET PROCESSING
    # ================================================================
    # Amazon contains product review data with fraud labels
    # Graph relations: U-P-U (same product), U-S-U (same seller), U-V-U (same view)
    
    print(f"processing AMAZON data...")
    amz = loadmat(os.path.join(DATADIR, 'Amazon.mat'))
    
    net_upu = amz['net_upu']  # User-Product-User (reviewed same product)
    net_usu = amz['net_usu']  # User-Seller-User (bought from same seller)
    net_uvu = amz['net_uvu']  # User-View-User (similar viewing patterns)
    amz_homo = amz['homo']    # Homogeneous combined graph

    sparse_to_adjlist(net_upu, os.path.join(DATADIR, "amz_upu_adjlists.pickle"))
    sparse_to_adjlist(net_usu, os.path.join(DATADIR, "amz_usu_adjlists.pickle"))
    sparse_to_adjlist(net_uvu, os.path.join(DATADIR, "amz_uvu_adjlists.pickle"))
    sparse_to_adjlist(amz_homo, os.path.join(DATADIR, "amz_homo_adjlists.pickle"))

    data_file = amz
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    
    with open(DATADIR + 'amz_homo_adjlists.pickle', 'rb') as file:
        homo = pickle.load(file)
    file.close()
    
    src = []
    tgt = []
    for i in homo:
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    src = np.array(src)
    tgt = np.array(tgt)
    
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-amazon.bin", [g])

    # ================================================================
    # S-FFSD DATASET PROCESSING
    # ================================================================
    # S-FFSD: Simulated Financial Fraud Semi-supervised Dataset
    # Graph: Transactions connected if they share Source/Target/Location/Type
    
    print(f"processing S-FFSD data...")
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    
    # Generate 127 temporal features
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    data = data.reset_index(drop=True)
    out = []
    alls = []  # All source nodes
    allt = []  # All target nodes
    
    # ============ GRAPH CONSTRUCTION ============
    # Connect transactions that share Source, Target, Location, or Type
    pair = ["Source", "Target", "Location", "Type"]
    for column in pair:
        src, tgt = [], []
        edge_per_trans = 3  # Connect to 3 nearest transactions in each group
        
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")  # Order by time
            df_len = len(c_df)
            sorted_idxs = c_df.index
            
            # Connect each transaction to next 3 transactions in same group
            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)
        
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))
    
    # Encode categorical features as integers
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)
        
    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-S-FFSD.bin", [g])

    # ================================================================
    # GENERATE NEIGHBOR RISK FEATURES (for RGTAN)
    # ================================================================
    # These features capture fraud patterns in node neighborhoods
    
    for file_name in ['S-FFSD', 'yelp', 'amazon']:
        print(f"Generating neighbor risk-aware features for {file_name} dataset...")
        graph = dgl.load_graphs(DATADIR + "graph-" + file_name + ".bin")[0][0]
        graph: dgl.DGLGraph
        print(f"graph info: {graph}")

        # ============ BASE FEATURES ============
        edge_feat: torch.Tensor
        degree_feat = graph.in_degrees().unsqueeze_(1).float()  # Node degree
        risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()  # Fraud neighbor count

        origin_feat_name = []
        edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
        origin_feat_name = ['degree', 'riskstat']

        # ============ K-HOP NEIGHBOR FEATURES ============
        # Generate 1-hop and 2-hop neighbor statistics
        features_neigh, feat_names = feat_map()

        # Concatenate all features
        features_neigh = torch.cat(
            (edge_feat, features_neigh), dim=1
        ).numpy()
        feat_names = origin_feat_name + feat_names
        features_neigh[np.isnan(features_neigh)] = 0.

        # ============ NORMALIZE AND SAVE ============
        output_path = DATADIR + file_name + "_neigh_feat.csv"
        features_neigh = pd.DataFrame(features_neigh, columns=feat_names)
        scaler = StandardScaler()
        features_neigh = pd.DataFrame(
            scaler.fit_transform(features_neigh), 
            columns=features_neigh.columns
        )

        features_neigh.to_csv(output_path, index=False)
