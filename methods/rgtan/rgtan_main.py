"""
RGTAN Main Training Script
===========================

This file contains the main training loop and data loading functions
for the RGTAN (Risk-aware Graph Temporal Attention Network) model.

Key Functions:
    - rgtan_main(): Complete training pipeline with K-fold cross-validation
    - loda_rgtan_data(): Load and preprocess datasets (S-FFSD, YelpChi, Amazon)

Training Pipeline:
    1. Load preprocessed features (127 temporal + 6 neighbor risk stats)
    2. Construct transaction graph
    3. K-fold stratified cross-validation
    4. Mini-batch training with neighbor sampling
    5. Early stopping based on validation loss
    6. Test evaluation with AUC, F1, and AP metrics

Supported Datasets:
    - S-FFSD: Simulated Financial Fraud Semi-supervised Dataset
    - YelpChi: Yelp Review Fraud Dataset
    - Amazon: Amazon Review Fraud Dataset
"""

import os
from dgl.dataloading import MultiLayerFullNeighborSampler
try:
    from dgl.dataloading import NodeDataLoader
except ImportError:  # compatibility for DGL builds without NodeDataLoader symbol
    from dgl.dataloading import DataLoader as NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from scipy.io import loadmat
from tqdm import tqdm
from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN


def rgtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame, nei_att_head):
    """
    Main RGTAN Training Function
    
    Implements the complete training pipeline with:
    - K-fold stratified cross-validation
    - Mini-batch training with neighbor sampling
    - Early stopping based on validation loss
    - Test set evaluation
    
    Args:
        feat_df (DataFrame): Feature data with 127 columns
        graph (DGLGraph): Transaction graph with edges connecting related transactions
        train_idx (list): Indices of training nodes
        test_idx (list): Indices of test nodes
        labels (Series): Node labels (0=normal, 1=fraud, 2=unlabeled)
        args (dict): Training configuration from YAML
        cat_features (list): Categorical feature column names
        neigh_features (DataFrame): 6 neighbor risk statistics
        nei_att_head (int): Number of attention heads for neighbor features (9 for S-FFSD)
    
    Training Process:
        1. For each fold in K-fold CV:
            a. Create train/val split
            b. Setup mini-batch dataloaders
            c. Initialize RGTAN model
            d. Train with Adam optimizer + LR scheduler
            e. Track best model by validation loss
            f. Early stopping after patience epochs
        2. Evaluate best model on test set
        3. Report AUC, F1, and AP metrics
    """
    device = args['device']
    graph = graph.to(device)
    
    # ============ INITIALIZE PREDICTION STORAGE ============
    # Store predictions for all nodes (filled during validation/test)
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)  # Out-of-fold predictions
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)  # Test predictions
    
    # ============ K-FOLD CROSS VALIDATION SETUP ============
    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    
    # ============ CONVERT FEATURES TO TENSORS ============
    # Numerical features: 127-dimensional temporal aggregations
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    
    # Categorical features: Target, Location, Type (as integer indices)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}

    # ============ NEIGHBOR RISK FEATURES (RGTAN SPECIFIC) ============
    neigh_padding_dict = {}
    nei_feat = []
    if isinstance(neigh_features, pd.DataFrame):
        # 6 neighbor statistics: degree, riskstat, 1hop_*, 2hop_*
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
        
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # ============ K-FOLD TRAINING LOOP ============
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        
        # Convert fold indices to tensor
        trn_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device)
        val_ind = torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        # ============ MINI-BATCH DATALOADERS ============
        # MultiLayerFullNeighborSampler: samples ALL neighbors at each hop
        # This is needed for full message passing
        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = NodeDataLoader(
            graph,
            trn_ind,
            train_sampler,
            device=device,
            use_ddp=False,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = NodeDataLoader(
            graph,
            val_ind,
            val_sampler,
            use_ddp=False,
            device=device,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        
        # ============ MODEL INITIALIZATION ============
        # RGTAN: 127 features + 6 neighbor stats = 133 input dimensions
        model = RGTAN(
            in_feats=feat_df.shape[1],           # 127
            hidden_dim=args['hid_dim']//4,       # 256//4 = 64
            n_classes=2,                          # fraud/normal
            heads=[4]*args['n_layers'],           # [4, 4] attention heads
            activation=nn.PReLU(),
            n_layers=args['n_layers'],            # 2 TransformerConv layers
            drop=args['dropout'],                 # [0.2, 0.1]
            device=device,
            gated=args['gated'],                  # Use gated skip connections
            ref_df=feat_df,
            cat_features=cat_feat,
            neigh_features=nei_feat,              # 6 neighbor risk stats
            nei_att_head=nei_att_head             # 9 attention heads for S-FFSD
        ).to(device)
        
        # ============ OPTIMIZER SETUP ============
        # Scale learning rate by sqrt(batch_size / 1024)
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args['wd'])
        
        # Learning rate scheduler: reduce LR at milestones
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)

        # ============ EARLY STOPPING ============
        earlystoper = early_stopper(patience=args['early_stopping'], verbose=True)
        
        # ============ TRAINING EPOCHS ============
        for epoch in range(args['max_epochs']):
            train_loss_list = []
            model.train()
            
            # ============ MINI-BATCH TRAINING ============
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # Load batch data with Label Propagation Attention
                # input_nodes: all nodes needed for message passing
                # seeds: target nodes for this batch (what we predict)
                # blocks: mini-batch subgraph for each layer
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks
                )

                blocks = [block.to(device) for block in blocks]
                
                # Forward pass through RGTAN
                train_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs
                )
                
                # Mask out unlabeled nodes (label == 2)
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]

                # Compute loss and backpropagate
                train_loss = loss_fn(train_batch_logits, batch_labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                # Log training progress every 10 batches
                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(
                        train_batch_logits.clone().detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(
                                  epoch, step, np.mean(train_loss_list),
                                  average_precision_score(batch_labels.cpu().numpy(), score),
                                  tr_batch_pred.detach(),
                                  roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass

            # ============ VALIDATION ============
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()
            
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                        num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                        seeds, input_nodes, device, blocks
                    )

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs
                    )
                    
                    # Store predictions for out-of-fold evaluation
                    oof_predictions[seeds] = val_batch_logits
                    
                    # Mask unlabeled nodes
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    
                    val_loss_list = val_loss_list + loss_fn(val_batch_logits, batch_labels)
                    val_batch_pred = torch.sum(torch.argmax(
                        val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * torch.tensor(batch_labels.shape[0])
                    val_all_list = val_all_list + batch_labels.shape[0]
                    
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(
                                      epoch, step, val_loss_list/val_all_list,
                                      average_precision_score(batch_labels.cpu().numpy(), score),
                                      val_batch_pred.detach(),
                                      roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except:
                            pass

            # Check early stopping
            earlystoper.earlystop(val_loss_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
                
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        
        # ============ TEST EVALUATION ============
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = NodeDataLoader(
            graph,
            test_ind,
            test_sampler,
            use_ddp=False,
            device=device,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        
        # Save best model checkpoint
        save_dir = args.get('save_dir', 'models')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(earlystoper.best_model.state_dict(), os.path.join(save_dir, f"rgtan_best_fold{fold+1}.pth"))
        
        # Evaluate on test set
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks
                )

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs
                )
                test_predictions[seeds] = test_batch_logits
                
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    
    # ============ FINAL METRICS ============
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(y_target, torch.softmax(
        oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    
    # Final test metrics
    test_score = torch.softmax(test_predictions, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_predictions, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))


def loda_rgtan_data(dataset: str, test_size: float):
    """
    Load and Preprocess Dataset for RGTAN
    
    Supports three datasets:
        1. S-FFSD: Financial transaction fraud
        2. YelpChi: Yelp review fraud
        3. Amazon: Amazon review fraud
    
    Data Loading Process:
        1. Load raw features from CSV/MAT files
        2. Construct transaction graph
        3. Label encode categorical features
        4. Load neighbor risk statistics (if available)
        5. Split into train/test sets
    
    Args:
        dataset (str): 'S-FFSD', 'yelp', or 'amazon'
        test_size (float): Fraction of data for testing
        
    Returns:
        tuple: (feat_data, labels, train_idx, test_idx, graph, cat_features, neigh_features)
    """
    prefix = "data/"
    
    # ============ S-FFSD DATASET ============
    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]
        
        # Load preprocessed features (127 columns)
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        
        neigh_features = []
        
        # Filter to valid labels (0, 1, 2)
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        
        # ============ GRAPH CONSTRUCTION ============
        # Connect transactions sharing Source, Target, Location, or Type
        out = []
        alls = []  # Source nodes
        allt = []  # Target nodes
        pair = ["Source", "Target", "Location", "Type"]
        
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3  # Connect to 3 nearest neighbors
            
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                
                # Connect each transaction to next 3 in group
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
            
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))
        
        # ============ LABEL ENCODING ============
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
            
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        # Attach features and labels to graph
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        # Save graph
        graph_path = prefix + "graph-{}.bin".format(dataset)
        if not os.path.exists(graph_path):
            dgl.data.utils.save_graphs(graph_path, [g])
        else:
            print(f"Graph file {graph_path} already exists, skipping save")
            
        index = list(range(len(labels)))

        # Train/test split (stratified by label)
        train_idx, test_idx, y_train, y_test = train_test_split(
            index, labels, stratify=labels, test_size=0.6,
            random_state=2, shuffle=True
        )
        
        # ============ LOAD NEIGHBOR RISK FEATURES ============
        # These 6 features are what makes RGTAN different from GTAN
        feat_neigh = pd.read_csv(prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    # ============ YELP DATASET ============
    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        
        # Load from MATLAB file
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        
        # Load preprocessed adjacency list
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(
            index, labels, stratify=labels, test_size=test_size,
            random_state=2, shuffle=True
        )
        
        # Build graph from adjacency list
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
        
        graph_path = prefix + "graph-{}.bin".format(dataset)
        if not os.path.exists(graph_path):
            dgl.data.utils.save_graphs(graph_path, [g])
        else:
            print(f"Graph file {graph_path} already exists, skipping save")

        # Load neighbor features if available
        try:
            feat_neigh = pd.read_csv(prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    # ============ AMAZON DATASET ============
    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []
        
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        
        # Amazon dataset: skip first 3305 nodes (different structure)
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(
            index, labels[3305:], stratify=labels[3305:],
            test_size=test_size, random_state=2, shuffle=True
        )
        
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
        
        graph_path = prefix + "graph-{}.bin".format(dataset)
        if not os.path.exists(graph_path):
            dgl.data.utils.save_graphs(graph_path, [g])
        else:
            print(f"Graph file {graph_path} already exists, skipping save")
            
        try:
            feat_neigh = pd.read_csv(prefix + "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
