"""
STAGN (Spatial-Temporal Attention Graph Network) Model for Fraud Detection
============================================================================

This file implements the STAGN-2D model which combines:
1. Temporal Attention - to capture time-dependent patterns across transaction windows
2. 2D CNN - to extract spatial-temporal features from the 2D feature matrix
3. Graph Convolutional Network (GCN) - to learn from transaction graph structure

Architecture Overview:
    Input (N, 5, 8) → Temporal Attention → 2D CNN → Feature Fusion with GCN → Output

Where:
    N = number of transactions
    5 = features per time window (AvgAmount, TotalAmount, BiasAmount, Count, Entropy)
    8 = number of time windows [1, 3, 5, 10, 20, 50, 100, 500]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class TransactionGCN(torch.nn.Module):
    """
    Graph Convolutional Network for Transaction Graph
    
    This GCN operates on a bipartite Source→Target transaction graph where:
    - Nodes represent transaction endpoints (sources and targets)
    - Edges represent transactions with features (normalized Amount + one-hot Location)
    
    The model learns both node embeddings and edge embeddings simultaneously.
    
    Architecture:
        Layer 1: GraphConv(in_feats → hidden_feats) + Linear for edges
        Layer 2: GraphConv(hidden_feats → out_feats) + Linear for edges
        
    Edge update rule: edge_new = src_node + dst_node + edge_current
    """
    
    def __init__(self, in_feats, hidden_feats, out_feats, g, device):
        """
        Initialize the Transaction GCN
        
        Args:
            in_feats (int): Input feature dimension (edge feature size)
            hidden_feats (int): Hidden layer dimension (128 by default)
            out_feats (int): Output embedding dimension (8 by default)
            g (dgl.DGLGraph): The transaction graph
            device (str): Device to run on ('cpu' or 'cuda')
        """
        super().__init__()
        
        # Graph convolution layers for node features
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)
        
        # Linear layers for edge features (processed in parallel with node convolution)
        self.lin1 = nn.Linear(in_feats, hidden_feats)
        self.lin2 = nn.Linear(hidden_feats, out_feats)

        # Initialize node features with Xavier uniform (nodes don't have initial features)
        # Node features are initialized randomly since we only have edge features from data
        g.ndata['feat'] = torch.nn.init.xavier_uniform_(torch.empty(
            g.num_nodes(), g.edata['feat'].shape[1])).to(torch.float32).to(device)
        g.ndata['h'] = g.ndata['feat']  # h stores the current node hidden state
        g.edata['x'] = g.edata['feat']  # x stores the current edge hidden state

    def forward(self, g, h, e):
        """
        Forward pass through the Transaction GCN
        
        Args:
            g (dgl.DGLGraph): The transaction graph
            h (Tensor): Node features [num_nodes, in_feats]
            e (Tensor): Edge features [num_edges, in_feats]
            
        Returns:
            Tuple[Tensor, Tensor]: Updated node embeddings, updated edge embeddings
        """
        # Layer 1: Convolve node features and transform edge features
        h1 = torch.relu(self.conv1(g, h))  # Node convolution
        e1 = torch.relu(self.lin1(e))       # Edge transformation
        
        # Store in graph and update edges based on connected nodes
        g.ndata['h'] = h1
        g.edata['x'] = e1
        # Edge update: combine source node, destination node, and current edge
        g.apply_edges(
            lambda edges: {'x': edges.src['h'] + edges.dst['h'] + edges.data['x']})

        # Layer 2: Second convolution and transformation
        h2 = self.conv2(g, h1)
        e2 = torch.relu(self.lin2(e1))
        
        g.ndata['h'] = h2
        g.edata['x'] = e2
        g.apply_edges(
            lambda edges: {'x': edges.src['h'] + edges.dst['h'] + edges.data['x']})
        
        return g.ndata['h'], g.edata['x']


class stagn_2d_model(nn.Module):
    """
    STAGN-2D Model: Spatial-Temporal Attention Graph Network
    
    This model processes 2D feature matrices representing transactions with
    multiple features across multiple time windows. It combines three key components:
    
    1. TEMPORAL ATTENTION LAYER:
       - Computes attention weights between different time windows
       - Allows the model to focus on relevant time periods for fraud detection
       - Uses additive attention mechanism: score = V * tanh(W*x_i + U*x_j)
    
    2. 2D CNN LAYER:
       - Extracts local patterns from the attention-enhanced feature matrix
       - Uses 2x2 kernels to capture spatial-temporal correlations
       - Outputs 64 feature maps
    
    3. TRANSACTION GCN:
       - Learns from the Source→Target graph structure
       - Captures relationships between transaction parties
       - Provides graph-based features to complement temporal features
    
    Final prediction combines CNN features with graph embeddings through
    fully connected layers.
    """

    def __init__(
        self,
        time_windows_dim: int,
        feat_dim: int,
        num_classes: int,
        attention_hidden_dim: int,
        g: dgl.DGLGraph,
        filter_sizes: tuple = (2, 2),
        num_filters: int = 64,
        in_channels: int = 1,
        device="cpu"
    ) -> None:
        """
        Initialize the STAGN-2d model

        Args:
            time_windows_dim (int): Number of time windows (8 for S-FFSD)
            feat_dim (int): Number of features per time window (5 for S-FFSD)
            num_classes (int): Number of output classes (2: fraud/normal)
            attention_hidden_dim (int): Hidden dimension for attention mechanism
            g (dgl.DGLGraph): Transaction graph with edge features
            filter_sizes (tuple): CNN kernel size (default: 2x2)
            num_filters (int): Number of CNN output channels (default: 64)
            in_channels (int): Number of input channels (default: 1)
            device (str): Device to run on ('cpu' or 'cuda')
        """
        super().__init__()
        self.time_windows_dim = time_windows_dim  # 8 time windows
        self.feat_dim = feat_dim                   # 5 features
        self.num_classes = num_classes             # 2 classes (fraud/normal)
        self.attention_hidden_dim = attention_hidden_dim

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.graph = g.to(device)

        # ============ TEMPORAL ATTENTION LAYER PARAMETERS ============
        # These implement additive attention: score = V * tanh(W*query + U*key)
        # W transforms the current time window (query)
        self.attention_W = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        # U transforms other time windows (keys)
        self.attention_U = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        # V computes the final attention score
        self.attention_V = nn.Parameter(torch.Tensor(
            self.attention_hidden_dim, 1).uniform_(0., 1.))

        # ============ 2D CNN LAYER ============
        # Extracts spatial-temporal patterns from the attention-enhanced feature matrix
        self.conv = nn.Conv2d(
            in_channels=in_channels,     # 1 channel input
            out_channels=num_filters,    # 64 output channels
            kernel_size=filter_sizes,    # 2x2 kernel
            padding='same'               # Preserve spatial dimensions
        )

        # ============ FULLY CONNECTED LAYERS ============
        # First FC block: reduces CNN output dimensions
        self.flatten = nn.Flatten()
        self.linears1 = nn.Sequential(
            nn.LazyLinear(256),  # Lazy: infers input size automatically
            nn.ReLU(),
            nn.LazyLinear(24),
            nn.ReLU())

        # Final classifier
        self.linears2 = nn.LazyLinear(self.num_classes)

        # ============ GRAPH NEURAL NETWORK ============
        # GCN for learning from the transaction graph structure
        # Input: edge features, Output: 8-dim node/edge embeddings
        self.gcn = TransactionGCN(
            g.edata['feat'].shape[1], 128, 8, g, device)

    def attention_layer(self, X: torch.Tensor):
        """
        Temporal Attention Layer
        
        Computes attention weights between each time window and all other
        time windows, then combines them to create context-enhanced features.
        
        Args:
            X (Tensor): Input features [batch, time_windows, features]
            
        Returns:
            Tensor: Attention-enhanced features [batch, 1, time_windows, features*2]
        """
        self.output_att = []
        
        # Split input by time windows (axis=1)
        input_att = torch.split(X, 1, dim=1)
        
        # For each time window, compute attention-weighted context
        for index, x_i in enumerate(input_att):
            x_i = x_i.reshape(-1, self.feat_dim)  # [batch, features]
            
            # Compute context vector using attention over all other time windows
            c_i = self.attention(x_i, input_att, index)
            
            # Concatenate original features with context
            inp = torch.concat([x_i, c_i], axis=1)  # [batch, features*2]
            self.output_att.append(inp)

        # Reshape to [batch, time_windows, features*2]
        input_conv = torch.reshape(torch.concat(self.output_att, axis=1),
                                   [-1, self.time_windows_dim, self.feat_dim*2])

        # Add channel dimension for CNN: [batch, 1, time_windows, features*2]
        self.input_conv_expanded = torch.unsqueeze(input_conv, 1)

        return self.input_conv_expanded

    def cnn_layer(self, input: torch.Tensor):
        """
        2D CNN Layer
        
        Applies 2D convolution to extract spatial-temporal patterns.
        
        Args:
            input (Tensor): 3D or 4D tensor
            
        Returns:
            Tensor: CNN output [batch, num_filters, height, width]
        """
        # Ensure input has 4 dimensions (add channel if needed)
        if len(input.shape) == 3:
            self.input_conv_expanded = torch.unsqueeze(input, 1)
        elif len(input.shape) == 4:
            self.input_conv_expanded = input
        else:
            print("Wrong conv input shape!")

        # Apply convolution with ReLU activation
        self.input_conv_expanded = F.relu(self.conv(input))

        return self.input_conv_expanded

    def attention(self, x_i, x, index):
        """
        Additive Attention Mechanism
        
        Computes attention scores between query (x_i) and all keys (x),
        then returns the weighted sum of values.
        
        Attention formula: α_ij = softmax(V * tanh(W*x_i + U*x_j))
        Context: c_i = Σ α_ij * x_j (for j ≠ i)
        
        Args:
            x_i (Tensor): Query - current time window features [batch, features]
            x (List[Tensor]): Keys - all time window features
            index (int): Index of current time window (to exclude self-attention)
            
        Returns:
            Tensor: Context vector [batch, features]
        """
        e_i = []  # Attention scores
        c_i = []  # Context contributions

        # Compute attention scores for all time windows
        for i in range(len(x)):
            output = x[i]
            output = output.reshape(-1, self.feat_dim)
            
            # Additive attention: tanh(W*query + U*key)
            att_hidden = torch.tanh(torch.add(
                torch.matmul(x_i, self.attention_W),
                torch.matmul(output, self.attention_U)))
            
            # Project to scalar score
            e_i_j = torch.matmul(att_hidden, self.attention_V)
            e_i.append(e_i_j)

        # Concatenate and apply softmax to get attention weights
        e_i = torch.concat(e_i, axis=1)
        alpha_i = F.softmax(e_i, dim=1)
        alpha_i = torch.split(alpha_i, 1, 1)

        # Compute weighted sum (excluding self)
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue  # Skip self-attention
            else:
                output = output.reshape(-1, self.feat_dim)
                c_i_j = torch.multiply(alpha_i_j, output)
                c_i.append(c_i_j)

        # Sum all context contributions
        c_i = torch.reshape(torch.concat(c_i, axis=1),
                            [-1, self.time_windows_dim-1, self.feat_dim])
        c_i = torch.sum(c_i, dim=1)
        return c_i

    def forward(self, X_nume, g):
        """
        Forward Pass
        
        Complete forward pass through STAGN:
        1. Apply temporal attention to input features
        2. Extract patterns with 2D CNN
        3. Get graph embeddings from Transaction GCN
        4. Fuse CNN and graph features
        5. Classify as fraud or normal
        
        Args:
            X_nume (Tensor): Input features [batch, features, time_windows]
            g (dgl.DGLGraph): Transaction graph
            
        Returns:
            Tensor: Class logits [batch, num_classes]
        """
        # Step 1: Temporal Attention
        # Input: [batch, features, time_windows] 
        # Output: [batch, 1, time_windows, features*2]
        out = self.attention_layer(X_nume)

        # Step 2: 2D CNN
        # Output: [batch, 64, time_windows, features*2]
        out = self.cnn_layer(out)
        
        # Step 3: Graph Convolution
        # Get node and edge embeddings from transaction graph
        node_embs, edge_embs = self.gcn(g, g.ndata['feat'], g.edata['feat'])

        # Step 4: Extract edge features for each transaction
        src_nds, dst_nds = g.edges()  # Get source and destination nodes
        src_feat = g.ndata['h'][src_nds]  # Source node embeddings
        dst_feat = g.ndata['h'][dst_nds]  # Destination node embeddings
        
        # Stack [source, destination, edge] features for each transaction
        # Shape: [batch, 3, embedding_dim] → [batch, 3*embedding_dim]
        node_feats = torch.stack(
            [src_feat, dst_feat, edge_embs], dim=1).view(X_nume.shape[0], -1)
        
        # Step 5: Flatten CNN output and pass through FC layers
        out = self.flatten(out)
        out = self.linears1(out)
        
        # Step 6: Concatenate with graph features
        out = torch.cat([out, node_feats], dim=1)
        
        # Step 7: Final classification
        out = self.linears2(out)

        return out
