"""
RGTAN (Risk-aware Graph Temporal Attention Network) Model
==========================================================

This file implements the RGTAN model which extends GTAN by adding
Risk-aware Neighbor Statistics for improved fraud detection.

Key Difference from GTAN:
    GTAN uses 127 temporal features
    RGTAN uses 127 temporal features + 6 neighbor risk statistics = 133 features

Architecture Overview:
    Input (127 features) + Neighbor Risk Stats (6) → TransEmbedding + 1D CNN
            ↓
    2× TransformerConv (4 heads each)
            ↓
    MLP Classifier → Fraud/Normal

Key Components:
    1. PosEncoding - Positional encoding for temporal features
    2. Tabular1DCNN2 - 1D CNN for processing neighbor risk statistics
    3. TransEmbedding - Extended with neighbor feature attention
    4. TransformerConv - Graph Transformer Convolution layer
    5. RGTAN - Main model combining all components

The 6 Neighbor Risk Statistics:
    - 1hop_riskstat: Fraud ratio in 1-hop neighbors
    - 2hop_riskstat: Fraud ratio in 2-hop neighbors  
    - degree: Node degree (number of connections)
    - pagerank: PageRank centrality score
    - rev_pagerank: Reverse PageRank score
    - kcore: K-core decomposition number
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
import pandas as pd
from math import sqrt


class PosEncoding(nn.Module):
    """
    Positional Encoding for Temporal Features
    
    Implements sinusoidal positional encoding to capture temporal patterns.
    Same as GTAN's PosEncoding.
    
    Formula: PE(pos, 2i) = sin(pos / base^(2i/dim))
             PE(pos, 2i+1) = cos(pos / base^(2i/dim))
    """

    def __init__(self, dim, device, base=10000, bias=0):
        """
        Args:
            dim (int): Output encoding dimension
            device (str): Device to run on
            base (int): Base for frequency scaling
            bias (float): Phase bias for sine function
        """
        super(PosEncoding, self).__init__()
        
        p = []    # Frequency multipliers
        sft = []  # Phase shifts
        
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)  # Cosine
            else:
                sft.append(bias)  # Sine
                
        self.device = device
        self.sft = torch.tensor(sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        """Compute positional encoding for given positions"""
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransformerConv(nn.Module):
    """
    Graph Transformer Convolution Layer
    
    Implements scaled dot-product attention on graphs:
        Attention(Q, K, V) = softmax(QK^T / √d) V
    
    Features:
        - Multi-head attention
        - Gated skip connection
        - Layer normalization
    
    Same as GTAN's TransformerConv - see gtan_model.py for detailed comments.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Args:
            in_feats (int): Input feature dimension
            out_feats (int): Output feature dimension per head
            num_heads (int): Number of attention heads
            bias (bool): Use bias in linear layers
            allow_zero_in_degree (bool): Allow nodes with no incoming edges
            skip_feat (bool): Use skip connections
            gated (bool): Use gating mechanism
            layer_norm (bool): Apply layer normalization
            activation: Activation function
        """
        super(TransformerConv, self).__init__()
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        # Q, K, V projections
        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

        # Skip connection
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        else:
            self.skip_feat = None
            
        # Gating mechanism
        if gated:
            self.gate = nn.Linear(
                3 * self._out_feats * self._num_heads, 1, bias=bias)
        else:
            self.gate = None
            
        # Layer normalization
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats * self._num_heads)
        else:
            self.layer_norm = None
            
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Forward pass through TransformerConv
        
        Args:
            graph: DGL graph or block
            feat: Node features
            get_attention: Return attention weights
            
        Returns:
            Updated node features (and optionally attention weights)
        """
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('Zero in-degree nodes found. Add self-loops.')

        # Handle bipartite graphs
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # Compute Q, K, V
        q_src = self.lin_query(h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(h_src).view(-1, self._num_heads, self._out_feats)
        
        # Assign to graph
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        
        # Dot product attention
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Scaled softmax
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats ** 0.5)

        # Aggregate with attention weights
        graph.update_all(
            fn.u_mul_e('ft_v', 'sa', 'attn'),
            fn.sum('attn', 'agg_u')
        )

        rst = graph.dstdata['agg_u'].reshape(-1, self._out_feats * self._num_heads)

        # Skip connection with gating
        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class Tabular1DCNN2(nn.Module):
    """
    1D CNN for Processing Neighbor Risk Statistics
    
    This is the key component that differentiates RGTAN from GTAN.
    It processes the 6 neighbor risk statistics using a series of 
    1D convolutions to extract patterns.
    
    Architecture:
        Input (6 neighbor stats) → Dense → Reshape → 1D CNN layers → Output embeddings
    
    The CNN uses:
        - Group convolutions for efficiency
        - Residual connections for gradient flow
        - Multiple conv layers for hierarchical features
    
    Args:
        input_dim (int): Number of neighbor features (6)
        embed_dim (int): Embedding dimension
        K (int): Channel expansion factor
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        K: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim      # 6 neighbor features
        self.embed_dim = embed_dim       # Same as in_feats
        self.hid_dim = input_dim * embed_dim * 2
        self.cha_input = self.cha_output = input_dim
        self.cha_hidden = (input_dim * K) // 2
        self.sign_size1 = 2 * embed_dim
        self.sign_size2 = embed_dim
        self.K = K

        # ============ INITIAL PROJECTION ============
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(input_dim, self.hid_dim)

        # ============ 1D CNN LAYER 1 ============
        # Uses group convolution (each feature channel processed separately)
        self.bn_cv1 = nn.BatchNorm1d(self.cha_input)
        self.conv1 = nn.Conv1d(
            in_channels=self.cha_input,
            out_channels=self.cha_input * self.K,
            kernel_size=5,
            padding=2,
            groups=self.cha_input,  # Depthwise convolution
            bias=False
        )

        self.ave_pool1 = nn.AdaptiveAvgPool1d(self.sign_size2)

        # ============ 1D CNN LAYER 2 (with residual) ============
        self.bn_cv2 = nn.BatchNorm1d(self.cha_input * self.K)
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_input * self.K,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ============ 1D CNN LAYER 3 (channel reduction) ============
        self.bn_cv3 = nn.BatchNorm1d(self.cha_input * self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_input * (self.K // 2),
            kernel_size=3,
            padding=1,
            bias=True
        )

        # ============ RESIDUAL BLOCKS (6 layers) ============
        self.bn_cvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(6):
            self.bn_cvs.append(nn.BatchNorm1d(self.cha_input * (self.K // 2)))
            self.convs.append(nn.Conv1d(
                in_channels=self.cha_input * (self.K // 2),
                out_channels=self.cha_input * (self.K // 2),
                kernel_size=3,
                padding=1,
                bias=True
            ))

        # ============ FINAL PROJECTION ============
        self.bn_cv10 = nn.BatchNorm1d(self.cha_input * (self.K // 2))
        self.conv10 = nn.Conv1d(
            in_channels=self.cha_input * (self.K // 2),
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x):
        """
        Forward pass through 1D CNN for neighbor statistics
        
        Args:
            x (Tensor): Neighbor statistics [batch, 6]
            
        Returns:
            Tensor: Processed embeddings [batch, 6, embed_dim]
        """
        # Initial projection
        x = self.dropout1(self.bn1(x))
        x = nn.functional.celu(self.dense1(x))
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        # Conv layer 1 + pooling
        x = self.bn_cv1(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.ave_pool1(x)

        # Conv layer 2 with residual
        x_input = x
        x = self.dropout2(self.bn_cv2(x))
        x = nn.functional.relu(self.conv2(x))
        x = x + x_input  # Residual connection

        # Conv layer 3 (channel reduction)
        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))

        # 6 residual blocks
        for i in range(6):
            x_input = x
            x = self.bn_cvs[i](x)
            x = nn.functional.relu(self.convs[i](x))
            x = x + x_input  # Residual connection

        # Final projection
        x = self.bn_cv10(x)
        x = nn.functional.relu(self.conv10(x))

        return x


class TransEmbedding(nn.Module):
    """
    Extended Categorical Embedding Layer for RGTAN
    
    This extends GTAN's TransEmbedding by adding:
        1. Tabular1DCNN2 for processing neighbor risk statistics
        2. Multi-head attention for aggregating neighbor embeddings
    
    The neighbor risk statistics (6 features) are processed through:
        1D CNN → Multi-head Self-Attention → Linear projection → Output
    
    Args:
        df: DataFrame with feature data
        device (str): Device to run on
        dropout (float): Dropout rate
        in_feats_dim (int): Input feature dimension
        cat_features (list): Categorical feature names
        neigh_features (dict): Neighbor risk statistics
        att_head_num (int): Number of attention heads for neighbor features
        neighstat_uni_dim (int): Unified dimension for neighbor stats
    """

    def __init__(
        self,
        df=None,
        device='cpu',
        dropout=0.2,
        in_feats_dim=82,
        cat_features=None,
        neigh_features: dict = None,
        att_head_num: int = 4,
        neighstat_uni_dim=64
    ):
        super(TransEmbedding, self).__init__()
        
        # Positional encoding for time
        self.time_pe = PosEncoding(dim=in_feats_dim, device=device, base=100)

        # ============ CATEGORICAL EMBEDDINGS (same as GTAN) ============
        self.cat_table = nn.ModuleDict({
            col: nn.Embedding(max(df[col].unique()) + 1, in_feats_dim).to(device) 
            for col in cat_features if col not in {"Labels", "Time"}
        })

        # ============ NEIGHBOR STATISTICS PROCESSING (RGTAN SPECIFIC) ============
        # 1D CNN to process the 6 neighbor risk statistics
        if isinstance(neigh_features, dict):
            self.nei_table = Tabular1DCNN2(
                input_dim=len(neigh_features), 
                embed_dim=in_feats_dim
            )

        # ============ MULTI-HEAD ATTENTION FOR NEIGHBOR FEATURES ============
        # This aggregates the 6 neighbor feature embeddings
        self.att_head_num = att_head_num      # Number of attention heads
        self.att_head_size = int(in_feats_dim / att_head_num)  # Size per head
        self.total_head_size = in_feats_dim
        
        # Q, K, V projections for self-attention
        self.lin_q = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_k = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_v = nn.Linear(in_feats_dim, self.total_head_size)

        # Final projection and normalization
        self.lin_final = nn.Linear(in_feats_dim, in_feats_dim)
        self.layer_norm = nn.LayerNorm(in_feats_dim, eps=1e-8)

        # MLP to aggregate neighbor embeddings to scalar weights
        self.neigh_mlp = nn.Linear(in_feats_dim, 1)

        self.neigh_add_mlp = nn.ModuleList([
            nn.Linear(in_feats_dim, in_feats_dim) 
            for i in range(len(neigh_features.columns))
        ]) if isinstance(neigh_features, pd.DataFrame) else None

        # Label embedding (same as GTAN)
        self.label_table = nn.Embedding(3, in_feats_dim, padding_idx=2).to(device)
        
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.neigh_features = neigh_features
        
        # MLPs for categorical features
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats_dim, in_feats_dim) for i in range(len(cat_features))]
        )
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, cat_feat):
        """Get embeddings for categorical features"""
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        support = {
            col: self.emb_dict[col](cat_feat[col]) 
            for col in self.cat_features if col not in {"Labels", "Time"}
        }
        return support

    def transpose_for_scores(self, input_tensor):
        """Reshape tensor for multi-head attention"""
        new_x_shape = input_tensor.size()[:-1] + (self.att_head_num, self.att_head_size)
        input_tensor = input_tensor.view(*new_x_shape)
        return input_tensor.permute(0, 2, 1, 3)

    def forward_neigh_emb(self, neighstat_feat):
        """
        Process neighbor risk statistics with 1D CNN and multi-head attention
        
        This is the key difference from GTAN:
        1. Stack the 6 neighbor statistics
        2. Process with 1D CNN (Tabular1DCNN2)
        3. Apply multi-head self-attention
        4. Project to final embeddings
        
        Args:
            neighstat_feat (dict): {stat_name: Tensor}
            
        Returns:
            Tuple[Tensor, list]: Processed embeddings and column names
        """
        cols = neighstat_feat.keys()
        tensor_list = []
        for col in cols:
            tensor_list.append(neighstat_feat[col])
        neis = torch.stack(tensor_list).T  # [batch, 6]
        
        # Step 1: Process with 1D CNN
        input_tensor = self.nei_table(neis)  # [batch, 6, embed_dim]

        # Step 2: Multi-head self-attention
        # Q, K, V projections
        mixed_q_layer = self.lin_q(input_tensor)
        mixed_k_layer = self.lin_k(input_tensor)
        mixed_v_layer = self.lin_v(input_tensor)

        # Reshape for multi-head attention
        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        # Scaled dot-product attention
        att_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        att_scores = att_scores / sqrt(self.att_head_size)

        att_probs = nn.Softmax(dim=-1)(att_scores)
        context_layer = torch.matmul(att_probs, v_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.total_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        
        # Final projection and normalization
        hidden_states = self.lin_final(context_layer)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, cols

    def forward(self, cat_feat: dict, neighstat_feat: dict):
        """
        Forward pass through TransEmbedding
        
        Process:
        1. Get categorical feature embeddings (same as GTAN)
        2. Get neighbor risk statistic embeddings (RGTAN specific)
        3. Aggregate categorical embeddings by summing
        4. Aggregate neighbor embeddings with MLP
        
        Args:
            cat_feat (dict): Categorical feature tensors
            neighstat_feat (dict): Neighbor statistic tensors
            
        Returns:
            Tuple[Tensor, Tensor]: (categorical_embedding, neighbor_embedding)
        """
        # Categorical embeddings (same as GTAN)
        support = self.forward_emb(cat_feat)
        cat_output = 0
        nei_output = 0
        
        for i, k in enumerate(support.keys()):
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            cat_output = cat_output + support[k]  # Sum aggregation

        # Neighbor risk statistic embeddings (RGTAN specific)
        if neighstat_feat is not None:
            nei_embs, cols_list = self.forward_neigh_emb(neighstat_feat)
            # Project to scalar per neighbor feature, output [batch, 6]
            nei_output = self.neigh_mlp(nei_embs).squeeze(-1)

        return cat_output, nei_output


class RGTAN(nn.Module):
    """
    RGTAN (Risk-aware Graph Temporal Attention Network) - Main Model
    
    This extends GTAN by adding neighbor risk statistics to capture
    fraud patterns in graph neighborhoods.
    
    Key Differences from GTAN:
        1. Uses 127 + 6 = 133 input features (127 temporal + 6 neighbor stats)
        2. TransEmbedding includes 1D CNN for neighbor statistics
        3. Multi-head attention to aggregate neighbor feature embeddings
    
    Architecture:
        Input Features (127) + Neighbor Statistics (6)
                                ↓
        TransEmbedding (Categorical + 1D CNN for neighbor)
                                ↓
        2× TransformerConv (4 heads each)
                                ↓
        MLP Classifier
                                ↓
        Output (2 classes: fraud/normal)
    
    Args:
        in_feats (int): Input feature dimension (127)
        hidden_dim (int): Hidden dimension per head
        n_layers (int): Number of TransformerConv layers
        n_classes (int): Number of output classes
        heads (list): Attention heads per layer [4, 4]
        activation: Activation function
        skip_feat (bool): Use skip connections
        gated (bool): Use gating mechanism
        layer_norm (bool): Use layer normalization
        post_proc (bool): Use post-processing MLP
        n2v_feat (bool): Use categorical embeddings
        drop (list): Dropout rates [input, layer]
        ref_df: Reference DataFrame for embeddings
        cat_features (list): Categorical feature names
        neigh_features (dict): Neighbor risk statistics
        nei_att_head (int): Attention heads for neighbor features (9 for S-FFSD)
        device (str): Device to run on
    """

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 neigh_features=None,
                 nei_att_head=4,
                 device='cpu'):
        super(RGTAN, self).__init__()
        
        # Store configuration
        self.in_feats = in_feats            # 127 temporal features
        self.hidden_dim = hidden_dim        # 64
        self.n_layers = n_layers            # 2
        self.n_classes = n_classes          # 2
        self.heads = heads                  # [4, 4]
        self.activation = activation        # PReLU
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        
        # ============ FEATURE EMBEDDING (RGTAN EXTENDED) ============
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, 
                device=device, 
                in_feats_dim=in_feats, 
                cat_features=cat_features, 
                neigh_features=neigh_features, 
                att_head_num=nei_att_head
            )
            # Number of neighbor risk features (6)
            self.nei_feat_dim = len(neigh_features.keys()) if isinstance(
                neigh_features, dict) else 0
        else:
            self.n2v_mlp = lambda x: x
            self.nei_feat_dim = 0
            
        # ============ BUILD LAYERS ============
        self.layers = nn.ModuleList()
        
        # Layer 0: Label embedding (dimension includes neighbor features)
        # Total dimension = 127 + 6 = 133
        self.layers.append(nn.Embedding(
            n_classes + 1, in_feats + self.nei_feat_dim, padding_idx=n_classes))
        
        # Layer 1: Project features for label fusion
        self.layers.append(nn.Linear(
            self.in_feats + self.nei_feat_dim, self.hidden_dim * self.heads[0]))
        
        # Layer 2: Project label embeddings
        self.layers.append(nn.Linear(
            self.in_feats + self.nei_feat_dim, self.hidden_dim * self.heads[0]))
        
        # Layer 3: Process fused features
        self.layers.append(nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * self.heads[0]),
            nn.PReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden_dim * self.heads[0], in_feats + self.nei_feat_dim)
        ))

        # ============ TRANSFORMER CONV LAYERS ============
        # First layer takes 133 input features (127 + 6)
        self.layers.append(TransformerConv(
            in_feats=self.in_feats + self.nei_feat_dim,  # 133
            out_feats=self.hidden_dim,
            num_heads=self.heads[0],
            skip_feat=skip_feat,
            gated=gated,
            layer_norm=layer_norm,
            activation=self.activation
        ))

        # Additional layers
        for l in range(0, (self.n_layers - 1)):
            self.layers.append(TransformerConv(
                in_feats=self.hidden_dim * self.heads[l - 1],
                out_feats=self.hidden_dim,
                num_heads=self.heads[l],
                skip_feat=skip_feat,
                gated=gated,
                layer_norm=layer_norm,
                activation=self.activation
            ))
            
        # ============ CLASSIFIER ============
        if post_proc:
            self.layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                nn.BatchNorm1d(self.hidden_dim * self.heads[-1]),
                nn.PReLU(),
                nn.Dropout(self.drop),
                nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)
            ))
        else:
            self.layers.append(nn.Linear(
                self.hidden_dim * self.heads[-1], self.n_classes))

    def forward(self, blocks, features, labels, n2v_feat=None, neighstat_feat=None):
        """
        Forward pass through RGTAN model
        
        Key difference from GTAN: processes neighbor risk statistics
        and concatenates them to the feature vector.
        
        Process:
        1. Get categorical + neighbor embeddings from TransEmbedding
        2. Add categorical embeddings to features
        3. CONCATENATE neighbor embeddings (6 dims) to features → 133 dims
        4. Apply label embedding for semi-supervised learning
        5. Apply TransformerConv layers
        6. Classify
        
        Args:
            blocks (list): DGL blocks for mini-batch training
            features (Tensor): Node features [batch, 127]
            labels (Tensor): Node labels (2 = unlabeled)
            n2v_feat (dict): Categorical feature tensors
            neighstat_feat (dict): Neighbor risk statistics
            
        Returns:
            Tensor: Class logits [batch, n_classes]
        """
        # ============ STEP 1: Get embeddings ============
        if n2v_feat is None and neighstat_feat is None:
            h = features
        else:
            # Get categorical embeddings and neighbor embeddings
            cat_h, nei_h = self.n2v_mlp(n2v_feat, neighstat_feat)
            
            # Add categorical embeddings to features
            h = features + cat_h
            
            # CONCATENATE neighbor embeddings (this is the key RGTAN difference!)
            # Features: [batch, 127] → [batch, 133]
            if isinstance(nei_h, torch.Tensor):
                h = torch.cat([h, nei_h], dim=-1)

        # ============ STEP 2: Label embedding (semi-supervised) ============
        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed  # Residual

        # ============ STEP 3: TransformerConv layers ============
        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l + 4](blocks[l], h))

        # ============ STEP 4: Classification ============
        logits = self.layers[-1](h)

        return logits
