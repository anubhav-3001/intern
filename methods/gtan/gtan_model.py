"""
GTAN (Graph Temporal Attention Network) Model for Fraud Detection
===================================================================

This file implements the GTAN model which uses Graph Transformer Convolution
for semi-supervised fraud detection with categorical embeddings.

Architecture Overview:
    Input (127 features) → TransEmbedding → 2× TransformerConv → MLP → Output

Key Components:
    1. PosEncoding - Positional encoding for temporal features
    2. TransEmbedding - Categorical feature embedding layer
    3. TransformerConv - Graph Transformer Convolution layer
    4. GraphAttnModel - Main GTAN model combining all components

The model processes graph-structured data where:
    - Nodes = Transactions
    - Edges = Transactions sharing Source/Target/Location/Type
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np

# Default categorical features used in GTAN
cat_features = ["Target", "Type", "Location"]


class PosEncoding(nn.Module):
    """
    Positional Encoding for Temporal Features
    
    This implements sinusoidal positional encoding to capture temporal
    patterns in transaction data. Similar to Transformer positional encoding.
    
    Formula: PE(pos, 2i) = sin(pos / base^(2i/dim))
             PE(pos, 2i+1) = cos(pos / base^(2i/dim))
    
    Args:
        dim (int): Output encoding dimension (same as feature dim)
        device (str): Device to run on
        base (int): Base for frequency scaling (default: 10000)
        bias (float): Phase bias for sine function
    """

    def __init__(self, dim, device, base=10000, bias=0):
        super(PosEncoding, self).__init__()
        
        # Compute frequency scaling factors for each dimension
        p = []    # Frequency multipliers
        sft = []  # Phase shifts (0 for sin, π/2 for cos)
        
        for i in range(dim):
            b = (i - i % 2) / dim  # Groups pairs of dimensions
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)  # Cosine (shifted sine)
            else:
                sft.append(bias)  # Sine
                
        self.device = device
        self.sft = torch.tensor(sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        """
        Compute positional encoding for given positions
        
        Args:
            pos: Position values (e.g., timestamps)
            
        Returns:
            Tensor: Positional encoding [batch, dim]
        """
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransEmbedding(nn.Module):
    """
    Categorical Feature Embedding Layer (TransEmbedding)
    
    Converts categorical features (Target, Location, Type) into dense
    embeddings and processes them through MLPs.
    
    Key components:
        1. Embedding tables for each categorical feature
        2. Label embedding (for semi-supervised learning)
        3. MLP layers to process each embedding
        4. Sum aggregation of all embeddings
    
    Args:
        df: DataFrame with feature data (to determine embedding sizes)
        device (str): Device to run on
        dropout (float): Dropout rate
        in_feats (int): Input feature dimension
        cat_features (list): List of categorical feature column names
    """

    def __init__(self, df=None, device='cpu', dropout=0.2, in_feats=82, cat_features=None):
        super(TransEmbedding, self).__init__()
        
        # Positional encoding for time features
        self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)
        
        # ============ CATEGORICAL EMBEDDINGS ============
        # Create an embedding table for each categorical feature
        # Size determined by number of unique values in each column
        self.cat_table = nn.ModuleDict({
            col: nn.Embedding(max(df[col].unique()) + 1, in_feats).to(device) 
            for col in cat_features if col not in {"Labels", "Time"}
        })
        
        # Label embedding for semi-supervised learning
        # 3 classes: 0=normal, 1=fraud, 2=unlabeled (padding)
        self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)
        
        # Storage for computed embeddings
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        
        # MLP layers to process each categorical embedding
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats, in_feats) for i in range(len(cat_features))]
        )
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, df):
        """
        Compute embeddings for all categorical features
        
        Args:
            df: Dictionary with categorical feature tensors
            
        Returns:
            dict: Embeddings for each feature
        """
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
            
        # Look up embedding for each categorical feature
        support = {
            col: self.emb_dict[col](df[col]) 
            for col in self.cat_features if col not in {"Labels", "Time"}
        }
        return support

    def forward(self, df):
        """
        Forward pass through TransEmbedding
        
        Process:
        1. Get embeddings for each categorical feature
        2. Apply dropout and MLP to each
        3. Sum all embeddings together
        
        Args:
            df: Dictionary with categorical feature tensors
            
        Returns:
            Tensor: Combined embedding [batch, in_feats]
        """
        support = self.forward_emb(df)
        output = 0
        
        for i, k in enumerate(support.keys()):
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            output = output + support[k]  # Sum aggregation
            
        return output


class TransformerConv(nn.Module):
    """
    Graph Transformer Convolution Layer
    
    This implements attention-based message passing on graphs using the
    Transformer attention mechanism:
        1. Compute Query, Key, Value projections
        2. Calculate attention scores: softmax(Q·K^T / √d)
        3. Aggregate neighbor features weighted by attention
        4. Apply gated skip connection and layer normalization
    
    Key features:
        - Multi-head attention for different representation subspaces
        - Gated skip connection: output = gate * skip + (1-gate) * attention
        - Layer normalization for stable training
    
    Args:
        in_feats (int): Input feature dimension
        out_feats (int): Output feature dimension per head
        num_heads (int): Number of attention heads
        bias (bool): Whether to use bias in linear layers
        allow_zero_in_degree (bool): Allow nodes with no incoming edges
        skip_feat (bool): Whether to use skip connections
        gated (bool): Whether to use gating mechanism
        layer_norm (bool): Whether to apply layer normalization
        activation: Activation function (default: PReLU)
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
        super(TransformerConv, self).__init__()
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        # ============ Q, K, V PROJECTIONS ============
        # Query projection for source nodes
        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        # Key projection for destination nodes
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        # Value projection for source nodes
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

        # ============ SKIP CONNECTION ============
        # Projects original features for residual addition
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        else:
            self.skip_feat = None
            
        # ============ GATING MECHANISM ============
        # Learns how much of skip vs attention to use
        # Input: [skip_feat, attention_output, skip_feat - attention_output]
        if gated:
            self.gate = nn.Linear(
                3 * self._out_feats * self._num_heads, 1, bias=bias)
        else:
            self.gate = None
            
        # ============ LAYER NORMALIZATION ============
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats * self._num_heads)
        else:
            self.layer_norm = None
            
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Forward pass through TransformerConv
        
        Implements scaled dot-product attention on graph:
            Attention(Q, K, V) = softmax(QK^T / √d) V
        
        Args:
            graph: DGL graph (or block for mini-batch)
            feat: Node features [num_nodes, in_feats]
            get_attention: Whether to return attention weights
            
        Returns:
            Tensor: Updated node features [num_dst_nodes, out_feats * num_heads]
            (optional) Tensor: Attention weights if get_attention=True
        """
        graph = graph.local_var()

        # Check for zero in-degree nodes
        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph. '
                               'Consider adding self-loops with `dgl.add_self_loop(g)`')

        # Handle bipartite graphs
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # ============ STEP 1: Compute Q, K, V ============
        # Query from source, Key from destination, Value from source
        q_src = self.lin_query(h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(h_src).view(-1, self._num_heads, self._out_feats)
        
        # Assign to graph for message passing
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        
        # ============ STEP 2: Compute attention scores ============
        # Dot product between query and key
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Scaled softmax attention: softmax(QK^T / √d)
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats ** 0.5)

        # ============ STEP 3: Aggregate with attention weights ============
        # Weighted sum of values
        graph.update_all(
            fn.u_mul_e('ft_v', 'sa', 'attn'),  # Multiply value by attention
            fn.sum('attn', 'agg_u')             # Sum at destination
        )

        # Reshape output: [num_dst, num_heads, out_feats] → [num_dst, num_heads * out_feats]
        rst = graph.dstdata['agg_u'].reshape(-1, self._out_feats * self._num_heads)

        # ============ STEP 4: Skip connection with gating ============
        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            
            if self.gate is not None:
                # Compute gate value from concatenation
                gate = torch.sigmoid(
                    self.gate(torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                # Gated combination
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                # Simple addition
                rst = skip_feat + rst

        # ============ STEP 5: Layer norm and activation ============
        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class GraphAttnModel(nn.Module):
    """
    GTAN (Graph Temporal Attention Network) - Main Model
    
    Complete architecture for fraud detection combining:
        1. TransEmbedding - Categorical feature embeddings
        2. Label embedding - For semi-supervised learning
        3. TransformerConv layers - Graph attention message passing
        4. MLP classifier - Final prediction
    
    The model uses Label Propagation Attention (LPA) where known labels
    are embedded and used to guide the learning process.
    
    Architecture:
        Input Features (127) + Categorical Embeddings + Label Embeddings
                                    ↓
                        TransformerConv Layer 1 (4 heads)
                                    ↓
                        TransformerConv Layer 2 (4 heads)
                                    ↓
                            MLP Classifier
                                    ↓
                        Output (2 classes: fraud/normal)
    
    Args:
        in_feats (int): Input feature dimension (127 for S-FFSD)
        hidden_dim (int): Hidden dimension per attention head
        n_layers (int): Number of TransformerConv layers
        n_classes (int): Number of output classes
        heads (list): Number of attention heads per layer
        activation: Activation function
        skip_feat (bool): Use skip connections
        gated (bool): Use gating mechanism
        layer_norm (bool): Use layer normalization
        post_proc (bool): Use post-processing MLP
        n2v_feat (bool): Use node2vec-style features (TransEmbedding)
        drop (list): Dropout rates [input_drop, layer_drop]
        ref_df: Reference DataFrame for embeddings
        cat_features (list): Categorical feature names
        nei_features: Neighbor features (not used in GTAN)
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
                 nei_features=None,
                 device='cpu'):
        super(GraphAttnModel, self).__init__()
        
        # Store configuration
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        
        # ============ CATEGORICAL EMBEDDING ============
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats=in_feats, cat_features=cat_features)
        else:
            self.n2v_mlp = lambda x: x
            
        # ============ BUILD LAYERS ============
        self.layers = nn.ModuleList()
        
        # Layer 0: Label embedding (for semi-supervised learning)
        # n_classes+1 to include padding index for unlabeled nodes
        self.layers.append(nn.Embedding(n_classes + 1, in_feats, padding_idx=n_classes))
        
        # Layer 1: Project features for label fusion
        self.layers.append(nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        
        # Layer 2: Project label embeddings for fusion
        self.layers.append(nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        
        # Layer 3: Process fused features (BatchNorm + PReLU + Dropout + Linear)
        self.layers.append(nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * self.heads[0]),
            nn.PReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden_dim * self.heads[0], in_feats)
        ))

        # ============ TRANSFORMER CONV LAYERS ============
        # First TransformerConv layer
        self.layers.append(TransformerConv(
            in_feats=self.in_feats,
            out_feats=self.hidden_dim,
            num_heads=self.heads[0],
            skip_feat=skip_feat,
            gated=gated,
            layer_norm=layer_norm,
            activation=self.activation
        ))

        # Additional TransformerConv layers
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
            # Full MLP classifier with BatchNorm
            self.layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                nn.BatchNorm1d(self.hidden_dim * self.heads[-1]),
                nn.PReLU(),
                nn.Dropout(self.drop),
                nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)
            ))
        else:
            # Simple linear classifier
            self.layers.append(nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes))

    def forward(self, blocks, features, labels, n2v_feat=None):
        """
        Forward pass through GTAN model
        
        Process:
        1. Compute categorical embeddings (TransEmbedding)
        2. Add embeddings to input features
        3. Compute label embeddings for semi-supervised learning
        4. Fuse features with label information
        5. Apply TransformerConv layers with dropout
        6. Classify with final MLP
        
        Args:
            blocks (list): DGL blocks for mini-batch training
            features (Tensor): Node features [batch, in_feats]
            labels (Tensor): Node labels (2 = unlabeled)
            n2v_feat (dict): Categorical feature tensors
            
        Returns:
            Tensor: Class logits [batch, n_classes]
        """
        # ============ STEP 1: Add categorical embeddings ============
        if n2v_feat is None:
            h = features
        else:
            h = self.n2v_mlp(n2v_feat)
            h = features + h  # Add to input features

        # ============ STEP 2: Label embedding for semi-supervised learning ============
        # This implements Label Propagation Attention (LPA)
        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed  # Residual connection

        # ============ STEP 3: Apply TransformerConv layers ============
        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l + 4](blocks[l], h))

        # ============ STEP 4: Final classification ============
        logits = self.layers[-1](h)

        return logits
