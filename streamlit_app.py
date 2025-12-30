import os
import yaml
import torch
import numpy as np
import pandas as pd
import streamlit as st
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler
try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader

# RGTAN imports
from methods.rgtan.rgtan_model import RGTAN
from methods.rgtan.rgtan_lpa import load_lpa_subtensor as load_lpa_subtensor_rgtan
from methods.rgtan.rgtan_main import loda_rgtan_data

# GTAN imports
from methods.gtan.gtan_model import GraphAttnModel
from methods.gtan.gtan_lpa import load_lpa_subtensor as load_lpa_subtensor_gtan
from methods.gtan.gtan_main import load_gtan_data

# STAGN imports
from methods.stagn.stagn_2d import stagn_2d_model
from methods.stagn.stagn_main import load_stagn_data

# ============================================
# BENCHMARK METRICS FROM README
# ============================================
BENCHMARK_METRICS = {
    "STAGN": {"AUC": 0.7659, "F1": 0.6852, "AP": 0.3599},
    "GTAN": {"AUC": 0.8286, "F1": 0.7336, "AP": 0.6585},
    "RGTAN": {"AUC": 0.8461, "F1": 0.7513, "AP": 0.6939},
}

# ============================================
# MODEL DESCRIPTIONS
# ============================================
MODEL_INFO = {
    "RGTAN": {
        "full_name": "Risk-aware Graph Temporal Attention Network",
        "description": """
**RGTAN** extends GTAN by adding **Risk-aware Neighbor Statistics** to capture fraud patterns in graph neighborhoods.

**Key Features:**
- 🔹 127 temporal features + 6 neighbor risk features
- 🔹 TransformerConv with 4-head attention
- 🔹 1D CNN for processing neighbor statistics  
- 🔹 Multi-head attention for risk aggregation (9 heads for S-FFSD)
- 🔹 Gated skip connections + Layer normalization
        """,
        "architecture": """
```
Input (127 features) + Neighbor Risk (6 features)
         │
         ▼
   TransEmbedding Layer
   (Categorical + 1D CNN for Risk)
         │
         ▼
   2× TransformerConv (4 heads each)
         │
         ▼
   MLP Classifier → Fraud/Normal
```
        """,
        "features_used": ["Time", "Source", "Target", "Amount", "Location", "Type", "Neighbor Risk Stats"],
        "input_shape": "(N, 133)",
    },
    "GTAN": {
        "full_name": "Graph Temporal Attention Network",
        "description": """
**GTAN** uses **Graph Transformer Convolution** for semi-supervised fraud detection with categorical embeddings.

**Key Features:**
- 🔹 127 temporal aggregation features
- 🔹 TransformerConv with 4-head attention
- 🔹 Categorical embeddings for Target, Location, Type
- 🔹 Label Propagation Attention (semi-supervised)
- 🔹 Gated skip connections
        """,
        "architecture": """
```
Input (127 features)
         │
         ▼
   TransEmbedding Layer
   (Categorical embeddings)
         │
         ▼
   2× TransformerConv (4 heads each)
         │
         ▼
   MLP Classifier → Fraud/Normal
```
        """,
        "features_used": ["Time", "Source", "Target", "Amount", "Location", "Type"],
        "input_shape": "(N, 127)",
    },
    "STAGN": {
        "full_name": "Spatial-Temporal Attention Graph Network",
        "description": """
**STAGN** combines **Temporal Attention**, **2D CNN**, and **Graph Convolution** for fraud detection.

**Key Features:**
- 🔹 2D feature matrices (5 features × 8 time windows)
- 🔹 Temporal attention across time windows
- 🔹 2D CNN for spatial-temporal pattern extraction
- 🔹 Graph Convolution on Source→Target transaction graph
- 🔹 Edge features (Amount + Location)
        """,
        "architecture": """
```
Input (5×8 matrix)          Transaction Graph
         │                        │
         ▼                        ▼
   Temporal Attention        GraphConv (2 layers)
         │                        │
         ▼                        │
      2D CNN                      │
         │                        │
         └───────► Fusion ◄───────┘
                     │
                     ▼
              MLP Classifier → Fraud/Normal
```
        """,
        "features_used": ["Time", "Source", "Target", "Amount", "Location", "Type"],
        "input_shape": "(N, 5, 8)",
    }
}

# ============================================
# CACHED DATA LOADING FUNCTIONS
# ============================================
@st.cache_resource(show_spinner=False)
def load_config(model_name: str):
    cfg_path = os.path.join("config", f"{model_name.lower()}_cfg.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["device"] = "cpu"
    return cfg

@st.cache_resource(show_spinner=False)
def load_data(model_name: str, cfg):
    if model_name == "RGTAN":
        feat_df, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            cfg["dataset"], cfg["test_size"]
        )
        return feat_df, labels, g, cat_features, neigh_features, train_idx, test_idx
    elif model_name == "GTAN":
        feat_df, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
            cfg["dataset"], cfg["test_size"]
        )
        return feat_df, labels, g, cat_features, None, train_idx, test_idx
    else:
        # STAGN
        features, labels, g = load_stagn_data(cfg)
        from sklearn.model_selection import train_test_split
        idx_all = np.arange(features.shape[0])
        train_idx, test_idx = train_test_split(idx_all, test_size=cfg["test_size"], stratify=labels, random_state=2, shuffle=True)
        return features, labels, g, [], None, train_idx, test_idx

@st.cache_resource(show_spinner=False)
def build_rgtan_model(cfg, feat_df, cat_features, neigh_features):
    device = torch.device("cpu")
    model = RGTAN(
        in_feats=feat_df.shape[1],
        hidden_dim=cfg["hid_dim"]//4,
        n_classes=2,
        heads=[4]*cfg["n_layers"],
        activation=torch.nn.PReLU(),
        n_layers=cfg["n_layers"],
        drop=cfg["dropout"],
        device=str(device),
        gated=cfg["gated"],
        ref_df=feat_df,
        cat_features={k: None for k in cat_features},
        neigh_features={k: None for k in (neigh_features.columns if hasattr(neigh_features, 'columns') else [])} if neigh_features is not None else {},
        nei_att_head=cfg["nei_att_heads"][cfg["dataset"]]
    )
    model.to(device)
    return model, device

@st.cache_resource(show_spinner=False)
def build_stagn_model(cfg, features, _g):
    device = torch.device("cpu")
    model = stagn_2d_model(
        time_windows_dim=features.shape[2],
        feat_dim=features.shape[1],
        num_classes=2,
        attention_hidden_dim=cfg["attention_hidden_dim"],
        g=_g,
        device=str(device)
    ).to(device)
    return model, device

@st.cache_resource(show_spinner=False)
def build_gtan_model(cfg, feat_df, cat_features):
    device = torch.device("cpu")
    model = GraphAttnModel(
        in_feats=feat_df.shape[1],
        hidden_dim=cfg["hid_dim"]//4,
        n_classes=2,
        heads=[4]*cfg["n_layers"],
        activation=torch.nn.PReLU(),
        n_layers=cfg["n_layers"],
        drop=cfg["dropout"],
        device=str(device),
        gated=cfg["gated"],
        ref_df=feat_df,
        cat_features=cat_features
    ).to(device)
    return model, device

def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except TypeError:
        model.load_state_dict(state, strict=False)
        missing, unexpected = [], []
    if unexpected:
        st.warning(f"Ignored unexpected keys: {sorted(list(unexpected))[:5]}{' ...' if len(unexpected)>5 else ''}")
    if missing:
        st.warning(f"Missing keys: {sorted(list(missing))[:5]}{' ...' if len(missing)>5 else ''}")
    model.eval()

@st.cache_resource(show_spinner=False)
def prepare_static_tensors(feat_df, labels, cat_features, neigh_features, device):
    if isinstance(feat_df, pd.DataFrame):
        num_feat = torch.from_numpy(feat_df.values).float().to(device)
        cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}
    else:
        num_feat = torch.from_numpy(feat_df).float().to(device)
        cat_feat = {}
    nei_feat = {}
    if isinstance(neigh_features, pd.DataFrame):
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(device) for col in neigh_features.columns}
    if isinstance(labels, (pd.Series, pd.DataFrame)):
        labels_arr = labels.values if isinstance(labels, pd.Series) else labels.iloc[:, 0].values
    else:
        labels_arr = labels
    labels_t = torch.from_numpy(labels_arr).long().to(device)
    return num_feat, cat_feat, nei_feat, labels_t

@torch.no_grad()
def predict_one_rgtan(idx, model, g, cfg, num_feat, cat_feat, nei_feat, labels_t, device):
    g = g.to(device)
    seeds = torch.tensor([idx]).long().to(device)
    sampler = MultiLayerFullNeighborSampler(cfg["n_layers"])
    dataloader = NodeDataLoader(
        g, seeds, sampler, batch_size=1, shuffle=False, drop_last=False, num_workers=0, device=device
    )
    for input_nodes, batch_seeds, blocks in dataloader:
        batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor_rgtan(
            num_feat, cat_feat, nei_feat, {}, labels_t, batch_seeds, input_nodes, device, blocks
        )
        blocks = [b.to(device) for b in blocks]
        logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
        prob = torch.softmax(logits, dim=1)[0,1].item()
        pred = int(torch.argmax(logits, dim=1).item())
        return prob, pred
    return None, None

@torch.no_grad()
def predict_one_gtan(idx, model, g, cfg, num_feat, cat_feat, labels_t, device):
    g = g.to(device)
    seeds = torch.tensor([idx]).long().to(device)
    sampler = MultiLayerFullNeighborSampler(cfg["n_layers"])
    dataloader = NodeDataLoader(
        g, seeds, sampler, batch_size=1, shuffle=False, drop_last=False, num_workers=0, device=device
    )
    for input_nodes, batch_seeds, blocks in dataloader:
        batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor_gtan(
            num_feat, cat_feat, labels_t, batch_seeds, input_nodes, device
        )
        blocks = [b.to(device) for b in blocks]
        logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
        prob = torch.softmax(logits, dim=1)[0,1].item()
        pred = int(torch.argmax(logits, dim=1).item())
        return prob, pred
    return None, None

# ============================================
# UI HELPER FUNCTIONS
# ============================================
def render_risk_gauge(prob):
    """Render a colorful risk gauge for fraud probability."""
    if prob < 0.3:
        color = "#28a745"  # Green
        risk_level = "LOW RISK"
    elif prob < 0.6:
        color = "#ffc107"  # Yellow
        risk_level = "MEDIUM RISK"
    else:
        color = "#dc3545"  # Red
        risk_level = "HIGH RISK"
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {color} {prob*100}%, #e0e0e0 {prob*100}%); 
                height: 30px; border-radius: 15px; margin: 10px 0;">
    </div>
    <div style="text-align: center; font-size: 24px; font-weight: bold; color: {color};">
        {risk_level}: {prob:.1%}
    </div>
    """, unsafe_allow_html=True)

def render_metrics_comparison():
    """Render benchmark metrics comparison chart."""
    models = list(BENCHMARK_METRICS.keys())
    metrics = ["AUC", "F1", "AP"]
    
    # Create DataFrame for chart
    chart_data = pd.DataFrame({
        metric: [BENCHMARK_METRICS[m][metric] for m in models]
        for metric in metrics
    }, index=models)
    
    return chart_data

def get_dataset_stats(feat_df, labels, g, model_name):
    """Get dataset statistics."""
    if isinstance(labels, (pd.Series, pd.DataFrame)):
        labels_arr = labels.values if isinstance(labels, pd.Series) else labels.iloc[:, 0].values
    else:
        labels_arr = labels
    
    n_samples = len(labels_arr)
    n_fraud = int((labels_arr == 1).sum())
    n_normal = int((labels_arr == 0).sum())
    n_unlabeled = int((labels_arr == 2).sum())
    
    if hasattr(g, 'num_nodes') and hasattr(g, 'num_edges'):
        n_nodes = g.num_nodes()
        n_edges = g.num_edges()
    else:
        n_nodes = n_samples
        n_edges = 0
    
    if isinstance(feat_df, pd.DataFrame):
        n_features = feat_df.shape[1]
    else:
        n_features = f"{feat_df.shape[1]}×{feat_df.shape[2]}"
    
    return {
        "samples": n_samples,
        "features": n_features,
        "nodes": n_nodes,
        "edges": n_edges,
        "fraud": n_fraud,
        "normal": n_normal,
        "unlabeled": n_unlabeled,
        "fraud_ratio": n_fraud / (n_fraud + n_normal) if (n_fraud + n_normal) > 0 else 0
    }

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(
        page_title="AntiFraud - Fraud Detection Dashboard", 
        page_icon="🛡️", 
        layout="wide"
    )
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/fraud.png", width=80)
        st.title("🛡️ AntiFraud")
        st.markdown("**Financial Fraud Detection Framework**")
        st.markdown("---")
        
        # Model Selection
        model_name = st.radio(
            "🔧 Select Model",
            options=["RGTAN", "GTAN", "STAGN"],
            index=0,
            help="Choose a fraud detection model"
        )
        
        st.markdown("---")
        
        # Model Quick Info
        info = MODEL_INFO[model_name]
        st.markdown(f"**{info['full_name']}**")

        st.caption(f"📊 Input Shape: {info['input_shape']}")
        
        st.markdown("---")
        st.markdown("### 📈 Benchmark Metrics")
        metrics = BENCHMARK_METRICS[model_name]
        col1, col2, col3 = st.columns(3)
        col1.metric("AUC", f"{metrics['AUC']:.4f}")
        col2.metric("F1", f"{metrics['F1']:.4f}")
        col3.metric("AP", f"{metrics['AP']:.4f}")
    
    # ========== MAIN CONTENT ==========
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Inference", "📖 Model Info", "📈 Comparison", "ℹ️ Dataset"])
    
    # Load data and model
    models_dir = os.path.join("models")
    if model_name == "RGTAN":
        available_ckpts = [os.path.join(models_dir, f) for f in [
            "rgtan_best_fold1.pth", "rgtan_best_fold2.pth", "rgtan_best_fold3.pth", "rgtan_ckpt.pth"
        ] if os.path.exists(os.path.join(models_dir, f))]
    elif model_name == "GTAN":
        available_ckpts = sorted([os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith("gtan_") and f.endswith(".pth")])
    else:
        candidates = ["stagn_best.pth", "stagn_ckpt.pth"]
        available_ckpts = [os.path.join(models_dir, f) for f in candidates if os.path.exists(os.path.join(models_dir, f))]
    
    cfg = load_config(model_name)
    feat_df, labels, g, cat_features, neigh_features, train_idx, test_idx = load_data(model_name, cfg)
    
    if model_name == "RGTAN":
        model, device = build_rgtan_model(cfg, feat_df, cat_features, neigh_features)
    elif model_name == "GTAN":
        model, device = build_gtan_model(cfg, feat_df, cat_features)
    else:
        model, device = build_stagn_model(cfg, feat_df, g)
    
    # ========== TAB 1: INFERENCE ==========
    with tab1:
        st.header(f"🔍 {model_name} Fraud Detection")
        
        if not available_ckpts:
            st.error(f"No {model_name} checkpoint found in models/. Please train or place a checkpoint there.")
            st.stop()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ckpt_path = st.selectbox("📁 Select checkpoint", options=available_ckpts, index=0)
            
        with col2:
            if st.button("🔄 Load Checkpoint", use_container_width=True):
                load_checkpoint(model, ckpt_path, device)
                st.success(f"✅ Loaded: {os.path.basename(ckpt_path)}")
        
        st.markdown("---")
        
        st.subheader("🎯 Predict Transaction")
        st.caption("Enter a transaction index to predict fraud probability")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            idx = st.number_input("Transaction Index", min_value=0, max_value=int(len(feat_df)-1), value=0, step=1)
            predict_btn = st.button("🔮 Predict", use_container_width=True, type="primary")
        
        with col2:
            if predict_btn:
                load_checkpoint(model, ckpt_path, device)
                num_feat, cat_feat, nei_feat, labels_t = prepare_static_tensors(feat_df, labels, cat_features, neigh_features, device)
                
                with st.spinner("Computing prediction..."):
                    if model_name == "RGTAN":
                        prob, pred = predict_one_rgtan(int(idx), model, g, cfg, num_feat, cat_feat, nei_feat, labels_t, device)
                    elif model_name == "GTAN":
                        prob, pred = predict_one_gtan(int(idx), model, g, cfg, num_feat, cat_feat, labels_t, device)
                    else:
                        features = torch.from_numpy(feat_df).float().to(device)
                        features.transpose_(1, 2)
                        labels_t = torch.from_numpy(labels).long().to(device)
                        logits = model(features, g.to(device))
                        prob = torch.softmax(logits, dim=1)[int(idx), 1].item()
                        pred = int(torch.argmax(logits, dim=1)[int(idx)].item())
                
                if prob is not None:
                    render_risk_gauge(prob)
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Fraud Probability", f"{prob:.4f}")
                    col_b.metric("Prediction", "🚨 FRAUD" if pred == 1 else "✅ NORMAL")
                else:
                    st.error("Failed to compute prediction.")
        
        # Top-K Section
        st.markdown("---")
        st.subheader("🔝 Top-K Suspicious Transactions")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            k = st.slider("K", min_value=5, max_value=100, value=20, step=5)
            show_topk = st.button("Show Top-K", use_container_width=True)
        
        if show_topk and model_name in ["GTAN", "RGTAN"]:
            load_checkpoint(model, ckpt_path, device)
            num_feat, cat_feat, nei_feat, labels_t = prepare_static_tensors(feat_df, labels, cat_features, neigh_features, device)
            test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
            sampler = MultiLayerFullNeighborSampler(cfg["n_layers"])
            dataloader = NodeDataLoader(
                g.to(device), test_ind, sampler,
                use_ddp=False, device=device,
                batch_size=cfg["batch_size"], shuffle=False, drop_last=False, num_workers=0
            )
            probs = torch.zeros(len(feat_df), dtype=torch.float32, device=device)
            model.eval()
            
            with st.spinner("Computing Top-K..."):
                with torch.no_grad():
                    for input_nodes, seeds, blocks in dataloader:
                        blocks = [b.to(device) for b in blocks]
                        if model_name == "RGTAN":
                            batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor_rgtan(
                                num_feat, cat_feat, nei_feat, {}, labels_t, seeds, input_nodes, device, blocks
                            )
                            logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                        else:
                            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor_gtan(
                                num_feat, cat_feat, labels_t, seeds, input_nodes, device
                            )
                            logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
                        probs[seeds] = torch.softmax(logits, dim=1)[:, 1]
            
            test_probs = probs[test_ind].detach().cpu().numpy()
            test_indices = test_ind.detach().cpu().numpy()
            order = np.argsort(-test_probs)[:k]
            top_idx = test_indices[order]
            top_prob = test_probs[order]
            top_df = pd.DataFrame({"Index": top_idx, "Fraud Probability": top_prob})
            top_df["Risk Level"] = top_df["Fraud Probability"].apply(
                lambda x: "🔴 HIGH" if x >= 0.6 else ("🟡 MEDIUM" if x >= 0.3 else "🟢 LOW")
            )
            
            with col2:
                st.dataframe(top_df, use_container_width=True, hide_index=True)
    
    # ========== TAB 2: MODEL INFO ==========
    with tab2:
        st.header(f"📖 {model_name} - Model Details")
        
        info = MODEL_INFO[model_name]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(info["description"])
            
            st.subheader("🏗️ Architecture")
            st.code(info["architecture"], language="text")
        
        with col2:
            st.subheader("📊 Performance")
            metrics = BENCHMARK_METRICS[model_name]
            st.metric("AUC-ROC", f"{metrics['AUC']:.4f}", help="Area Under ROC Curve")
            st.metric("F1-Score", f"{metrics['F1']:.4f}", help="Macro F1 Score")
            st.metric("Avg Precision", f"{metrics['AP']:.4f}", help="Average Precision")
            
            st.subheader("📋 Features Used")
            for feat in info["features_used"]:
                st.markdown(f"- {feat}")
    
    # ========== TAB 3: MODEL COMPARISON ==========
    with tab3:
        st.header("📈 Model Comparison")
        
        st.subheader("Benchmark Results on S-FFSD Dataset")
        
        # Metrics table
        comparison_df = pd.DataFrame(BENCHMARK_METRICS).T
        comparison_df = comparison_df.reset_index()
        comparison_df.columns = ["Model", "AUC", "F1", "AP"]
        
        # Highlight best values
        st.dataframe(
            comparison_df.style.highlight_max(subset=["AUC", "F1", "AP"], color="lightgreen"),
            use_container_width=True,
            hide_index=True
        )
        
        # Bar chart
        st.subheader("📊 Visual Comparison")
        chart_data = render_metrics_comparison()
        st.bar_chart(chart_data)
        
        # Feature comparison
        st.subheader("🔧 Feature Comparison")
        feature_comparison = pd.DataFrame({
            "Feature": ["Input Shape", "Uses CNN", "Uses Transformer", "Risk Stats", "Edge Features"],
            "STAGN": ["5×8 (2D)", "✅", "❌", "❌", "✅"],
            "GTAN": ["127 (1D)", "❌", "✅", "❌", "❌"],
            "RGTAN": ["133 (1D)", "✅", "✅", "✅", "❌"],
        })
        st.dataframe(feature_comparison, use_container_width=True, hide_index=True)
    
    # ========== TAB 4: DATASET INFO ==========
    with tab4:
        st.header("ℹ️ Dataset Information")
        
        stats = get_dataset_stats(feat_df, labels, g, model_name)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Samples", f"{stats['samples']:,}")
        col2.metric("🔢 Features", f"{stats['features']}")
        col3.metric("🔗 Graph Nodes", f"{stats['nodes']:,}")
        col4.metric("➡️ Graph Edges", f"{stats['edges']:,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Class Distribution")
            class_df = pd.DataFrame({
                "Class": ["Normal (0)", "Fraud (1)", "Unlabeled (2)"],
                "Count": [stats["normal"], stats["fraud"], stats["unlabeled"]]
            })
            st.bar_chart(class_df.set_index("Class"))
            
            st.metric("Fraud Ratio (labeled)", f"{stats['fraud_ratio']:.2%}")
        
        with col2:
            st.subheader("📋 Raw S-FFSD Fields")
            raw_fields = pd.DataFrame({
                "Field": ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"],
                "Type": ["int32", "string", "string", "float32", "string", "string", "int32"],
                "Description": [
                    "Transaction timestamp",
                    "Sender account ID",
                    "Receiver account ID",
                    "Transaction amount",
                    "Location code",
                    "Transaction type",
                    "0=Normal, 1=Fraud, 2=Unlabeled"
                ]
            })
            st.dataframe(raw_fields, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
