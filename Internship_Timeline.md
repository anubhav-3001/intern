# ANNEXURE – I

## INTERNSHIP TIMELINE (12 WEEKS)

**Week 1**  
Orientation to the Centre of Excellence in Cognitive Intelligent Systems for Sustainable Solutions (CISSS). Understanding internship objectives, problem statement, and high-level system requirements. Introduction to Graph Neural Networks (GNNs) and the HPCC Systems Big Data platform.

**Week 2**  
Literature review on deep learning-based financial fraud detection. Study of temporal transaction patterns, semi-supervised learning techniques, and recent papers on GNNs (e.g., Graph Attention Networks, TransformerConvs).

**Week 3**  
Exploratory Data Analysis (EDA) of the Financial Fraud Dataset. Understanding the schema, distribution of fraud vs. normal transactions, and analyzing key attributes like time, amount, and location to identify potential patterns.

**Week 4**  
System design and architecture planning. Designing the temporal feature engineering pipeline to create 2D feature matrices (Time Windows × Statistics) for capturing sequential user behaviors over multiple timeframes.

**Week 5**  
Development of graph-based feature engineering logic. Implementation of "Neighbor Risk-Aware" statistics to quantify network risk and design of graph topology connectivity (connecting users based on shared attributes).

**Week 6**  
Development of the GATE-Net (Graph Attention Temporal Embedding Network) model. Implementation of the 2D CNN module for temporal pattern extraction and integration with initial graph layers.

**Week 7**  
Development of the GAP-Net (Gated Attention & Propagation Network) model. Implementation of TransformerConv layers and label propagation mechanisms to handle semi-supervised learning scenarios efficiently.

**Week 8**  
Development of the STRA-GNN (Structural-Temporal Risk Attention GNN) model. Refining the attention mechanism to explicitly attend to neighbor risk statistics and integrating them into the transformer architecture.

**Week 9**  
Comprehensive model training and validation. Execution of K-fold cross-validation, hyperparameter tuning (learning rates, hidden dimensions, attention heads), and analyzing performance on class-imbalanced data.

**Week 10**  
Backend integration for the interactive dashboard. Setting up the inference engine, developing caching mechanisms for efficient data loading, and implementing real-time fraud probability computation logic.

**Week 11**  
Frontend development of the Streamlit dashboard. Implementation of user interface components including risk gauges, model comparison tables, confusion matrices, and top-k suspicious transaction lists.

**Week 12**  
Final testing against benchmark metrics (AUC-ROC, F1, AP). Documentation of model architectures, code refactoring, creation of the final internship report, and project presentation.
