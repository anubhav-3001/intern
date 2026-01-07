# Internship Report - Chapters 3 & 4

## Financial Fraud Detection Using Graph Neural Networks

**Intern Name:** [Your Name]  
**Duration:** 6 Weeks  
**Organization:** [Organization Name]  
**Project Title:** AntiFraud - Financial Fraud Detection Framework using Graph Neural Networks

---

# Chapter 3: Tasks Performed

## 3.1 Summary of the Tasks Performed During Internship

### 3.1.1 Project Understanding and Literature Review

During the initial phase of the internship, I conducted an extensive literature review of state-of-the-art fraud detection techniques, focusing on:

- **Graph Neural Networks (GNNs)** for fraud detection
- **Temporal feature engineering** for transaction data
- **Semi-supervised learning** approaches for handling unlabeled data
- Reviewed research papers including:
  - "Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation" (AAAI 2023)
  - "Enhancing Attribute-driven Fraud Detection with Risk-aware Graph Representation" (TKDE 2025)
  - "Graph Neural Network for Fraud Detection via Spatial-temporal Attention" (TKDE 2020)

### 3.1.2 Implementation and Analysis of Three Fraud Detection Models

I worked on understanding, implementing, and documenting three distinct graph neural network models for financial fraud detection:

#### A. GATE-Net (Graph Attention Temporal Embedding Network) - Previously STAGN

**Key Implementation Tasks:**
- Implemented 2D feature matrix generation (5 features × 8 time windows) using temporal aggregation
- Built the temporal attention mechanism to identify important time windows
- Implemented 2D CNN layer for spatial-temporal pattern extraction (64 filters, 2×2 kernel)
- Constructed Source→Target transaction graph with edge features (Amount + Location)
- Integrated Graph Convolution layers for learning account relationships

**Technical Specifications:**
| Component | Implementation Detail |
|-----------|----------------------|
| Input Shape | (N, 5, 8) - 2D feature matrices |
| Time Windows | [1, 3, 5, 10, 20, 50, 100, 500] |
| Features | AvgAmount, TotalAmount, BiasAmount, Count, TradingEntropy |
| Graph Type | Account graph (Source → Target) |
| Edge Features | Normalized Amount + One-Hot Location |

#### B. GAP-Net (Gated Attention & Propagation Network) - Previously GTAN

**Key Implementation Tasks:**
- Implemented temporal feature engineering generating 127 features across 15 time windows
- Built TransformerConv layers with 4-head attention mechanism
- Implemented categorical embeddings for Target, Location, and Type
- Developed Label Propagation Attention (LPA) for semi-supervised learning
- Implemented gated skip connections for stable training

**Technical Specifications:**
| Component | Implementation Detail |
|-----------|----------------------|
| Input Shape | (N, 127) - 1D temporal features |
| Time Windows | 15 windows with 8 statistics each |
| Graph Type | Transaction similarity graph (shared attributes) |
| Attention Heads | 4 per TransformerConv layer |
| Training | 5-fold cross-validation, 15 epochs |

#### C. STRA-GNN (Structural-Temporal Risk Attention GNN) - Previously RGTAN

**Key Implementation Tasks:**
- Extended GAP-Net with 6 Risk-aware Neighbor Statistics features
- Implemented 1D CNN (Tabular1DCNN2) for processing neighbor risk statistics
- Built multi-head attention mechanism for risk aggregation (9 heads for S-FFSD dataset)
- Integrated neighbor features: degree, riskstat, 1hop_degree, 2hop_degree, 1hop_riskstat, 2hop_riskstat
- Standardized neighbor features using StandardScaler

**Technical Specifications:**
| Component | Implementation Detail |
|-----------|----------------------|
| Input Shape | (N, 133) - 127 temporal + 6 neighbor risk features |
| Risk Features | 6 neighbor statistics at 1-hop and 2-hop levels |
| CNN Module | 1D Convolution for risk statistics processing |
| Risk Attention Heads | 9 heads for S-FFSD dataset |
| Training | 3-fold cross-validation, 5 epochs |

### 3.1.3 Feature Engineering Pipeline Development

Created a comprehensive feature engineering pipeline that processes raw transaction data:

**Raw Dataset Fields (S-FFSD - 7 columns):**
- Time, Source, Target, Amount, Location, Type, Labels

**Generated Features:**
1. **Temporal Aggregation Features (120 features):**
   - 15 time windows × 8 statistics per window
   - Statistics: avg, total, std, bias, count, target_count, location_count, type_count

2. **Neighbor Risk-Aware Features (6 features):**
   - In-degree of nodes
   - Count of fraudulent 1-hop neighbors
   - Sum of 1-hop and 2-hop neighbor degrees
   - Sum of 1-hop and 2-hop neighbor risk statistics

### 3.1.4 Graph Construction and Data Processing

Implemented graph construction algorithms for different model requirements:

**For GAP-Net and STRA-GNN:**
- Transaction similarity graph connecting nodes with shared attributes
- Edge creation: Each transaction linked to 3 nearest temporal neighbors
- 4 edge types: Same Source, Same Target, Same Location, Same Type

**For GATE-Net:**
- Bipartite-like graph between Source and Target accounts
- Edge features include normalized transaction amounts and one-hot encoded locations

### 3.1.5 Interactive Dashboard Development (Streamlit)

Developed a comprehensive Streamlit dashboard for fraud detection inference:

**Dashboard Features:**
- Model selection interface (GATE-Net, GAP-Net, STRA-GNN)
- Single transaction prediction with risk gauge visualization
- Top-K suspicious transactions ranking
- Benchmark metrics comparison
- Dataset information and statistics display
- Model architecture documentation

**Technical Implementation:**
- Cached data loading and model initialization for performance
- Real-time fraud probability computation
- Color-coded risk level indicators (GREEN: Low, YELLOW: Medium, RED: High)
- Confusion matrix and performance metrics visualization

### 3.1.6 Model Training and Evaluation

Trained all three models on the S-FFSD (Simulated Financial Fraud Semi-supervised Dataset):

**Training Results:**
| Model | AUC | F1-Score | Average Precision |
|-------|-----|----------|-------------------|
| GATE-Net | 0.7659 | 0.6852 | 0.3599 |
| GAP-Net | 0.8286 | 0.7336 | 0.6585 |
| STRA-GNN | 0.8461 | 0.7513 | 0.6939 |

### 3.1.7 Technical Documentation

Created comprehensive model documentation for all three architectures:

1. **GTAN_Model_Documentation.md** - Complete walkthrough of GAP-Net implementation
2. **RGTAN_Model_Documentation.md** - Complete walkthrough of STRA-GNN implementation  
3. **STAGN_Model_Documentation.md** - Complete walkthrough of GATE-Net implementation

Each documentation includes:
- Dataset description and preprocessing pipeline
- Feature engineering methodology
- Model architecture with diagrams
- Training process and hyperparameters
- Evaluation metrics and results

### 3.1.8 Version Control and Deployment

- Maintained code in GitHub repository for version control
- Updated and synchronized code across multiple directories
- Ensured reproducibility with proper requirements.txt and configuration files

---

# Chapter 4: Reflections

## 4.1 Technical Knowledge Acquired

### 4.1.1 Deep Learning and Neural Networks

**Graph Neural Networks (GNNs):**
- Mastered the fundamentals of Graph Neural Networks and their application in fraud detection
- Learned how to construct graph representations from tabular transaction data
- Understood message passing mechanisms and neighbor aggregation techniques
- Implemented various GNN architectures including GraphConv and TransformerConv

**Attention Mechanisms:**
- Gained hands-on experience with multi-head attention mechanisms
- Implemented scaled dot-product attention in graph context
- Understood the role of attention in capturing important patterns in sequential and graph-structured data
- Learned about temporal attention for time-series pattern recognition

**Convolutional Neural Networks:**
- Applied 2D CNNs for spatial-temporal pattern extraction
- Understood the concept of learnable filters for feature detection
- Implemented 1D CNNs for processing tabular neighbor statistics

### 4.1.2 Data Science and Feature Engineering

**Temporal Feature Engineering:**
- Learned to create rolling window aggregations for transaction data
- Implemented time-based statistics (mean, sum, std, bias) across multiple time horizons
- Understood the importance of capturing behavioral patterns over different time scales

**Graph-Based Feature Engineering:**
- Computed neighbor-level risk statistics (1-hop and 2-hop)
- Implemented degree centrality and risk propagation features
- Learned how graph topology encodes fraudulent patterns

**Data Preprocessing:**
- Mastered handling of categorical variables through embedding tables
- Applied Label Encoding and One-Hot Encoding techniques
- Implemented feature standardization using StandardScaler

### 4.1.3 Frameworks and Libraries

**PyTorch:**
- Gained proficiency in PyTorch for deep learning model implementation
- Learned module design patterns (nn.Module, forward methods)
- Implemented custom loss functions and optimizers
- Understood gradient computation and backpropagation

**DGL (Deep Graph Library):**
- Mastered DGL for graph neural network implementation
- Learned graph construction from edge lists (COO format)
- Understood node feature and edge feature attachment
- Implemented message passing using DGL functions (u_dot_v, u_mul_e, etc.)

**Streamlit:**
- Built interactive web dashboards for machine learning models
- Implemented caching mechanisms for efficient data loading
- Created visualizations including charts, metrics, and gauges
- Designed user-friendly interfaces for model inference

**Other Libraries:**
- NumPy and Pandas for data manipulation
- Scikit-learn for metrics and preprocessing (AUC, F1, AP, LabelEncoder)
- YAML for configuration management

### 4.1.4 Machine Learning Concepts

**Semi-Supervised Learning:**
- Understood how to leverage unlabeled data for model training
- Implemented Label Propagation Attention for semi-supervised classification
- Learned about masking strategies for training with incomplete labels

**Model Training Techniques:**
- Implemented K-fold cross-validation for robust evaluation
- Applied early stopping to prevent overfitting
- Used learning rate scheduling for optimal convergence
- Handled class imbalance through weighted loss functions

**Evaluation Metrics:**
- Mastered fraud detection metrics: AUC-ROC, F1-Score, Average Precision
- Understood the trade-offs between precision and recall
- Learned to interpret confusion matrices and ROC curves

### 4.1.5 Software Engineering Practices

- Modular code organization with separate files for models, data loading, and configuration
- Configuration management using YAML files
- Version control with Git and GitHub
- Documentation using Markdown with diagrams and tables

---

## 4.2 Soft Skills Acquired

### 4.2.1 Problem-Solving and Analytical Thinking

**Complex Problem Decomposition:**
- Learned to break down the complex fraud detection problem into manageable components
- Developed systematic approaches to debug model training issues
- Acquired the ability to trace through multi-layered neural network architectures

**Critical Analysis:**
- Developed skills to compare and contrast different model architectures
- Learned to identify strengths and weaknesses of each approach
- Gained ability to make informed decisions about model selection based on requirements

**Debugging and Troubleshooting:**
- Enhanced skills in debugging deep learning models
- Learned to identify and resolve tensor shape mismatches
- Developed strategies for handling runtime errors in complex pipelines

### 4.2.2 Research and Self-Learning

**Literature Review Skills:**
- Developed ability to read and understand academic papers in deep learning
- Learned to extract key concepts and implementation details from research
- Acquired skills to translate research ideas into working code

**Independent Learning:**
- Enhanced ability to learn new frameworks and libraries independently
- Developed skills to navigate documentation and API references
- Gained confidence in exploring unfamiliar codebases

### 4.2.3 Technical Communication

**Documentation:**
- Improved ability to write clear and comprehensive technical documentation
- Learned to create effective diagrams and visualizations for complex architectures
- Developed skills in explaining technical concepts to diverse audiences

**Code Comments and Readability:**
- Enhanced skills in writing self-documenting code
- Learned best practices for code commenting and organization
- Developed ability to create user-friendly interfaces

### 4.2.4 Project Management

**Time Management:**
- Learned to prioritize tasks and manage multiple concurrent objectives
- Developed skills in estimating effort for technical tasks
- Gained experience in meeting deadlines while maintaining code quality

**Task Organization:**
- Acquired skills in organizing complex projects into phases
- Learned to track progress and identify blockers
- Developed ability to adapt plans based on new requirements

### 4.2.5 Attention to Detail

**Code Review and Quality:**
- Developed a keen eye for code correctness and consistency
- Learned to verify model outputs and validate results
- Acquired skills in ensuring reproducibility of experiments

**Documentation Accuracy:**
- Enhanced attention to accuracy in technical writing
- Learned to cross-verify information across multiple sources
- Developed skills in maintaining consistency across documentation

### 4.2.6 Adaptability and Flexibility

**Technology Adaptation:**
- Demonstrated ability to quickly learn new frameworks (DGL, Streamlit)
- Adapted to working with unfamiliar codebases and architectures
- Showed flexibility in modifying approaches based on challenges encountered

**Problem Adaptation:**
- Learned to pivot strategies when initial approaches faced obstacles
- Developed resilience in handling unexpected errors and issues
- Acquired ability to find alternative solutions when facing constraints

---

## Summary

This internship provided invaluable hands-on experience in building production-ready machine learning systems for fraud detection. The combination of theoretical knowledge from research papers and practical implementation skills has established a strong foundation for future work in graph-based machine learning and financial applications. The project demonstrated the power of combining temporal patterns with graph structures for detecting complex fraud patterns, and the importance of proper feature engineering in achieving high model performance.

Key Achievements:
1. Successfully implemented and documented three state-of-the-art fraud detection models
2. Built an interactive dashboard for real-time fraud prediction
3. Created comprehensive technical documentation for future reference
4. Achieved competitive results on the S-FFSD benchmark dataset
5. Gained proficiency in cutting-edge deep learning frameworks and techniques

---

*Report prepared as part of the 6-week internship program*
