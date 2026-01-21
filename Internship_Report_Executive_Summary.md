# Executive Summary

**Project Title:** AntiFraud - Financial Fraud Detection Framework using Graph Neural Networks  
**Intern Name:** [Your Name]  
**Organization:** Centre of Excellence in Cognitive Intelligent Systems for Sustainable Solutions (CISSS), RVCE  
**Duration:** 6 Weeks

### Overview
In the evolving landscape of digital finance, fraudulent activities have become increasingly sophisticated, often evading detection by traditional rule-based systems. This internship project, conducted under the **HPCC Systems Centre of Excellence**, addresses this critical challenge by developing a next-generation fraud detection framework utilizing **Graph Neural Networks (GNNs)**. The core philosophy of this project is that fraud is not an isolated event but a networked phenomenon; therefore, analyzing the relationships and interactions between entities is as crucial as analyzing the transactions themselves.

### Objectives
The primary objective was to design, implement, and evaluate advanced deep learning architectures capable of detecting anomalies in semi-supervised financial datasets. Specific goals included:
1.  **Feature Engineering:** developing pipelines to extract temporal behavioral patterns and graph-based risk statistics.
2.  **Model Development:** implementing three distinct GNN architectures to compare different approaches to structural learning.
3.  **Deployment:** creating an interactive dashboard to translate complex model outputs into actionable insights for fraud analysts.

### Methodology
The project utilized the **S-FFSD** (Simulated Financial Fraud Semi-supervised Dataset) to train and model varying fraud scenarios. The technical approach involved:
*   **Advanced Feature Engineering:** Transforming raw transaction logs into rich feature sets, including **2D temporal matrices** (capturing sequential behavior over time) and **Risk-aware Neighbor Statistics** (quantifying the risk level of a user's network).
*   **Graph Construction:** Modeling transactions as nodes in a graph, connected by shared attributes (User ID, Location, Device), enabling the system to trace complex money flows.
*   **Algorithm Implementation:**
    *   **GATE-Net (Graph Attention Temporal Embedding Network):** A unique architecture combining 2D Convolutional Neural Networks (CNNs) for temporal pattern recognition with Graph Convolution for spatial analysis.
    *   **GAP-Net (Gated Attention & Propagation Network):** A semi-supervised model utilizing Graph Transformer Convolutions and Label Propagation to learn from limited labeled data.
    *   **STRA-GNN (Structural-Temporal Risk Attention GNN):** An enhanced transformer model that explicitly integrates neighbor risk statistics, allowing it to "flag" users interacting with known suspicious entities.

### Key Outcomes and Results
All three models were successfully implemented and benchmarked. The **STRA-GNN** model emerged as the top performer, achieving an **AUC-ROC of 0.8461** and an **F1-Score of 0.7513**. This superior performance validates the hypothesis that explicitly modeling neighborhood risk significantly enhances detection capabilities.

Key deliverables include:
*   A fully functional **Python/Pytorch codebase** leveraging the Deep Graph Library (DGL).
*   An interactive **Streamlit Dashboard** that allows users to select models, run real-time inference on specific transactions, and visualize risk scores via an intuitive gauge interface.
*   Comprehensive **technical documentation** detailing the architecture and usage of each model.

### Conclusion
This project successfully demonstrates the efficacy of Graph Neural Networks in identifying complex financial fraud. By moving beyond isolated transaction analysis to a holistic, graph-based view, the developed framework offers a robust solution for modern financial security. The work aligns perfectly with the CoE’s mission to create intelligent, sustainable systems for real-world problems, laying a strong foundation for future research in anomaly detection.
