# Fraud Detection - GTAN, RGTAN, STAGN Models

A focused fraud detection framework implementing three graph neural network models:
- **GTAN** - Graph Temporal Attention Network (AAAI 2023)
- **RGTAN** - Risk-aware Graph Temporal Attention Network (TKDE 2025)
- **STAGN** - Spatial-Temporal Attention Graph Network (TKDE 2020)

## Performance on S-FFSD Dataset

| Model | AUC | F1 | AP |
|-------|-----|----|----|
| STAGN | 0.7659 | 0.6852 | 0.3599 |
| GTAN | 0.8286 | 0.7336 | 0.6585 |
| RGTAN | 0.8461 | 0.7513 | 0.6939 |

## Repository Structure

```
├── methods/
│   ├── gtan/          # GTAN model implementation
│   ├── rgtan/         # RGTAN model implementation
│   └── stagn/         # STAGN model implementation
├── config/            # Model configurations
├── feature_engineering/  # Data preprocessing
├── models/            # Trained model checkpoints
├── main.py            # Training entry point
├── streamlit_app.py   # Interactive dashboard
├── requirements.txt   # Dependencies
└── *_Model_Documentation.md  # Detailed documentation
```

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Training
```bash
python main.py --method gtan
python main.py --method rgtan
python main.py --method stagn
```

### Dashboard
```bash
streamlit run streamlit_app.py
```

## Documentation

- [GTAN Model Documentation](GTAN_Model_Documentation.md)
- [RGTAN Model Documentation](RGTAN_Model_Documentation.md)
- [STAGN Model Documentation](STAGN_Model_Documentation.md)

## License

GPL-3.0
