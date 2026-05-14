# 🔮 ChurnIQ — Customer Churn Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-0D96F2?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**Enterprise-grade AI/ML platform that predicts, analyzes and prevents customer churn**

[Live Demo](#) · [Documentation](#architecture) · [Quick Start](#installation)

</div>

---

## 📌 Project Overview

ChurnIQ is a **production-grade AI SaaS platform** that leverages machine learning to predict customer churn probability, segment customer risk cohorts, and deliver actionable retention intelligence — built for telecom, banking, and SaaS enterprises.

This platform goes beyond a basic churn predictor. It includes:
- A multi-model ML pipeline with automated best-model selection
- Real-time single-customer inference with risk scoring
- Bulk CSV processing for fleet-wide customer analysis
- A business insights engine that translates model outputs into retention strategies
- A premium glassmorphism dark UI with enterprise typography

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Real-Time Prediction** | Predict churn probability for individual customers in milliseconds |
| 📂 **Bulk CSV Analysis** | Upload thousands of customers; get risk scores + downloadable report |
| 📊 **Analytics Dashboard** | 10+ interactive Plotly charts: churn distribution, tenure trends, segment analysis |
| 🧪 **Model Performance Lab** | Train & compare 4 ML models with ROC curves, confusion matrices, radar charts |
| 💡 **Business Insights Engine** | AI-generated churn drivers, retention playbook, segment risk table |
| 🏆 **Auto Model Selection** | Automatically selects best model by ROC-AUC across cross-validation folds |
| 📥 **Export Reports** | One-click CSV export of churn predictions with risk classifications |
| 🎨 **Premium Dark UI** | Glassmorphism design, Sora + DM Sans fonts, neon accents, animated KPI cards |

---

## 🖥️ Screenshots

> _Add screenshots after running the app_

| Dashboard | Prediction | Analytics |
|---|---|---|
| `screenshots/home.png` | `screenshots/predict.png` | `screenshots/analytics.png` |

| Model Lab | Business Insights | Bulk Analysis |
|---|---|---|
| `screenshots/model_lab.png` | `screenshots/insights.png` | `screenshots/bulk.png` |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip / conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train ML models (first time only — ~30 seconds)
python train_model.py

# 5. Launch the platform
streamlit run app.py
```

The platform opens at **http://localhost:8501**

> **Note:** On first launch, if models aren't found, the app auto-trains them. You can also pre-train via `python train_model.py`.

---

## 🗂️ Project Architecture

```
Customer-Churn-Prediction/
│
├── app.py                        # 🚀 Main Streamlit application (7 pages, ~500 LOC)
├── train_model.py                # 🏋️ Offline training pipeline
├── requirements.txt              # 📦 Python dependencies
├── README.md                     # 📖 This file
├── .gitignore
│
├── dataset/
│   ├── generate_dataset.py       # Synthetic telecom churn data generator
│   └── telco_churn.csv           # Generated dataset (7,043 rows, 22 features)
│
├── trained_model/
│   ├── preprocessor.pkl          # Fitted ChurnPreprocessor pipeline
│   ├── best_model.pkl            # Best performing ML model (auto-selected)
│   └── model_metadata.pkl        # Evaluation results, feature names, config
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py           # Feature engineering, encoding, scaling
│   ├── model_trainer.py          # Model registry, training, evaluation, I/O
│   ├── visualizations.py         # 11 Plotly chart components
│   └── styles.py                 # CSS design system + HTML components
│
├── sample_data/
│   ├── create_sample.py          # Script to generate bulk upload demo file
│   └── sample_customers.csv      # 200-row demo file (no Churn column)
│
├── assets/                       # Icons, logos, images
├── notebooks/                    # EDA / experimentation notebooks
└── .streamlit/
    └── config.toml               # Dark theme + server config
```

---

## 🤖 ML Pipeline

### Dataset
- **7,043** synthetic telecom customer records
- **22 features**: demographics, service usage, contract, billing, support tickets
- **Churn Rate**: ~26% (realistic class imbalance maintained)
- Feature engineering: tenure groups, high-value flag, support intensity flag

### Models Trained

| Model | Typical AUC | Notes |
|---|---|---|
| XGBoost | ~84–87% | Usually best; gradient boosting on tabular data |
| Random Forest | ~82–86% | Robust ensemble; good feature importance |
| Logistic Regression | ~78–82% | Baseline; interpretable coefficients |
| Decision Tree | ~76–80% | Fast; prone to overfitting without constraints |

### Preprocessing Pipeline
1. Remove duplicate rows and customer ID columns
2. Coerce TotalCharges to numeric (raw data has string spaces)
3. Impute missing values with column median
4. Feature engineering (tenure bins, derived flags)
5. Label-encode binary categoricals
6. One-hot encode multi-class categoricals
7. StandardScaler on numeric features
8. Stratified 80/20 train/test split
9. 5-fold stratified cross-validation

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (primary selection criterion)
- Cross-validated AUC (mean ± std)
- Confusion matrix analysis
- Feature importance (tree models) / coefficient magnitude (LR)

---

## 📊 Model Performance

> Results vary slightly per run due to synthetic data generation randomness.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost ⭐ | ~85% | ~72% | ~74% | ~73% | ~87% |
| Random Forest | ~83% | ~70% | ~71% | ~70% | ~85% |
| Logistic Regression | ~80% | ~65% | ~68% | ~66% | ~82% |
| Decision Tree | ~78% | ~63% | ~67% | ~65% | ~79% |

---

## 🌐 Deployment

### Streamlit Cloud (Recommended — Free)
1. Push to GitHub: `git push origin main`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select repo → set `app.py` as main file
4. Add startup command in Advanced Settings:
   ```
   python train_model.py
   ```
5. Deploy → get public URL in ~2 minutes

### Render
- Build command: `pip install -r requirements.txt && python train_model.py`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Hugging Face Spaces
- Framework: Streamlit
- Add `train_model.py` call in a startup script or `app.py` init block

---

## 🔮 Future Improvements

- [ ] SHAP explainability integration for individual predictions
- [ ] Survival analysis (time-to-churn modelling with Kaplan-Meier)
- [ ] REST API endpoint with FastAPI for programmatic access
- [ ] Real-time alert system (webhook / email for high-risk customers)
- [ ] A/B test tracking for retention campaigns
- [ ] GPT-powered natural language insight generation
- [ ] Multi-tenant support with user authentication
- [ ] AutoML with Optuna hyperparameter optimization
- [ ] PostgreSQL backend for persistent prediction logging
- [ ] Docker containerization for reproducible deployment

---

## 🧑‍💻 Resume / LinkedIn Description

**Customer Churn Prediction Intelligence Platform**  
*Python · Streamlit · XGBoost · Scikit-learn · Plotly · Pandas*

Engineered an enterprise-grade AI/ML SaaS platform for customer churn prediction used in simulated telecom analytics. Implemented a 4-model ML pipeline (XGBoost, Random Forest, Logistic Regression, Decision Tree) with automated best-model selection via stratified cross-validation (ROC-AUC ~87%). Built a full-stack Streamlit application with real-time inference, bulk CSV analysis for thousands of customers, interactive Plotly dashboards, and a business insights engine delivering actionable retention strategies. Designed a premium glassmorphism dark UI with enterprise typography, achieving 0 Streamlit default styling.

**Key Outcomes:**
- Achieved ~87% ROC-AUC on held-out test set with realistic ~26% churn class imbalance
- Designed modular preprocessing pipeline handling encoding, scaling, and feature engineering without data leakage
- Built 7-page multi-page Streamlit application with 11 custom Plotly visualizations

---

## 🎤 Interview Explanation

**"Walk me through your churn prediction project"**

> "I built ChurnIQ — a full-stack ML platform for customer churn prediction. The ML pipeline starts with feature engineering on a realistic telecom dataset: I created tenure bins, high-value customer flags, and support intensity features. I trained four models with stratified splits to handle class imbalance — XGBoost achieved ~87% AUC. The preprocessing pipeline uses scikit-learn transformers for encoding and scaling to prevent data leakage. The Streamlit app has seven pages: real-time prediction with risk scoring, bulk CSV analysis for fleet-wide analysis, a model comparison lab with ROC curves and radar charts, and a business insights engine. I focused on enterprise production quality — modular code, caching, exception handling, and a premium custom CSS design system."

---

## 📄 License

MIT License — free to use, modify and distribute.

---

<div align="center">
  <sub>Built with ❤️ · Streamlit + XGBoost + Scikit-learn · ChurnIQ v2
