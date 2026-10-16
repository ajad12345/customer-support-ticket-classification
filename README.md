# Customer Support Ticket Classification & Automated Routing Engine
**Team Batch 10 | Project 6: Customer Support Ticket Classification**  
*Natural Language Processing Course Practical Implementation*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.10.3-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-red.svg)
![Test Macro-F1](https://img.shields.io/badge/Macro--F1-1.0000-brightgreen.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ajad12345/customer-support-ticket-classification&branch=main&mainModule=app.py)

**GitHub Repository:** [https://github.com/ajad12345/customer-support-ticket-classification](https://github.com/ajad12345/customer-support-ticket-classification)  
**One-Click Streamlit Cloud Deploy:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=ajad12345/customer-support-ticket-classification&branch=main&mainModule=app.py)

---

## 📌 Project Overview
In modern customer service operations, businesses receive thousands of multi-channel support requests every day. Manually triaging tickets causes resolution delays, misdirection, and high operational overhead. 

This project provides an autonomous, end-to-end NLP solution that:
1. Classifies customer requests into **6 core operational categories**:
   - `Billing` 💳
   - `Technical Support` 🛠️
   - `Account Access` 🔐
   - `Order Issues` 📦
   - `Refund/Returns` 🔄
   - `General Inquiry` ℹ️
2. Dynamically routes each ticket to its designated departmental team (e.g. *Finance*, *Engineering*, *Logistics*).
3. Evaluates prediction confidence against an adjustable safety threshold (default: `50%`), routing ambiguous or out-of-scope tickets to a **Manual Review Queue** for human-in-the-loop oversight.
4. Operates strictly with classical machine learning (TF-IDF + Calibrated Linear SVM) without reliance on external paid APIs or large language models.

---

## 🗂️ Project Directory Structure
```
NLP_Project/
├── data/
│   ├── tickets.csv              # Curated balanced dataset (1,528 records across 6 classes)
│   ├── generate_dataset.py      # Reproducible dataset generator script (seed=42)
│   └── DATA_README.md           # Dataset documentation, schema, and limitations
├── notebooks/
│   ├── ticket_classification.ipynb  # Fully executed end-to-end Jupyter Notebook with all outputs
│   └── build_and_run_notebook.py   # Script to programmatically build & execute notebook
├── models/
│   ├── model.pkl                # Serialized champion scikit-learn Pipeline
│   ├── vectorizer.pkl           # Fitted TfidfVectorizer object
│   └── metrics.json             # Real evaluation metrics from cross-validation and test set
├── reports/
│   ├── confusion_matrix.png     # Heatmap visualization of test set confusion matrix
│   ├── model_comparison.png     # Comparative bar chart (Accuracy vs Macro-F1)
│   ├── class_distribution.png   # Class frequency distribution visualization
│   ├── classification_report.txt# Full precision, recall, and F1 text report
│   └── error_analysis.csv       # Error log and edge-case boundary explanations
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Unified negation-preserving NLP cleaning & lemmatization
│   ├── train.py                 # 5-fold CV, GridSearchCV tuning, and artifact exporter
│   └── predict.py               # Standalone CLI inference & department routing module
├── tests/
│   ├── run_tests.py             # Systematic test runner evaluating 16 scenarios
│   └── test_cases.md            # Empirical test results, observations, and edge-case table
├── docs/
│   └── presentation_outline.md  # 12-slide presentation structure with speaker notes
├── app.py                       # Interactive Streamlit multi-tab web application
├── requirements.txt             # Pinned dependency manifest
├── README.md                    # Project overview, setup, and usage instructions
└── project_report.md            # Comprehensive 17-section formal project report
```

---

## ⚙️ Environment Setup & Installation

### 1. Clone or Open the Repository
```bash
git clone <repo-url>
cd "NLP GENAI"
```

### 2. Create and Activate a Python Virtual Environment
```bash
# On Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpora
Run this one-liner to download all required tokenizers, stopwords, and WordNet:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

---

## 🚀 Execution & Reproduction Guide

### A. Run Model Training & Evaluation Pipeline
Trains 3 distinct architectures with 5-Fold Stratified Cross-Validation, runs GridSearchCV hyperparameter tuning, evaluates on the unseen 20% holdout test set, and exports all model artifacts and evaluation graphs:
```bash
python src/train.py
```

### B. Run the Interactive Streamlit Web Application
Launches the full interactive UI featuring Single Ticket Triage, Bulk CSV Ingestion, and Performance Dashboards:
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

### C. Test Standalone Inference (CLI)
Classify any raw text string directly from the terminal:
```bash
# High-confidence Billing ticket
python src/predict.py "I was charged twice on my credit card for invoice #49201."

# Technical issue ticket
python src/predict.py "The mobile application freezes with error 500 when exporting reports."

# Ambiguous / Low-confidence ticket
python src/predict.py "asdfjk 12345"
```

### D. Run the Systematic Test Suite
Executes all 16 predefined test cases (normal, empty, short, gibberish, negations, mixed-category, out-of-scope) and records predictions:
```bash
python tests/run_tests.py
```

### E. Open the Executed Jupyter Notebook
```bash
jupyter notebook notebooks/ticket_classification.ipynb
```

---

## 📊 Experimental Results Summary (Real Empirical Metrics)

All metrics are generated from actual training runs (`seed=42`) and recorded in `models/metrics.json`:

| Model Architecture | 5-Fold CV Macro-F1 | Holdout Test Accuracy | Holdout Test Macro-F1 | Holdout Weighted F1 | Selected Status |
|---|---|---|---|---|---|
| **Multinomial Naive Bayes** | 0.9943 (±0.0076) | 100.00% | 1.0000 | 1.0000 | Baseline |
| **Logistic Regression** | 0.9967 (±0.0066) | 100.00% | 1.0000 | 1.0000 | Candidate |
| **Linear SVM (Calibrated)** | **1.0000 (±0.0000)** | **100.00%** | **1.0000** | **1.0000** | **🏆 Final Champion** |

### Per-Class Performance (Holdout Test Set: 306 samples)
- **Account Access:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 (Support: 51)
- **Billing:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 (Support: 52)
- **General Inquiry:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 (Support: 49)
- **Order Issues:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 (Support: 52)
- **Refund/Returns:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 (Support: 51)
- **Technical Support:** Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000 (Support: 51)

---

## 💡 NLP Engineering Highlights
- **Negation Preservation:** Whitelists negation tokens (`no`, `not`, `never`, `cannot`) during stopword removal so sentences like *"I did NOT receive my parcel"* are correctly understood rather than inverted to *"receive parcel"*.
- **Zero Train-Test Contamination:** Scikit-learn pipelines encapsulate `TfidfVectorizer` to guarantee that vocabulary frequencies and IDF scores are computed strictly on training folds.
- **Calibrated Probabilities:** Utilizes Platt Scaling (`CalibratedClassifierCV`) on top of `LinearSVC` to yield genuine probability estimates for confidence gating.
- **Safety Fallback:** Tickets with maximum class probability below `50.0%` are automatically intercepted and routed to human specialists, preventing misdirection of unfamiliar or noisy requests.

---

## 👥 Course & Team Attribution
- **Course:** Natural Language Processing (NLP)
- **Team Batch:** Batch 10
- **Project Assigned:** Project 6 — *Customer Support Ticket Classification*
- **Role Allocation:**
  - *Data Engineering & Preprocessing:* Student Name 1 (Roll #XXXX)
  - *Model Training & Tuning:* Student Name 2 (Roll #XXXX)
  - *Evaluation & Error Analysis:* Student Name 3 (Roll #XXXX)
  - *Streamlit UI & Deployment:* Student Name 4 (Roll #XXXX)
