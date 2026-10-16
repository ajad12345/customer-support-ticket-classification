"""
Model Training and Evaluation Pipeline
Team Batch 10, Project 6: Customer Support Ticket Classification

This script executes the complete machine learning workflow:
1. Loads raw dataset from data/tickets.csv
2. Applies the shared preprocessing pipeline (src/preprocess.py)
3. Enforces an 80/20 Stratified train-test split (random_state=42)
4. Constructs scikit-learn Pipelines (TfidfVectorizer + Classifier) to prevent data leakage
5. Evaluates 3 distinct models with 5-Fold Stratified Cross-Validation:
   - Multinomial Naive Bayes (MNB)
   - Logistic Regression (LR)
   - Linear Support Vector Machine (Linear SVM with CalibratedClassifierCV)
6. Performs hyperparameter tuning using GridSearchCV on the leading model
7. Selects the champion model based on Macro-F1
8. Evaluates on unseen test data and exports:
   - reports/confusion_matrix.png
   - reports/model_comparison.png
   - reports/classification_report.txt
   - reports/error_analysis.csv
   - models/model.pkl
   - models/vectorizer.pkl
   - models/metrics.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Import shared preprocessing pipeline
from src.preprocess import preprocess_text

SEED = 42

def run_training_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "tickets.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=" * 70)
    print("CUSTOMER SUPPORT TICKET CLASSIFICATION: TRAINING PIPELINE")
    print("=" * 70)
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples loaded: {len(df)}")
    print("Class breakdown:\n", df["category"].value_counts())
    
    # Preprocess text
    print("\n[Step 1] Applying text preprocessing (cleaning, negation preservation, lemmatization)...")
    df["clean_text"] = df["text"].apply(preprocess_text)
    
    # Remove any empty texts if resulting from filtering
    initial_len = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"Processed {len(df)} non-empty samples (dropped {initial_len - len(df)}).")
    
    X = df["clean_text"].values
    y = df["category"].values
    categories = sorted(list(set(y)))
    
    # Stratified 80/20 train-test split
    print("\n[Step 2] Splitting into 80% Train and 20% Test (Stratified, seed=42)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")
    
    # Define candidate model pipelines
    # We use CalibratedClassifierCV on LinearSVC to obtain well-calibrated class probability scores for confidence routing
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    pipelines = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.1))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=SEED))
        ]),
        "Linear SVM (Calibrated)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED, dual="auto"), cv=3))
        ])
    }
    
    print("\n[Step 3] Running 5-Fold Stratified Cross-Validation on Training Set...")
    model_cv_scores = {}
    model_test_metrics = {}
    fitted_pipelines = {}
    
    for name, pipe in pipelines.items():
        print(f"\n--- Evaluating: {name} ---")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_cv = float(np.mean(cv_scores))
        std_cv = float(np.std(cv_scores))
        model_cv_scores[name] = {"mean": mean_cv, "std": std_cv, "scores": [float(s) for s in cv_scores]}
        print(f"5-Fold CV Macro-F1: {mean_cv:.4f} (+/- {std_cv:.4f})")
        
        # Fit on full training set and evaluate on holdout test set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        y_pred = pipe.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        macro_p = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        model_test_metrics[name] = {
            "accuracy": acc,
            "precision_macro": macro_p,
            "recall_macro": macro_r,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1
        }
        print(f"Test Accuracy: {acc:.4f} | Test Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")
        
    # Hyperparameter Tuning with GridSearchCV on Logistic Regression (or best CV model)
    best_candidate_name = max(model_cv_scores, key=lambda k: model_cv_scores[k]["mean"])
    print(f"\n[Step 4] Hyperparameter Tuning for Best Candidate: {best_candidate_name}")
    
    if "Logistic Regression" in best_candidate_name:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "clf__C": [0.5, 1.0, 2.0, 5.0]
        }
        base_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_df=0.95, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
        ])
    elif "Linear SVM" in best_candidate_name:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__estimator__C": [0.5, 1.0, 2.0]
        }
        base_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(random_state=SEED, dual="auto"), cv=3))
        ])
    else:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__alpha": [0.01, 0.1, 0.5, 1.0]
        }
        base_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", MultinomialNB())
        ])
        
    grid_search = GridSearchCV(
        base_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"GridSearchCV Best Params: {grid_search.best_params_}")
    print(f"GridSearchCV Best CV Macro-F1: {grid_search.best_score_:.4f}")
    
    # Final Champion Model
    final_pipeline = grid_search.best_estimator_
    y_test_pred = final_pipeline.predict(X_test)
    y_test_proba = final_pipeline.predict_proba(X_test)
    
    final_acc = float(accuracy_score(y_test, y_test_pred))
    final_macro_p = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    final_macro_r = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    final_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    final_weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    
    print("\n" + "=" * 70)
    print("FINAL CHAMPION MODEL EVALUATION ON UNSEEN TEST SET")
    print("=" * 70)
    print(f"Accuracy:         {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Macro Precision:  {final_macro_p:.4f}")
    print(f"Macro Recall:     {final_macro_r:.4f}")
    print(f"Macro F1-Score:   {final_macro_f1:.4f}")
    print(f"Weighted F1-Score:{final_weighted_f1:.4f}")
    
    # Detailed Classification Report
    cls_report = classification_report(y_test, y_test_pred, digits=4, zero_division=0)
    print("\nClassification Report:\n", cls_report)
    
    report_file_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("CUSTOMER SUPPORT TICKET CLASSIFICATION - FINAL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Selected Champion Model: {best_candidate_name} (Tuned via GridSearchCV)\n")
        f.write(f"Optimal Hyperparameters: {grid_search.best_params_}\n\n")
        f.write(f"Test Accuracy:         {final_acc:.4f}\n")
        f.write(f"Test Macro Precision:  {final_macro_p:.4f}\n")
        f.write(f"Test Macro Recall:     {final_macro_r:.4f}\n")
        f.write(f"Test Macro F1-Score:   {final_macro_f1:.4f}\n")
        f.write(f"Test Weighted F1-Score:{final_weighted_f1:.4f}\n\n")
        f.write("PER-CLASS CLASSIFICATION REPORT:\n")
        f.write(cls_report)
    print(f"Saved classification report to: {report_file_path}")
    
    # Confusion Matrix Visualization
    print("\n[Step 5] Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, y_test_pred, labels=categories)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar=True
    )
    plt.title(f"Confusion Matrix: {best_candidate_name}\n(Test Set, Macro-F1 = {final_macro_f1:.4f})", fontsize=13, pad=15)
    plt.xlabel("Predicted Category", fontsize=11, labelpad=10)
    plt.ylabel("True Category", fontsize=11, labelpad=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")
    
    # Model Comparison Chart
    print("\n[Step 6] Generating Model Comparison Bar Chart...")
    models_list = list(model_test_metrics.keys())
    acc_list = [model_test_metrics[m]["accuracy"] for m in models_list]
    f1_list = [model_test_metrics[m]["f1_macro"] for m in models_list]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, acc_list, width, label="Test Accuracy", color="#3b82f6")
    rects2 = ax.bar(x + width/2, f1_list, width, label="Test Macro-F1", color="#10b981")
    
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison on Test Set (Accuracy vs Macro-F1)", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=10)
    ax.set_ylim(0.80, 1.02)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Add numerical labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    comp_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to: {comp_path}")
    
    # Error Analysis
    print("\n[Step 7] Conducting Detailed Error Analysis...")
    test_df = df.iloc[idx_test].copy().reset_index(drop=True)
    test_df["predicted_category"] = y_test_pred
    # Max predicted probability as confidence
    class_order = list(final_pipeline.classes_)
    confidences = np.max(y_test_proba, axis=1)
    test_df["confidence"] = confidences
    
    misclassified = test_df[test_df["category"] != test_df["predicted_category"]].copy()
    
    # Add confusion reason annotation
    def explain_confusion(row):
        true_c = row["category"]
        pred_c = row["predicted_category"]
        text_lower = row["text"].lower()
        if (true_c == "Billing" and pred_c == "Refund/Returns") or (true_c == "Refund/Returns" and pred_c == "Billing"):
            return "Cross-boundary lexical overlap: mentions money/charge alongside cancellation or return keywords."
        elif (true_c == "Order Issues" and pred_c == "Refund/Returns") or (true_c == "Refund/Returns" and pred_c == "Order Issues"):
            return "Shared supply chain vocabulary: damaged package or missing items triggering return intent."
        elif (true_c == "Technical Support" and pred_c == "Account Access") or (true_c == "Account Access" and pred_c == "Technical Support"):
            return "Authentication bug overlap: system error/timeout during login or profile verification."
        else:
            return "Subtle multi-intent phrasing or high semantic proximity."
            
    if len(misclassified) > 0:
        misclassified["explanation"] = misclassified.apply(explain_confusion, axis=1)
        error_csv_path = os.path.join(reports_dir, "error_analysis.csv")
        misclassified.to_csv(error_csv_path, index=False)
        print(f"Total misclassifications: {len(misclassified)} out of {len(test_df)} ({len(misclassified)/len(test_df)*100:.2f}%)")
        print(f"Saved misclassified tickets to: {error_csv_path}")
        print("\nTop Confused Pairs:")
        confused_pairs = misclassified.groupby(["category", "predicted_category"]).size().reset_index(name="count")
        confused_pairs = confused_pairs.sort_values(by="count", ascending=False)
        print(confused_pairs.to_string(index=False))
    else:
        print("Zero test errors found! Creating representative edge-case analysis...")
        test_df["explanation"] = "Correctly classified test sample"
        error_csv_path = os.path.join(reports_dir, "error_analysis.csv")
        test_df.head(10).to_csv(error_csv_path, index=False)
        
    # Save Artifacts
    print("\n[Step 8] Persisting Model Artifacts to models/...")
    model_file = os.path.join(models_dir, "model.pkl")
    vectorizer_file = os.path.join(models_dir, "vectorizer.pkl")
    metrics_file = os.path.join(models_dir, "metrics.json")
    
    # Save entire pipeline
    joblib.dump(final_pipeline, model_file)
    # Also save vectorizer separately for inspectability
    joblib.dump(final_pipeline.named_steps["tfidf"], vectorizer_file)
    
    # Collect metrics dictionary
    report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    
    metrics_data = {
        "project": "Customer Support Ticket Classification (Team Batch 10, Project 6)",
        "selected_model": best_candidate_name,
        "grid_search_best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
        "cv_5fold_macro_f1": model_cv_scores,
        "model_test_comparisons": model_test_metrics,
        "final_test_metrics": {
            "accuracy": final_acc,
            "macro_precision": final_macro_p,
            "macro_recall": final_macro_r,
            "macro_f1": final_macro_f1,
            "weighted_f1": final_weighted_f1,
            "total_test_samples": int(len(y_test)),
            "misclassified_samples": int(len(misclassified))
        },
        "per_class_metrics": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1-score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }
            for cls in categories if cls in report_dict
        },
        "categories": categories
    }
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Artifacts successfully saved:")
    print(f" - Model Pipeline:    {model_file}")
    print(f" - TF-IDF Vectorizer: {vectorizer_file}")
    print(f" - Real Metrics JSON: {metrics_file}")
    print("=" * 70)
    print("TRAINING AND ARTIFACT PERSISTENCE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_training_pipeline()
