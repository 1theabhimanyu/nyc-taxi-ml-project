import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.tree import DecisionTreeClassifier
from config import PLOTS_DIR

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Trains models, evaluates via cross-validation and holdout metrics, and returns trained pipelines."""
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_list = []
    trained_pipelines = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING & PERFORMANCE EVALUATION")
    print("=" * 60)

    for name, pipeline in models.items():
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring="roc_auc")
        
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        results_list.append({
            "Model": name,
            "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "Test ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "5-Fold CV AUC": f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f}"
        })

    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))
    return trained_pipelines

def generate_evaluation_plots(trained_pipelines, preprocessor, X_train, X_test, y_train, y_test):
    """Generates and saves the 3 required evaluation visualizations."""
    print("\n[INFO] Generating model evaluation visualizations...")

    # --- Viz 1: Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, name, cmap in zip(axes, ["Logistic Regression", "Decision Tree (Depth=5)"], ["Blues", "Greens"]):
        ConfusionMatrixDisplay.from_estimator(
            trained_pipelines[name], X_test, y_test,
            display_labels=["Low Tip (<$2.50)", "High Tip (>=$2.50)"],
            cmap=cmap, values_format="d", ax=ax, colorbar=False
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        ax.grid(False)
    plt.suptitle("Visualization 1: Holdout Test-Set Confusion Matrix Comparison", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "week4_viz1_confusion_matrices.png"), bbox_inches="tight")
    plt.close()

    # --- Viz 2: ROC & Precision-Recall Curves (FIXED) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {
        "Logistic Regression": "#1f77b4", 
        "Decision Tree (Depth=5)": "#ff7f0e", 
        "Random Forest (Ensemble)": "#2ca02c"
    }
    
    for name, pipe in trained_pipelines.items():
        # Plot ROC Curve and set color and line width safely
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=axes[0], name=name)
        axes[0].lines[-1].set_color(colors[name])
        axes[0].lines[-1].set_linewidth(2.2)
        
        # Plot Precision-Recall Curve and set color and line width safely
        PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=axes[1], name=name)
        axes[1].lines[-1].set_color(colors[name])
        axes[1].lines[-1].set_linewidth(2.2)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.50)")
    axes[0].set_title("2A. ROC Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")

    axes[1].set_title("2B. Precision-Recall Tradeoff Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "week4_viz2_roc_and_pr_curves.png"), bbox_inches="tight")
    plt.close()

    # --- Viz 3: Overfitting Depth Curve & Feature Importance ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    depths = range(1, 19)
    train_scores, test_scores = [], []

    for d in depths:
        dt_temp = DecisionTreeClassifier(max_depth=d, random_state=42)
        pipe_dt = Pipeline([("prep", preprocessor), ("clf", dt_temp)])
        pipe_dt.fit(X_train, y_train)
        train_scores.append(roc_auc_score(y_train, pipe_dt.predict_proba(X_train)[:, 1]))
        test_scores.append(roc_auc_score(y_test, pipe_dt.predict_proba(X_test)[:, 1]))

    axes[0].plot(depths, train_scores, marker="o", color="#2b83ba", lw=2.2, label="Training ROC-AUC")
    axes[0].plot(depths, test_scores, marker="s", color="#d7191c", lw=2.2, label="Test ROC-AUC")
    axes[0].axvline(5, color="green", linestyle="--", lw=1.8, label="Optimal Depth (d=5)")
    axes[0].set_title("3A. Bias-Variance Tradeoff (Decision Tree Depth)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Max Depth")
    axes[0].set_ylabel("ROC-AUC Score")
    axes[0].legend(loc="center right")

    rf_pipe = trained_pipelines["Random Forest (Ensemble)"]
    feat_imp = pd.Series(
        rf_pipe.named_steps["clf"].feature_importances_,
        index=rf_pipe.named_steps["prep"].get_feature_names_out()
    ).sort_values(ascending=True).tail(10)

    feat_imp.plot(kind="barh", color="#1b9e77", edgecolor="black", ax=axes[1])
    axes[1].set_title("3B. Top 10 Predictive Features (Random Forest)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Gini Feature Importance Share")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "week4_viz3_overfitting_and_features.png"), bbox_inches="tight")
    plt.close()
    print("[INFO] All evaluation plots saved successfully.")