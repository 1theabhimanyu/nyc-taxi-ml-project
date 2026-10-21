"""
NYC Taxi EDA Project - Week 4: Machine Learning Model Development & Evaluation
Author: Abhimanyu Kumar
Repository: https://github.com/1theabhimanyu/nyc_taxi_eda_project
Description:
    Implements an end-to-end Scikit-Learn machine learning pipeline covering:
    1. Data Preprocessing (Imputation, StandardScaler, OneHotEncoder via ColumnTransformer)
    2. Supervised Classification (Logistic Regression, Unpruned Decision Tree,
       Pruned Decision Tree, and Random Forest)
    3. Supervised Fare Regression (Ridge Regression, Decision Tree Regressor,
       Random Forest Regressor)
    4. Overfitting vs. Underfitting validation curve analysis & 4-panel visualization suite.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.08)


# ==============================================================================
# 1. DATA INGESTION & PREPARATION (Synced with Week 3 nyc_taxi_eda_project)
# ==============================================================================
def load_and_prepare_data(filepath: str = "nyc_taxi_data.csv", n_samples: int = 50000) -> pd.DataFrame:
    """
    Loads the cleaned NYC Taxi dataset if available locally, or generates a
    deterministic NYC TLC dataset matching the project schema and distributions.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Loading NYC Taxi dataset from '{filepath}'...")
        df = pd.read_csv(filepath)
    else:
        print(f"[INFO] '{filepath}' not found locally. Generating calibrated NYC TLC dataset (N={n_samples})...")
        np.random.seed(42)

        boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
        borough_probs = [0.730, 0.136, 0.098, 0.036]
        pickup_borough = np.random.choice(boroughs, size=n_samples, p=borough_probs)

        pickup_hour = np.random.randint(0, 24, size=n_samples)
        day_of_week = np.random.randint(0, 7, size=n_samples)
        passenger_count = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.68, 0.15, 0.07, 0.05, 0.03, 0.02])

        is_weekday = day_of_week < 5
        is_peak_hour = (
            is_weekday & (((pickup_hour >= 7) & (pickup_hour <= 10)) | ((pickup_hour >= 16) & (pickup_hour <= 20)))
        ).astype(int)

        trip_distance = np.zeros(n_samples)
        is_airport_trip = np.zeros(n_samples, dtype=int)

        for b in boroughs:
            mask = pickup_borough == b
            count = mask.sum()
            if b == "Manhattan":
                trip_distance[mask] = np.random.gamma(shape=2.2, scale=1.15, size=count)
            elif b == "Queens":
                airport_flags = np.random.binomial(1, 0.62, size=count)
                is_airport_trip[mask] = airport_flags
                trip_distance[mask] = np.where(
                    airport_flags == 1,
                    np.random.normal(loc=11.5, scale=3.0, size=count),
                    np.random.gamma(shape=2.5, scale=1.6, size=count)
                )
            elif b == "Brooklyn":
                trip_distance[mask] = np.random.gamma(shape=2.6, scale=1.55, size=count)
            else:
                trip_distance[mask] = np.random.gamma(shape=2.8, scale=1.65, size=count)

        trip_distance = np.clip(trip_distance, 0.2, 45.0)

        speed_mph = np.where(
            is_peak_hour == 1,
            np.random.normal(11.5, 2.4, n_samples),
            np.random.normal(16.8, 3.1, n_samples)
        )
        speed_mph = np.clip(speed_mph, 4.5, 45.0)
        trip_duration_min = (trip_distance / speed_mph) * 60.0

        # Non-linear fare dynamics (Airport flat/surcharge + Peak congestion + Distance/Time)
        peak_surcharge = np.where(is_peak_hour == 1, 2.50 + 0.18 * trip_distance, 0.0)
        airport_surcharge = np.where(is_airport_trip == 1, 5.50, 0.0)
        noise = np.random.normal(0, 1.80, n_samples)

        fare_amount = (
            3.00
            + (2.80 * trip_distance)
            + (0.14 * trip_duration_min)
            + peak_surcharge
            + airport_surcharge
            + noise
        )
        fare_amount = np.clip(fare_amount, 3.00, 200.00)

        card_prob_map = {"Manhattan": 0.764, "Queens": 0.741, "Brooklyn": 0.685, "Bronx": 0.618}
        probs = np.array([card_prob_map[b] for b in pickup_borough])
        payment_type = np.where(np.random.rand(n_samples) < probs, "Credit Card", "Cash")

        df = pd.DataFrame({
            "pickup_borough": pickup_borough,
            "payment_type": payment_type,
            "pickup_hour": pickup_hour,
            "day_of_week": day_of_week,
            "passenger_count": passenger_count,
            "is_peak_hour": is_peak_hour,
            "is_airport_trip": is_airport_trip,
            "trip_distance": np.round(trip_distance, 2),
            "trip_duration_min": np.round(trip_duration_min, 2),
            "fare_amount": np.round(fare_amount, 2)
        })

    # Clean bounds
    df = df[(df["fare_amount"] >= 2.50) & (df["fare_amount"] <= 250.0)].copy()
    df = df[(df["trip_distance"] >= 0.10) & (df["trip_distance"] <= 100.0)].copy()

    # Define Classification Target: High-Value Trip (Fare > $22.00, ~top 28% of rides)
    df["is_high_value_trip"] = (df["fare_amount"] > 22.00).astype(int)
    return df.reset_index(drop=True)


# ==============================================================================
# 2. BUILD SCIKIT-LEARN PREPROCESSING PIPELINE
# ==============================================================================
def build_preprocessor():
    """Constructs a leak-free ColumnTransformer for numeric and categorical features."""
    numeric_features = ["trip_distance", "trip_duration_min", "pickup_hour", "day_of_week", "passenger_count"]
    binary_features = ["is_peak_hour", "is_airport_trip"]
    categorical_features = ["pickup_borough", "payment_type"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", "passthrough", binary_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


# ==============================================================================
# 3. TRAIN & EVALUATE CLASSIFICATION AND REGRESSION MODELS
# ==============================================================================
def run_week4_ml_pipeline(df: pd.DataFrame):
    feature_cols = [
        "trip_distance", "trip_duration_min", "pickup_hour", "day_of_week",
        "passenger_count", "is_peak_hour", "is_airport_trip", "pickup_borough", "payment_type"
    ]
    X = df[feature_cols]
    y_class = df["is_high_value_trip"]
    y_reg = df["fare_amount"]

    # Stratified Train/Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_c_train, y_c_test, y_r_train, y_r_test = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )

    preprocessor = build_preprocessor()

    # --------------------------------------------------------------------------
    # TRACK A: CLASSIFICATION MODELS (High-Value Trip Prediction)
    # --------------------------------------------------------------------------
    clf_models = {
        "Logistic Regression (L2)": LogisticRegression(C=1.0, max_iter=500, random_state=42),
        "Decision Tree (Unpruned - Overfit)": DecisionTreeClassifier(max_depth=None, random_state=42),
        "Decision Tree (Pruned, depth=6)": DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, random_state=42),
        "Random Forest (n=100)": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    }

    clf_results = {}
    roc_curves = {}

    print("\n" + "=" * 85)
    print("WEEK 4 TRACK A: SUPERVISED CLASSIFICATION EVALUATION (Target: is_high_value_trip)")
    print("=" * 85)
    print(f"{'Model Name':<36} | {'Train Acc':<9} | {'Test Acc':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<7}")
    print("-" * 85)

    for name, model in clf_models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_c_train)

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        y_test_prob = pipe.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_c_train, y_train_pred)
        test_acc = accuracy_score(y_c_test, y_test_pred)
        prec = precision_score(y_c_test, y_test_pred)
        rec = recall_score(y_c_test, y_test_pred)
        f1 = f1_score(y_c_test, y_test_pred)
        auc = roc_auc_score(y_c_test, y_test_prob)

        fpr, tpr, _ = roc_curve(y_c_test, y_test_prob)
        roc_curves[name] = (fpr, tpr, auc)

        clf_results[name] = {
            "pipeline": pipe, "train_acc": train_acc, "test_acc": test_acc,
            "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "cm": confusion_matrix(y_c_test, y_test_pred)
        }

        print(f"{name:<36} | {train_acc:>8.2%} | {test_acc:>7.2%} | {prec:>9.3f} | {rec:>8.3f} | {f1:>8.3f} | {auc:>7.3f}")

    # --------------------------------------------------------------------------
    # BIAS-VARIANCE EXPERIMENT: Decision Tree Max-Depth Sweep (1 to 22)
    # --------------------------------------------------------------------------
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22]
    train_f1_scores, test_f1_scores = [], []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train_proc, y_c_train)
        train_f1_scores.append(f1_score(y_c_train, dt.predict(X_train_proc)))
        test_f1_scores.append(f1_score(y_c_test, dt.predict(X_test_proc)))

    # --------------------------------------------------------------------------
    # TRACK B: REGRESSION MODELS (Continuous Fare Prediction)
    # --------------------------------------------------------------------------
    reg_models = {
        "Linear Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree Regressor (Unpruned)": DecisionTreeRegressor(max_depth=None, random_state=42),
        "Decision Tree Regressor (Pruned, d=8)": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=42),
        "Random Forest Regressor (n=100)": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    }

    reg_results = {}
    print("\n" + "=" * 85)
    print("WEEK 4 TRACK B: CONTINUOUS FARE REGRESSION EVALUATION (Target: fare_amount $ USD)")
    print("=" * 85)
    print(f"{'Model Name':<38} | {'Train RMSE':<10} | {'Test RMSE':<10} | {'Test MAE':<10} | {'Test R^2':<8}")
    print("-" * 85)

    for name, model in reg_models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
        pipe.fit(X_train, y_r_train)

        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_r_train, pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_r_test, pred_test))
        test_mae = mean_absolute_error(y_r_test, pred_test)
        test_r2 = r2_score(y_r_test, pred_test)

        reg_results[name] = {
            "train_rmse": train_rmse, "test_rmse": test_rmse,
            "test_mae": test_mae, "test_r2": test_r2,
            "y_test": y_r_test.values, "pred_test": pred_test
        }
        print(f"{name:<38} | ${train_rmse:>9.2f} | ${test_rmse:>9.2f} | ${test_mae:>9.2f} | {test_r2:>8.4f}")
    print("=" * 85 + "\n")

    return {
        "clf_results": clf_results,
        "roc_curves": roc_curves,
        "depth_sweep": (depths, train_f1_scores, test_f1_scores),
        "reg_results": reg_results
    }


# ==============================================================================
# 4. GENERATE MODEL PERFORMANCE COMPARISON VISUALIZATIONS
# ==============================================================================
def generate_week4_visualizations(eval_data: dict, output_path: str = "week4_ml_evaluation_visuals.png"):
    """Generates a 4-panel dashboard comparing ROC curves, Confusion Matrix, Overfitting, and Regression."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(
        "NYC Taxi Project (Week 4): Machine Learning Model Development & Evaluation Suite",
        fontsize=18, fontweight="bold", y=0.98
    )

    # Panel 1: Multi-Model ROC Curve Comparison
    ax1 = axes[0, 0]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for (name, (fpr, tpr, auc)), col in zip(eval_data["roc_curves"].items(), colors):
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2.3, color=col)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Chance (AUC = 0.500)")
    ax1.set_title("1. ROC Curve Comparison Across Classifiers", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax1.legend(loc="lower right", frameon=True)

    # Panel 2: Confusion Matrix Heatmap (Pruned Decision Tree Classifier)
    ax2 = axes[0, 1]
    cm = eval_data["clf_results"]["Decision Tree (Pruned, depth=6)"]["cm"]
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = np.array([
        [f"TN: {cm[0,0]}\n({cm_norm[0,0]:.1%})", f"FP: {cm[0,1]}\n({cm_norm[0,1]:.1%})"],
        [f"FN: {cm[1,0]}\n({cm_norm[1,0]:.1%})", f"TP: {cm[1,1]}\n({cm_norm[1,1]:.1%})"]
    ])
    sns.heatmap(
        cm, annot=annot_labels, fmt="", cmap="Blues", cbar=False, ax=ax2,
        xticklabels=["Predicted Standard ($<=22)", "Predicted High-Value ($>22)"],
        yticklabels=["Actual Standard ($<=22)", "Actual High-Value ($>22)"],
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax2.set_title("2. Confusion Matrix: Pruned Decision Tree (Test Set n=10,000)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Predicted Class Label")
    ax2.set_ylabel("True Class Label")

    # Panel 3: Overfitting vs. Underfitting Curve (Max Depth vs. F1-Score)
    ax3 = axes[1, 0]
    depths, train_f1, test_f1 = eval_data["depth_sweep"]
    ax3.plot(depths, train_f1, "o-", color="#d95f02", linewidth=2.4, label="Training F1-Score (Memorization)")
    ax3.plot(depths, test_f1, "s-", color="#1b9e77", linewidth=2.4, label="Holdout Test F1-Score (Generalization)")
    best_idx = int(np.argmax(test_f1))
    best_depth = depths[best_idx]
    ax3.axvline(best_depth, color="navy", linestyle="--", linewidth=1.8, label=f"Optimal Depth = {best_depth}")
    ax3.annotate(
        "Underfitting\n(High Bias)", xy=(2, test_f1[1]), xytext=(2.5, test_f1[1] - 0.06),
        arrowprops=dict(arrowstyle="->", color="gray"), fontweight="bold", color="#444444"
    )
    ax3.annotate(
        "Overfitting Zone\n(High Variance Gap)", xy=(18, test_f1[-2]), xytext=(13.5, test_f1[-2] - 0.05),
        arrowprops=dict(arrowstyle="->", color="#d95f02"), fontweight="bold", color="#b22222"
    )
    ax3.set_title("3. Bias-Variance Tradeoff: Decision Tree Depth vs. F1-Score", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Decision Tree Maximum Depth (max_depth)")
    ax3.set_ylabel("F1-Score")
    ax3.legend(loc="lower right")

    # Panel 4: Continuous Regression Predicted vs. Actual Fare
    ax4 = axes[1, 1]
    best_reg = eval_data["reg_results"]["Random Forest Regressor (n=100)"]
    y_true = best_reg["y_test"][:1500]
    y_pred = best_reg["pred_test"][:1500]
    ax4.scatter(y_true, y_pred, alpha=0.30, color="#2b5c8f", s=20, label="Test Predictions (Sample n=1,500)")
    ax4.plot([3, 75], [3, 75], "r--", linewidth=2.2, label="Perfect Prediction Line (y = y_hat)")
    ax4.set_xlim(2, 75)
    ax4.set_ylim(2, 75)
    ax4.set_title(
        f"4. Continuous Fare Regression: Actual vs. Predicted\n"
        f"Test RMSE = ${best_reg['test_rmse']:.2f} | MAE = ${best_reg['test_mae']:.2f} | R² = {best_reg['test_r2']:.3f}",
        fontsize=13, fontweight="bold"
    )
    ax4.set_xlabel("Actual Total Fare ($ USD)")
    ax4.set_ylabel("Predicted Total Fare ($ USD)")
    ax4.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Week 4 ML evaluation dashboard to '{output_path}'.")


if __name__ == "__main__":
    taxi_df = load_and_prepare_data()
    evaluation_outputs = run_week4_ml_pipeline(taxi_df)
    generate_week4_visualizations(evaluation_outputs)