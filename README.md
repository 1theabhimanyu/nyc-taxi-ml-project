# 🤖 Machine Learning Model Development & Evaluation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Milestone-Week%204%20Complete-brightgreen.svg)

## 📌 1. Project Overview & Objective
This repository contains the **Machine Learning Model Development** pipeline for predicting high-value passenger tipping behavior (`high_tip >= $2.50`) in NYC taxis. Building upon our Weeks 1–3 data preparation and statistical analyses, this project implements a supervised classification pipeline using `scikit-learn`.

## ⚙️ 2. Modular Architecture
* `config.py`: Shared constants and paths.
* `preprocessing.py`: Cohort filtering (credit-card only), target engineering, and `ColumnTransformer` preprocessing.
* `models.py`: Pipeline definitions for Logistic Regression, Decision Trees, and Random Forests.
* `evaluation.py`: Cross-validation, metrics calculation, and chart generation.
* `main.py`: Orchestrator script.

## 📊 3. Model Performance Summary
* **Logistic Regression:** Test Accuracy = $87.5\%$, ROC-AUC = $0.928$
* **Decision Tree (`max_depth=5`):** Test Accuracy = $87.8\%$, ROC-AUC = $0.924$
* **Random Forest (Ensemble):** Test Accuracy = $88.4\%$, ROC-AUC = $0.936$

## 🚀 4. How to Run Locally
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
python main.py