from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_model_pipelines(preprocessor):
    """Instantiates Scikit-Learn machine learning pipelines."""
    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Decision Tree (Depth=5)": Pipeline([
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42))
        ]),
        "Random Forest (Ensemble)": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=10, random_state=42))
        ])
    }
    return models