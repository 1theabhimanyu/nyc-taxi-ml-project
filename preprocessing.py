import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def load_and_preprocess_data(data_url: str):
    """Loads raw data, cleans it, filters credit-card cohort, and creates ML splits."""
    print("[INFO] Loading and preprocessing data for machine learning...")
    df_raw = pd.read_csv(data_url)
    
    # 1. Base Cleaning & Imputation (Week 1 Pipeline)
    df = df_raw.drop_duplicates().dropna(
        subset=["pickup_zone", "dropoff_zone", "pickup_borough", "dropoff_borough"]
    ).copy()
    
    df.loc[df["payment"].isnull() & (df["tip"] > 0), "payment"] = "credit card"
    df["payment"] = df["payment"].fillna(df["payment"].mode()[0])

    df["pickup"] = pd.to_datetime(df["pickup"])
    df["dropoff"] = pd.to_datetime(df["dropoff"])
    df["trip_duration_mins"] = (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60.0
    df["pickup_hour"] = df["pickup"].dt.hour
    df["day_of_week"] = df["pickup"].dt.day_name()

    valid_mask = (
        (df["distance"] > 0.1) & 
        (df["trip_duration_mins"] > 0.5) & 
        (df["trip_duration_mins"] <= 180.0) & 
        (df["passengers"] > 0) & 
        (df["fare"] > 0)
    )
    df_clean = df[valid_mask].copy()

    # 2. Filter Credit Card Cohort (Avoids cash-tip $0 telemetry censorship)
    df_ml = df_clean[df_clean["payment"] == "credit card"].copy()

    # 3. Define Binary Target: High Tip (>= $2.50) vs Low/Zero Tip (< $2.50)
    df_ml["high_tip"] = (df_ml["tip"] >= 2.50).astype(int)

    X = df_ml[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df_ml["high_tip"]

    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 5. Build ColumnTransformer Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

    print(f"[INFO] Dataset ready: {len(df_ml)} credit card trips. Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, preprocessor