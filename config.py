import os
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset & Directory Paths
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "week4_plots")

# Feature Column Groups (Strictly excluding target leakage features like 'tip' and 'total')
NUMERIC_FEATURES = ["distance", "trip_duration_mins", "fare", "passengers", "tolls", "pickup_hour"]
CATEGORICAL_FEATURES = ["pickup_borough", "dropoff_borough", "color", "day_of_week"]

def ensure_directories():
    """Creates the output directories if they do not exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

def apply_ml_style():
    """Sets global Seaborn and Matplotlib styles for model evaluation charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["font.family"] = "sans-serif"