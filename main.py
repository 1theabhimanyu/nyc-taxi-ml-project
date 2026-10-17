from config import ensure_directories, apply_ml_style, DATA_URL
from preprocessing import load_and_preprocess_data
from models import get_model_pipelines
from evaluation import evaluate_models, generate_evaluation_plots

def main():
    print("=" * 60)
    print("STARTING WEEK 4: MACHINE LEARNING MODEL DEVELOPMENT & EVALUATION")
    print("=" * 60)

    ensure_directories()
    apply_ml_style()

    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(DATA_URL)
    
    models = get_model_pipelines(preprocessor)
    trained_pipelines = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    generate_evaluation_plots(trained_pipelines, preprocessor, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("WEEK 4 PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check the 'outputs/week4_plots/' folder for your deliverables.")
    print("=" * 60)

if __name__ == "__main__":
    main()