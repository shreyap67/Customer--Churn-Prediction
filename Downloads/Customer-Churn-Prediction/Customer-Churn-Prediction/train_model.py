"""
Model training script — run once before launching the app.
Usage: python train_model.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_model")

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main():
    logger.info("=" * 60)
    logger.info("ChurnIQ — Model Training Pipeline")
    logger.info("=" * 60)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from dataset.generate_dataset import generate_churn_dataset
    from utils.preprocessor import ChurnPreprocessor, prepare_target
    from utils.model_trainer import train_all_models, save_artifacts

    # 1. Generate / Load dataset
    dataset_path = ROOT / "dataset" / "telco_churn.csv"
    if dataset_path.exists():
        logger.info(f"Loading existing dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        logger.info("Generating synthetic dataset...")
        df = generate_churn_dataset(output_path=dataset_path)

    logger.info(f"Dataset shape: {df.shape} | Churn rate: {(df['Churn']=='Yes').mean():.1%}")

    # 2. Prepare target
    y = prepare_target(df)

    # 3. Preprocessing
    logger.info("Fitting preprocessing pipeline...")
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(df)
    X = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names()
    logger.info(f"Feature matrix: {X.shape}")

    # 4. Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 5. Train all models
    logger.info("Training models...")
    results, best_name, trained_models = train_all_models(
        X_train.values, X_test.values, y_train.values, y_test.values,
        feature_names=feature_names,
    )

    # 6. Report
    logger.info("\n" + "=" * 60)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    for name, m in results.items():
        marker = " ◀ BEST" if name == best_name else ""
        logger.info(
            f"{name:<22} | ACC:{m['accuracy']:5.1f}% | F1:{m['f1']:5.1f}% "
            f"| AUC:{m['roc_auc']:5.1f}%{marker}"
        )

    # 7. Save artifacts
    model_dir = ROOT / "trained_model"
    save_artifacts(
        preprocessor=preprocessor,
        model=trained_models[best_name],
        model_name=best_name,
        results=results,
        feature_names=feature_names,
        output_dir=model_dir,
    )

    logger.info("\n✅ Training complete! Run: streamlit run app.py")
    return 0


if __name__ == "__main__":
    # Safe for Windows
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
