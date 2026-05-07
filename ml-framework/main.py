#!/usr/bin/env python3
"""
ML Framework - End-to-end machine learning pipeline for large datasets.

This script demonstrates the complete workflow: data loading, preprocessing,
model training, evaluation, and cross-validation.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.utils import setup_logging, log_execution_time
from src.config import ConfigManager
from src.data import DataLoader
from src.features import FeatureTransformer, create_default_transformer
from src.model import ModelRegistry
from src.train import Trainer


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ML Framework for analyzing large datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a random forest model on CSV data
  python main.py --data data.csv --model random_forest --output results/

  # Train with cross-validation
  python main.py --data data.csv --model svm --cv 5 --cv-strategy stratified

  # Train with custom configuration
  python main.py --data data.csv --config config.yaml --model logistic_regression
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (CSV, Parquet, HDF5)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["logistic_regression", "random_forest", "svm", "gradient_boosting"],
        help="Model type to train",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading",
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="Number of cross-validation folds (if specified, performs CV instead of train/test split)",
    )

    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "kfold"],
        help="Cross-validation strategy",
    )

    parser.add_argument(
        "--scale",
        type=str,
        default="standard",
        choices=["standard", "minmax", "robust"],
        help="Feature scaling method",
    )

    parser.add_argument(
        "--encode",
        type=str,
        default="label",
        choices=["label", "onehot"],
        help="Categorical encoding method",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser


@log_execution_time()
def main():
    """Main entry point for the ML framework."""
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging("ml_framework", level=log_level)

    logger.info("=" * 80)
    logger.info("ML Framework - Large Dataset Analysis")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = ConfigManager(env="production")
        if args.config:
            if args.config.endswith(".yaml") or args.config.endswith(".yml"):
                config.load_from_yaml(args.config)
            else:
                config.load_from_json(args.config)

        # Update config with command-line arguments
        config.update(
            data_test_size=args.test_size,
            data_batch_size=args.batch_size,
            model_type=args.model,
            experiment_output_dir=args.output,
        )

        logger.info(f"Configuration: {config}")

        # Load data
        logger.info(f"Loading data from {args.data}...")
        data_loader = DataLoader(batch_size=args.batch_size, random_state=args.seed)
        data_loader.load(args.data)

        # Display data metadata
        metadata = data_loader.get_metadata()
        logger.info(f"Data shape: {metadata['shape']}")
        logger.info(f"Numerical features: {len(metadata['numerical_features'])}")
        logger.info(f"Categorical features: {len(metadata['categorical_features'])}")
        logger.info(f"Missing values: {sum(metadata['missing_values'].values())}")

        # Handle missing values
        logger.info("Handling missing values...")
        data_loader.handle_missing_values(strategy=config.data.handle_missing)

        # Create feature transformer
        logger.info("Creating feature transformer...")
        feature_transformer = create_default_transformer(
            scale_method=args.scale, encode_method=args.encode
        )

        # Create model
        logger.info(f"Creating {args.model} model...")
        model = ModelRegistry.create(args.model, random_state=args.seed)

        # Create trainer
        trainer = Trainer(model, feature_transformer, output_dir=args.output)

        # Train and evaluate
        if args.cv:
            # Cross-validation mode
            logger.info(f"Performing {args.cv}-fold cross-validation...")
            X, y = data_loader.to_numpy()
            cv_results = trainer.cross_validate(X, y, cv=args.cv, strategy=args.cv_strategy)

            # Log results
            logger.info("Cross-validation Results:")
            for metric in cv_results["test_scores"]:
                test_score = cv_results["test_scores"][metric]
                logger.info(
                    f"  {metric}: {test_score['mean']:.4f} (+/- {test_score['std']:.4f})"
                )

        else:
            # Train/validation/test split mode
            logger.info("Splitting data...")
            X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split(
                test_size=args.test_size, validation_size=args.validation_size
            )

            logger.info(f"Training set size: {X_train.shape[0]}")
            logger.info(f"Validation set size: {X_val.shape[0]}")
            logger.info(f"Test set size: {X_test.shape[0]}")

            # Train model
            logger.info("Training model...")
            train_results = trainer.train(X_train, y_train, X_val, y_val)

            logger.info("Training Results:")
            for metric, value in train_results["train_metrics"].items():
                logger.info(f"  Train {metric}: {value:.4f}")
            if train_results["val_metrics"]:
                for metric, value in train_results["val_metrics"].items():
                    logger.info(f"  Val {metric}: {value:.4f}")

            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(X_test, y_test, save_results=True)

            logger.info("Test Results:")
            for metric in ["accuracy", "precision", "recall", "f1"]:
                if metric in test_results:
                    logger.info(f"  {metric}: {test_results[metric]:.4f}")

            # Save model
            model_path = Path(args.output) / f"{args.model}_model.pkl"
            trainer.save_model(str(model_path))

        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
