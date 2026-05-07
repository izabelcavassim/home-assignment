import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    KFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.model import BaseModel
from src.features import FeatureTransformer


class Trainer:
    """
    Training engine for ML models with validation, metrics tracking, and cross-validation.
    """

    def __init__(
        self,
        model: BaseModel,
        feature_transformer: Optional[FeatureTransformer] = None,
        output_dir: str = "results",
    ):
        """
        Initialize trainer.

        Args:
            model: Model instance to train
            feature_transformer: Optional feature transformer pipeline
            output_dir: Directory to save results
        """
        self.model = model
        self.feature_transformer = feature_transformer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ml_framework")

        self.training_history = {
            "train_metrics": [],
            "val_metrics": [],
            "test_metrics": {},
        }
        self.best_model_path = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training arguments

        Returns:
            Dictionary with training results
        """
        try:
            # Apply feature transformation if available
            if self.feature_transformer is not None:
                self.logger.info("Applying feature transformation...")
                X_train = self.feature_transformer.fit_transform(X_train)
                if X_val is not None:
                    X_val = self.feature_transformer.transform(X_val)

            # Train model
            self.logger.info(f"Training {self.model.name}...")
            self.model.fit(X_train, y_train, **kwargs)

            # Evaluate on training set
            train_metrics = self._evaluate(X_train, y_train, "train")
            self.training_history["train_metrics"].append(train_metrics)

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_metrics = self._evaluate(X_val, y_val, "validation")
                self.training_history["val_metrics"].append(val_metrics)
                self.logger.info(f"Validation metrics: {val_metrics}")

            self.logger.info(f"Training completed successfully")
            return {
                "train_metrics": train_metrics,
                "val_metrics": self.training_history["val_metrics"][-1]
                if self.training_history["val_metrics"]
                else None,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            save_results: Whether to save results to disk

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Apply feature transformation if available
            if self.feature_transformer is not None:
                X_test = self.feature_transformer.transform(X_test)

            # Get predictions
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            self.training_history["test_metrics"] = metrics

            # Generate classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics["classification_report"] = report

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            self.logger.info(f"Test metrics: {metrics}")

            if save_results:
                self._save_results(metrics, y_test, y_pred)

            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        strategy: str = "stratified",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds
            strategy: Cross-validation strategy (stratified, kfold)

        Returns:
            Dictionary with cross-validation results
        """
        try:
            # Apply feature transformation if available
            if self.feature_transformer is not None:
                X = self.feature_transformer.fit_transform(X)

            # Select CV strategy
            if strategy == "stratified":
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            elif strategy == "kfold":
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                raise ValueError(f"Unknown CV strategy: {strategy}")

            # Perform cross-validation
            scoring = {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted",
            }

            cv_results = cross_validate(
                self.model.model, X, y, cv=cv_splitter, scoring=scoring, return_train_score=True
            )

            # Aggregate results
            results = {
                "cv_folds": cv,
                "strategy": strategy,
                "train_scores": {},
                "test_scores": {},
            }

            for metric in scoring.keys():
                train_key = f"train_{metric}"
                test_key = f"test_{metric}"

                results["train_scores"][metric] = {
                    "mean": cv_results[train_key].mean(),
                    "std": cv_results[train_key].std(),
                    "scores": cv_results[train_key].tolist(),
                }
                results["test_scores"][metric] = {
                    "mean": cv_results[test_key].mean(),
                    "std": cv_results[test_key].std(),
                    "scores": cv_results[test_key].tolist(),
                }

            self.logger.info(f"Cross-validation results: {results}")
            return results

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            raise

    def _evaluate(self, X: np.ndarray, y: np.ndarray, set_name: str) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        y_pred = self.model.predict(X)
        metrics = self._calculate_metrics(y, y_pred)
        self.logger.info(f"{set_name.capitalize()} metrics: {metrics}")
        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Add ROC-AUC if binary classification
        if len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            except Exception:
                pass

        return metrics

    def _save_results(
        self, metrics: Dict[str, Any], y_test: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.output_dir / f"results_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = results_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_to_save = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
            json.dump(metrics_to_save, f, indent=2)

        # Save predictions
        predictions_file = results_dir / "predictions.json"
        with open(predictions_file, "w") as f:
            json.dump(
                {
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                },
                f,
                indent=2,
            )

        # Save model
        model_file = results_dir / f"{self.model.name}.pkl"
        self.model.save(str(model_file))
        self.best_model_path = model_file

        self.logger.info(f"Results saved to {results_dir}")

    def save_model(self, path: str) -> None:
        """Save trained model."""
        self.model.save(path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained model."""
        self.model.load(path)
        self.logger.info(f"Model loaded from {path}")

    def get_training_history(self) -> Dict[str, Any]:
        """Get training history."""
        return self.training_history

    def __repr__(self) -> str:
        """String representation."""
        return f"Trainer(model={self.model.name}, output_dir={self.output_dir})"
