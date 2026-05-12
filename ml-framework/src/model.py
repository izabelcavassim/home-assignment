import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pickle
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for all models in the framework.

    Provides a unified interface for both scikit-learn and PyTorch models, making it easier to manage different types of models within the same framework.
    """

    def __init__(self, name: str, model_type: str):
        """
        Initialize base model.

        Args:
            name: Model name/identifier
            model_type: Type of model (sklearn, pytorch, etc.)
        """
        self.name = name
        self.model_type = model_type
        self.logger = logging.getLogger("ml_framework")
        self.is_trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseModel":
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training arguments

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary of metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        raise NotImplementedError

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        raise NotImplementedError


class ScikitLearnModel(BaseModel):
    """Wrapper for scikit-learn models."""

    def __init__(self, sklearn_model: Any, name: str = "sklearn_model"):
        """
        Initialize scikit-learn model wrapper.

        Args:
            sklearn_model: Instantiated scikit-learn model
            name: Model name
        """
        super().__init__(name, "sklearn")
        self.model = sklearn_model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ScikitLearnModel":
        """Train the scikit-learn model."""
        try:
            self.model.fit(X, y, **kwargs)
            self.is_trained = True
            self.logger.info(f"Trained {self.name} successfully")
            return self
        except Exception as e:
            self.logger.error(f"Failed to train {self.name}: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using scikit-learn model."""
        if not self.is_trained:
            raise ValueError(f"Model {self.name} must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (if available)."""
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(f"{self.name} does not support predict_proba")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        predictions = self.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y, predictions, average="weighted", zero_division=0),
            "f1": f1_score(y, predictions, average="weighted", zero_division=0),
        }
        return metrics

    def save(self, path: str) -> None:
        """Save model to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Saved {self.name} to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save {self.name}: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load model from disk."""
        path = Path(path)
        try:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.is_trained = True
            self.logger.info(f"Loaded {self.name} from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load {self.name}: {str(e)}")
            raise

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()

    def set_params(self, **params) -> "ScikitLearnModel":
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class ModelRegistry:
    """Registry for creating and managing models."""

    _models = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a model class."""
        cls._models[name] = model_class

    @classmethod
    def create(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            **kwargs: Model-specific arguments

        Returns:
            Model instance
        """
        if model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            sklearn_model = LogisticRegression(**kwargs)
            return ScikitLearnModel(sklearn_model, "logistic_regression")

        elif model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression

            sklearn_model = LinearRegression(**kwargs)
            return ScikitLearnModel(sklearn_model, "linear_regression")

        elif model_type == "ridge_regression":
            from sklearn.linear_model import Ridge

            sklearn_model = Ridge(**kwargs)
            return ScikitLearnModel(sklearn_model, "ridge_regression")

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            sklearn_model = RandomForestClassifier(**kwargs)
            return ScikitLearnModel(sklearn_model, "random_forest")

        elif model_type == "random_forest_regressor":
            from sklearn.ensemble import RandomForestRegressor

            sklearn_model = RandomForestRegressor(**kwargs)
            return ScikitLearnModel(sklearn_model, "random_forest_regressor")

        elif model_type == "svm":
            from sklearn.svm import SVC

            sklearn_model = SVC(**kwargs)
            return ScikitLearnModel(sklearn_model, "svm")

        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier

            sklearn_model = GradientBoostingClassifier(**kwargs)
            return ScikitLearnModel(sklearn_model, "gradient_boosting")

        elif model_type == "gradient_boosting_regressor":
            from sklearn.ensemble import GradientBoostingRegressor

            sklearn_model = GradientBoostingRegressor(**kwargs)
            return ScikitLearnModel(sklearn_model, "gradient_boosting_regressor")
        
        # TODO: Add xgboost model
        # How one would add a new model to the registry is by adding a new elif statement here.
        # The model type should be the name of the model as it will be used to create the model instance.
        # The kwargs are the arguments to pass to the model constructor.
        # The return value should be a ScikitLearnModel instance.
        # One need to import the model from the sklearn library and then create the model instance.
        # The ScikitLearnModel instance should be created by calling the ScikitLearnModel constructor with the model instance and the name of the model.
        # The ScikitLearnModel constructor should be defined in the model.py file.
        # The ScikitLearnModel constructor should take the model instance and the name of the model as arguments.
        # The ScikitLearnModel constructor should call the super() constructor with the name of the model and the model type.
        elif model_type == "xgboost":
            from xgboost import XGBRegressor

            sklearn_model = XGBRegressor(**kwargs)
            return ScikitLearnModel(sklearn_model, "xgboost")

        else:
            raise ValueError(f"Unknown model type: {model_type}")
