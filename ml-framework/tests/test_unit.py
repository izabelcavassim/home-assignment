"""Unit tests for ML Framework components."""

import pytest
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import tempfile

from src.config import ConfigManager, DataConfig, ModelConfig
from src.data import DataLoader
from src.features import FeatureTransformer, ScalingTransformer, EncodingTransformer
from src.model import ModelRegistry, ScikitLearnModel
from src.train import Trainer
from src.utils import setup_logging


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_initialization(self):
        """Test ConfigManager initialization."""
        config = ConfigManager(env="development")
        assert config.env == "development"
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)

    def test_update_config(self):
        """Test updating configuration."""
        config = ConfigManager()
        config.update(data_test_size=0.3, model_type="svm")
        assert config.data.test_size == 0.3
        assert config.model.model_type == "svm"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ConfigManager()
        config_dict = config.to_dict()
        assert "data" in config_dict
        assert "model" in config_dict
        assert config_dict["env"] == "development"

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Save
            config = ConfigManager()
            config.update(data_test_size=0.25)
            config.save_to_yaml(str(config_path))
            assert config_path.exists()

            # Load
            config2 = ConfigManager()
            config2.load_from_yaml(str(config_path))
            assert config2.data.test_size == 0.25


class TestDataLoader:
    """Tests for DataLoader."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        df["target"] = y
        return df

    def test_load_from_dataframe(self, sample_data):
        """Test loading data from DataFrame."""
        loader = DataLoader()
        loader.load_from_dataframe(sample_data)
        assert loader.data is not None
        assert loader.data.shape == sample_data.shape

    def test_metadata_extraction(self, sample_data):
        """Test metadata extraction."""
        loader = DataLoader()
        loader.load_from_dataframe(sample_data)
        metadata = loader.get_metadata()

        assert metadata["shape"] == sample_data.shape
        assert len(metadata["numerical_features"]) == 6  # 5 features + target
        assert len(metadata["categorical_features"]) == 0

    def test_handle_missing_values(self, sample_data):
        """Test handling missing values."""
        # Add missing values
        sample_data.iloc[0, 0] = np.nan
        sample_data.iloc[1, 1] = np.nan

        loader = DataLoader()
        loader.load_from_dataframe(sample_data)
        loader.handle_missing_values(strategy="mean")

        assert loader.data.isnull().sum().sum() == 0

    def test_split_data(self, sample_data):
        """Test data splitting."""
        loader = DataLoader()
        loader.load_from_dataframe(sample_data)
        X_train, X_test, y_train, y_test = loader.split(test_size=0.2)

        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) + len(y_test) == len(sample_data)

    def test_split_with_validation(self, sample_data):
        """Test data splitting with validation set."""
        loader = DataLoader()
        loader.load_from_dataframe(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split(
            test_size=0.2, validation_size=0.1
        )

        assert len(X_train) + len(X_val) + len(X_test) == len(sample_data)

    def test_get_batches(self, sample_data):
        """Test batch generation."""
        loader = DataLoader(batch_size=10)
        loader.load_from_dataframe(sample_data)
        X, y = loader.to_numpy()

        batches = list(loader.get_batches(X, y, shuffle=False))
        assert len(batches) == 10  # 100 samples / 10 batch size
        assert batches[0][0].shape[0] == 10


class TestFeatureTransformer:
    """Tests for FeatureTransformer."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        return X

    def test_scaling_transformer(self, sample_features):
        """Test scaling transformer."""
        transformer = ScalingTransformer(method="standard")
        X_scaled = transformer.fit_transform(sample_features)

        assert X_scaled.shape == sample_features.shape
        assert np.abs(X_scaled.mean()) < 1e-10  # Mean should be ~0
        assert np.abs(X_scaled.std() - 1.0) < 1e-10  # Std should be ~1

    def test_feature_transformer_pipeline(self, sample_features):
        """Test feature transformer pipeline."""
        transformer = FeatureTransformer()
        transformer.add_scaling("standard")

        X_transformed = transformer.fit_transform(sample_features)
        assert X_transformed.shape == sample_features.shape

    def test_feature_transformer_chaining(self, sample_features):
        """Test method chaining in feature transformer."""
        transformer = (
            FeatureTransformer()
            .add_scaling("standard")
            .add_scaling("minmax")  # Apply another scaling
        )

        X_transformed = transformer.fit_transform(sample_features)
        assert X_transformed.shape == sample_features.shape


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_create_logistic_regression(self):
        """Test creating logistic regression model."""
        model = ModelRegistry.create("logistic_regression", random_state=42)
        assert model is not None
        assert model.name == "logistic_regression"

    def test_create_random_forest(self):
        """Test creating random forest model."""
        model = ModelRegistry.create("random_forest", random_state=42)
        assert model is not None
        assert model.name == "random_forest"

    def test_create_svm(self):
        """Test creating SVM model."""
        model = ModelRegistry.create("svm", random_state=42)
        assert model is not None
        assert model.name == "svm"

    def test_invalid_model_type(self):
        """Test creating invalid model type."""
        with pytest.raises(ValueError):
            ModelRegistry.create("invalid_model")


class TestTrainer:
    """Tests for Trainer."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        return X_train, X_test, y_train, y_test

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = ModelRegistry.create("random_forest")
        trainer = Trainer(model)
        assert trainer.model is not None
        assert trainer.output_dir.exists()

    def test_train_and_evaluate(self, sample_dataset):
        """Test training and evaluation."""
        X_train, X_test, y_train, y_test = sample_dataset

        model = ModelRegistry.create("random_forest", random_state=42)
        trainer = Trainer(model)

        # Train
        trainer.train(X_train, y_train)
        assert model.is_trained

        # Evaluate
        metrics = trainer.evaluate(X_test, y_test, save_results=False)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_cross_validation(self, sample_dataset):
        """Test cross-validation."""
        X_train, _, y_train, _ = sample_dataset

        model = ModelRegistry.create("random_forest", random_state=42)
        trainer = Trainer(model)

        cv_results = trainer.cross_validate(X_train, y_train, cv=3)
        assert "test_scores" in cv_results
        assert "accuracy" in cv_results["test_scores"]

    def test_save_and_load_model(self, sample_dataset):
        """Test saving and loading model."""
        X_train, X_test, y_train, y_test = sample_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            # Train and save
            model = ModelRegistry.create("random_forest", random_state=42)
            trainer = Trainer(model)
            trainer.train(X_train, y_train)
            trainer.save_model(str(model_path))
            assert model_path.exists()

            # Load and evaluate
            model2 = ModelRegistry.create("random_forest", random_state=42)
            trainer2 = Trainer(model2)
            trainer2.load_model(str(model_path))
            metrics = trainer2.evaluate(X_test, y_test, save_results=False)
            assert "accuracy" in metrics


class TestLogging:
    """Tests for logging utilities."""

    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging("test_logger", level=logging.INFO)
        assert logger is not None
        assert logger.name == "test_logger"

    def test_logging_to_file(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging("test_logger", log_file=str(log_file))
            logger.info("Test message")
            assert log_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
