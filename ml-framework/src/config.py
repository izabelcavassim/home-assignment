import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    normalize: bool = True
    handle_missing: str = "mean"  # mean, median, drop, forward_fill


@dataclass
class ModelConfig:
    """Configuration for model training."""

    model_type: str = "random_forest"  # random_forest, logistic_regression, svm, neural_net
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    early_stopping_metric: str = "loss"
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    metrics: list = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, timeseries


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    name: str = "default_experiment"
    description: str = ""
    save_model: bool = True
    save_predictions: bool = True
    save_metrics: bool = True
    output_dir: str = "results"


@dataclass
class EntityMatchingConfig:
    """Configuration for entity matching tasks."""

    similarity_metrics: list = field(default_factory=lambda: ["levenshtein", "jaccard"])
    text_fields: list = field(default_factory=list)
    numeric_fields: list = field(default_factory=list)
    categorical_fields: list = field(default_factory=list)
    field_weights: dict = field(default_factory=dict)
    blocking_enabled: bool = True
    threshold: float = 0.5
    negative_sampling_ratio: float = 1.0
    record1_prefix: str = "record1_"
    record2_prefix: str = "record2_"
    label_column: str = "label"


class ConfigManager:
    """
    Centralized configuration management for the ML framework.

    Supports loading from YAML/JSON files, environment variables, and programmatic updates.
    """

    def __init__(self, env: str = "development"):
        """
        Initialize ConfigManager.

        Args:
            env: Environment mode (development, testing, production)
        """
        self.env = env
        self.logger = logging.getLogger("ml_framework")

        # Initialize default configs
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.experiment = ExperimentConfig()
        self.entity_matching = EntityMatchingConfig()

    def load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return

        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            self._update_from_dict(config_dict)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def load_from_json(self, config_path: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return

        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            self._update_from_dict(config_dict)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration objects from dictionary."""
        if "data" in config_dict:
            self.data = DataConfig(**config_dict["data"])
        if "model" in config_dict:
            self.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            self.training = TrainingConfig(**config_dict["training"])
        if "evaluation" in config_dict:
            self.evaluation = EvaluationConfig(**config_dict["evaluation"])
        if "experiment" in config_dict:
            self.experiment = ExperimentConfig(**config_dict["experiment"])
        if "entity_matching" in config_dict:
            self.entity_matching = EntityMatchingConfig(**config_dict["entity_matching"])

    def update(self, **kwargs) -> None:
        """
        Update configuration programmatically.

        Args:
            **kwargs: Configuration updates (e.g., data_test_size=0.3, model_type='svm')
        """
        for key, value in kwargs.items():
            parts = key.split("_", 1)
            if len(parts) == 2:
                config_name, attr_name = parts
                if hasattr(self, config_name):
                    config_obj = getattr(self, config_name)
                    if hasattr(config_obj, attr_name):
                        setattr(config_obj, attr_name, value)
                        self.logger.debug(f"Updated {config_name}.{attr_name} = {value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to dictionary."""
        return {
            "env": self.env,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "experiment": asdict(self.experiment),
        }

    def save_to_yaml(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save YAML configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            self.logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config to {output_path}: {str(e)}")
            raise

    def save_to_json(self, output_path: str) -> None:
        """
        Save current configuration to JSON file.

        Args:
            output_path: Path to save JSON configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            self.logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config to {output_path}: {str(e)}")
            raise

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(env={self.env}, data={self.data}, model={self.model})"
