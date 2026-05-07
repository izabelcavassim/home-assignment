import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)


class DataLoader:
    """
    Efficient data loader for handling large datasets with streaming and batching.

    Supports multiple file formats (CSV, Parquet, HDF5) and mixed data types.
    """

    def __init__(self, batch_size: int = 32, random_state: int = 42):
        """
        Initialize DataLoader.

        Args:
            batch_size: Batch size for loading data
            random_state: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.random_state = random_state
        self.logger = logging.getLogger("ml_framework")
        self.data = None
        self.metadata = {}

    def load(self, path: str, file_format: Optional[str] = None) -> "DataLoader":
        """
        Load data from file.

        Args:
            path: Path to data file
            file_format: File format (csv, parquet, hdf5). Auto-detected if None.

        Returns:
            Self for method chaining
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        # Auto-detect format if not specified
        if file_format is None:
            file_format = path.suffix.lstrip(".").lower()

        try:
            if file_format == "csv":
                self.data = pd.read_csv(path)
            elif file_format == "parquet":
                self.data = pd.read_parquet(path)
            elif file_format in ["h5", "hdf5"]:
                self.data = pd.read_hdf(path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            self.logger.info(f"Loaded data from {path}: shape {self.data.shape}")
            self._profile_data()
            return self
        except Exception as e:
            self.logger.error(f"Failed to load data from {path}: {str(e)}")
            raise

    def load_from_dataframe(self, df: pd.DataFrame) -> "DataLoader":
        """
        Load data from existing DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Self for method chaining
        """
        self.data = df.copy()
        self.logger.info(f"Loaded data from DataFrame: shape {self.data.shape}")
        self._profile_data()
        return self

    def _profile_data(self) -> None:
        """Profile data to extract metadata."""
        if self.data is None:
            return

        self.metadata = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numerical_features": self.data.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "categorical_features": self.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get data metadata."""
        return self.metadata

    def handle_missing_values(self, strategy: str = "mean") -> "DataLoader":
        """
        Handle missing values in data.

        Args:
            strategy: Strategy for handling missing values
                     (mean, median, drop, forward_fill, backward_fill)

        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded")

        try:
            if strategy == "drop":
                self.data = self.data.dropna()
            elif strategy == "mean":
                numerical_cols = self.data.select_dtypes(include=[np.number]).columns
                self.data[numerical_cols] = self.data[numerical_cols].fillna(
                    self.data[numerical_cols].mean()
                )
            elif strategy == "median":
                numerical_cols = self.data.select_dtypes(include=[np.number]).columns
                self.data[numerical_cols] = self.data[numerical_cols].fillna(
                    self.data[numerical_cols].median()
                )
            elif strategy == "forward_fill":
                self.data = self.data.fillna(method="ffill")
            elif strategy == "backward_fill":
                self.data = self.data.fillna(method="bfill")
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            self.logger.info(f"Handled missing values using {strategy} strategy")
            return self
        except Exception as e:
            self.logger.error(f"Failed to handle missing values: {str(e)}")
            raise

    def remove_outliers(self, method: str = "iqr", threshold: float = 1.5) -> "DataLoader":
        """
        Remove outliers from numerical features.

        Args:
            method: Method for outlier detection (iqr, zscore)
            threshold: Threshold for outlier detection

        Returns:
            Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded")

        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns

            if method == "iqr":
                Q1 = self.data[numerical_cols].quantile(0.25)
                Q3 = self.data[numerical_cols].quantile(0.75)
                IQR = Q3 - Q1
                mask = ~(
                    (self.data[numerical_cols] < (Q1 - threshold * IQR))
                    | (self.data[numerical_cols] > (Q3 + threshold * IQR))
                ).any(axis=1)
                self.data = self.data[mask]

            elif method == "zscore":
                from scipy import stats

                z_scores = np.abs(stats.zscore(self.data[numerical_cols]))
                mask = (z_scores < threshold).all(axis=1)
                self.data = self.data[mask]

            else:
                raise ValueError(f"Unknown method: {method}")

            self.logger.info(f"Removed outliers using {method} method")
            return self
        except Exception as e:
            self.logger.error(f"Failed to remove outliers: {str(e)}")
            raise

    def split(
        self, test_size: float = 0.2, validation_size: Optional[float] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data into train/test or train/validation/test sets.

        Args:
            test_size: Fraction of data for test set
            validation_size: Fraction of data for validation set (optional)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or
            (X_train, X_val, X_test, y_train, y_val, y_test) if validation_size is specified
        """
        if self.data is None:
            raise ValueError("No data loaded")

        try:
            # Separate features and target (assume last column is target)
            X = self.data.iloc[:, :-1].values
            y = self.data.iloc[:, -1].values

            if validation_size is not None:
                # Split into train and temp (val + test)
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=(test_size + validation_size), random_state=self.random_state
                )
                # Split temp into val and test
                val_ratio = validation_size / (test_size + validation_size)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=1 - val_ratio, random_state=self.random_state
                )
                self.logger.info(
                    f"Split data: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}"
                )
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                self.logger.info(f"Split data: train {X_train.shape}, test {X_test.shape}")
                return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Failed to split data: {str(e)}")
            raise

    def get_batches(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True):
        """
        Generate batches of data.

        Args:
            X: Features
            y: Labels
            shuffle: Whether to shuffle data

        Yields:
            Tuples of (X_batch, y_batch)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.RandomState(self.random_state).shuffle(indices)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert data to numpy arrays (features and target).

        Returns:
            Tuple of (X, y)
        """
        if self.data is None:
            raise ValueError("No data loaded")

        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        return X, y

    def to_dataframe(self) -> pd.DataFrame:
        """Get data as DataFrame."""
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.copy()