import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


class Transformer(ABC):
    """Base class for feature transformers."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Transformer":
        """Fit transformer on data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(X).transform(X)


class ScalingTransformer(Transformer):
    """Scaling transformer for numerical features."""

    def __init__(self, method: str = "standard"):
        """
        Initialize scaling transformer.

        Args:
            method: Scaling method (standard, minmax, robust)
        """
        self.method = method
        self.scaler = None

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(self, X: np.ndarray) -> "ScalingTransformer":
        """Fit scaler on data."""
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)


class EncodingTransformer(Transformer):
    """Encoding transformer for categorical features."""

    def __init__(self, method: str = "label"):
        """
        Initialize encoding transformer.

        Args:
            method: Encoding method (label, onehot)
        """
        self.method = method
        self.encoders = {}

    def fit(self, X: pd.DataFrame) -> "EncodingTransformer":
        """Fit encoders on categorical data."""
        for col in X.columns:
            if self.method == "label":
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str))
                self.encoders[col] = encoder
            elif self.method == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform categorical data."""
        if self.method == "label":
            X_encoded = X.copy()
            for col in X.columns:
                if col in self.encoders:
                    X_encoded[col] = self.encoders[col].transform(X[col].astype(str))
            return X_encoded.values
        elif self.method == "onehot":
            transformed_parts = []
            for col in X.columns:
                if col in self.encoders:
                    encoded = self.encoders[col].transform(X[[col]])
                    transformed_parts.append(encoded)
            return np.hstack(transformed_parts) if transformed_parts else X.values


class DimensionalityReductionTransformer(Transformer):
    """Dimensionality reduction transformer."""

    def __init__(self, method: str = "pca", n_components: int = 10):
        """
        Initialize dimensionality reduction transformer.

        Args:
            method: Reduction method (pca)
            n_components: Number of components to keep
        """
        self.method = method
        self.n_components = n_components
        self.reducer = None

        if method == "pca":
            self.reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

    def fit(self, X: np.ndarray) -> "DimensionalityReductionTransformer":
        """Fit reducer on data."""
        self.reducer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted reducer."""
        return self.reducer.transform(X)


class SimilarityTransformer(Transformer):
    """Transformer for computing string similarity features."""

    def __init__(self, similarity_type: str = "tfidf"):
        """
        Initialize similarity transformer.

        Args:
            similarity_type: Type of similarity (tfidf, cosine)
        """
        self.similarity_type = similarity_type
        self.vectorizer = None

    def fit(self, X: np.ndarray) -> "SimilarityTransformer":
        """Fit similarity transformer on data."""
        if self.similarity_type == "tfidf":
            # Convert to strings if needed
            X_str = np.array([str(x) for x in X.flatten()])
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
            self.vectorizer.fit(X_str)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using similarity metrics."""
        if self.similarity_type == "tfidf" and self.vectorizer:
            X_str = np.array([str(x) for x in X.flatten()])
            return self.vectorizer.transform(X_str).toarray()
        return X


class EmbeddingTransformer(Transformer):
    """Transformer for generating embeddings from text."""

    def __init__(self, embedding_type: str = "tfidf", max_features: int = 100):
        """
        Initialize embedding transformer.

        Args:
            embedding_type: Type of embedding (tfidf)
            max_features: Maximum number of features
        """
        self.embedding_type = embedding_type
        self.max_features = max_features
        self.vectorizer = None

    def fit(self, X: np.ndarray) -> "EmbeddingTransformer":
        """Fit embedding transformer on data."""
        if self.embedding_type == "tfidf":
            X_str = np.array([str(x) for x in X.flatten()])
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 3),
                max_features=self.max_features
            )
            self.vectorizer.fit(X_str)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to embeddings."""
        if self.embedding_type == "tfidf" and self.vectorizer:
            X_str = np.array([str(x) for x in X.flatten()])
            return self.vectorizer.transform(X_str).toarray()
        return X


class FeatureTransformer:
    """
    Composable feature transformation pipeline.

    Automatically detects feature types and applies appropriate transformations.
    """

    def __init__(self):
        """Initialize feature transformer."""
        self.logger = logging.getLogger("ml_framework")
        self.transformers: List[Tuple[str, Transformer]] = []
        self.feature_types: Dict[str, str] = {}
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []

    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect feature types.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping feature names to types
        """
        feature_types = {}

        for col in df.columns:
            if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
                feature_types[col] = "numerical"
                self.numerical_features.append(col)
            else:
                feature_types[col] = "categorical"
                self.categorical_features.append(col)

        self.feature_types = feature_types
        self.logger.info(
            f"Detected {len(self.numerical_features)} numerical and "
            f"{len(self.categorical_features)} categorical features"
        )
        return feature_types

    def add_transformer(self, name: str, transformer: Transformer) -> "FeatureTransformer":
        """
        Add a transformer to the pipeline.

        Args:
            name: Name of the transformer
            transformer: Transformer instance

        Returns:
            Self for method chaining
        """
        self.transformers.append((name, transformer))
        self.logger.debug(f"Added transformer: {name}")
        return self

    def add_scaling(self, method: str = "standard") -> "FeatureTransformer":
        """
        Add scaling transformer.

        Args:
            method: Scaling method (standard, minmax, robust)

        Returns:
            Self for method chaining
        """
        self.add_transformer(f"scaling_{method}", ScalingTransformer(method))
        return self

    def add_encoding(self, method: str = "label") -> "FeatureTransformer":
        """
        Add encoding transformer.

        Args:
            method: Encoding method (label, onehot)

        Returns:
            Self for method chaining
        """
        self.add_transformer(f"encoding_{method}", EncodingTransformer(method))
        return self

    def add_dimensionality_reduction(
        self, method: str = "pca", n_components: int = 10
    ) -> "FeatureTransformer":
        """
        Add dimensionality reduction transformer.

        Args:
            method: Reduction method (pca)
            n_components: Number of components

        Returns:
            Self for method chaining
        """
        self.add_transformer(
            f"reduction_{method}_{n_components}",
            DimensionalityReductionTransformer(method, n_components),
        )
        return self

    def add_similarity(self, similarity_type: str = "tfidf") -> "FeatureTransformer":
        """
        Add similarity transformer.

        Args:
            similarity_type: Type of similarity (tfidf, cosine)

        Returns:
            Self for method chaining
        """
        self.add_transformer(f"similarity_{similarity_type}", SimilarityTransformer(similarity_type))
        return self

    def add_embedding(self, embedding_type: str = "tfidf", max_features: int = 100) -> "FeatureTransformer":
        """
        Add embedding transformer.

        Args:
            embedding_type: Type of embedding (tfidf)
            max_features: Maximum number of features

        Returns:
            Self for method chaining
        """
        self.add_transformer(
            f"embedding_{embedding_type}_{max_features}",
            EmbeddingTransformer(embedding_type, max_features)
        )
        return self

    def fit(self, X: np.ndarray) -> "FeatureTransformer":
        """
        Fit all transformers in the pipeline.

        Args:
            X: Input features

        Returns:
            Self for method chaining
        """
        current_X = X
        for name, transformer in self.transformers:
            self.logger.debug(f"Fitting transformer: {name}")
            transformer.fit(current_X)
            current_X = transformer.transform(current_X)

        self.logger.info(f"Fitted {len(self.transformers)} transformers")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data through the pipeline.

        Args:
            X: Input features

        Returns:
            Transformed features
        """
        current_X = X
        for name, transformer in self.transformers:
            current_X = transformer.transform(current_X)
        return current_X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.

        Args:
            X: Input features

        Returns:
            Transformed features
        """
        return self.fit(X).transform(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from transformers (if available).

        Returns:
            Feature importance array or None
        """
        for name, transformer in self.transformers:
            if isinstance(transformer, DimensionalityReductionTransformer):
                if hasattr(transformer.reducer, "explained_variance_ratio_"):
                    return transformer.reducer.explained_variance_ratio_
        return None

    def __repr__(self) -> str:
        """String representation."""
        transformer_names = [name for name, _ in self.transformers]
        return f"FeatureTransformer(transformers={transformer_names})"


def create_default_transformer(
    scale_method: str = "standard", encode_method: str = "label"
) -> FeatureTransformer:
    """
    Create a default feature transformer pipeline.

    Args:
        scale_method: Scaling method for numerical features
        encode_method: Encoding method for categorical features

    Returns:
        Configured FeatureTransformer instance
    """
    transformer = FeatureTransformer()
    transformer.add_scaling(scale_method)
    transformer.add_encoding(encode_method)
    return transformer
