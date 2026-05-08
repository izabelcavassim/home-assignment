"""
Streaming Growth Feature Engineering

Module for creating time-series features with proper temporal alignment
and leakage prevention for streaming growth prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple


logger = logging.getLogger("ml_framework")


class StreamingGrowthFeatures:
    """
    Engineer features for streaming growth prediction with temporal awareness.
    
    Key principles:
    - All features must be lagged to prevent temporal leakage
    - Target is week t, features are from week t-lag and earlier
    - Growth rates calculated with proper handling of zeros/negatives
    """
    
    def __init__(
        self,
        target_col: str = 'number_of_streams',
        lags: List[int] = [1, 2, 4],
        rolling_windows: List[int] = [4, 8],
        min_periods_pct: float = 0.5,
        use_cumulative: bool = False,
        include_cumulative_features: bool = False
    ):
        """
        Initialize feature engineering pipeline.
        
        Args:
            target_col: Column to predict (streaming metric)
            lags: Lag periods for features (in weeks)
            rolling_windows: Window sizes for rolling aggregates
            min_periods_pct: Minimum fraction of periods required for rolling stats
            use_cumulative: If True, predict cumulative streams instead of growth rates
            include_cumulative_features: If True, include cumulative as features (causes leakage!)
        """
        self.target_col = target_col
        self.lags = sorted(lags)
        self.rolling_windows = sorted(rolling_windows)
        self.min_periods_pct = min_periods_pct
        self.use_cumulative = use_cumulative
        self.include_cumulative_features = include_cumulative_features
        
        logger.info("Initialized StreamingGrowthFeatures:")
        logger.info(f"  Target column: {self.target_col}")
        logger.info(f"  Lags: {self.lags}")
        logger.info(f"  Rolling windows: {self.rolling_windows}")
        logger.info(f"  Use cumulative: {self.use_cumulative}")
        logger.info(f"  Include cumulative features: {self.include_cumulative_features}")
        if self.use_cumulative and self.include_cumulative_features:
            logger.warning("  ⚠️  WARNING: Including cumulative features causes data leakage!")
            logger.warning("     Use only for comparison analysis, not production!")
    
    def calculate_growth_rate(
        self,
        df: pd.DataFrame,
        col: str,
        periods: int = 1,
        method: str = 'pct_change'
    ) -> pd.Series:
        """
        Calculate growth rate with proper handling of edge cases.
        
        Args:
            df: DataFrame with time-series data
            col: Column to calculate growth for
            periods: Number of periods for growth calculation
            method: 'pct_change' or 'log_diff'
        
        Returns:
            Series with growth rates
        """
        if method == 'pct_change':
            # Standard percentage change
            growth = df.groupby('artist_id')[col].pct_change(periods=periods)
        elif method == 'log_diff':
            # Log difference (more stable for large changes)
            growth = df.groupby('artist_id')[col].transform(
                lambda x: np.log1p(x).diff(periods=periods)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Handle infinities and extreme values
        growth = growth.replace([np.inf, -np.inf], np.nan)
        growth = growth.clip(-10, 10)  # Cap at +/- 1000% change
        
        return growth

    # I have created lagged features for 1, 2, and 4 weeks to capture short-term trends without overfitting.
    # Lagged features are essential to prevent temporal leakage, ensuring that the model only has access to
    # information that would have been available at the time of prediction.
    def create_lagged_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lagged versions of features to prevent temporal leakage.
        
        Args:
            df: DataFrame with time-series data
            cols: Columns to lag
            lags: Lag periods (uses self.lags if None)
        
        Returns:
            DataFrame with lagged features added
        """
        if lags is None:
            lags = self.lags
        
        result_df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            for lag in lags:
                lagged_col = f"{col}_lag{lag}"
                result_df[lagged_col] = result_df.groupby('artist_id')[col].shift(lag)
                logger.debug(f"Created {lagged_col}")
        
        return result_df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window statistics (lagged to prevent leakage).
        
        Args:
            df: DataFrame with time-series data
            cols: Columns to aggregate
            windows: Window sizes (uses self.rolling_windows if None)
        
        Returns:
            DataFrame with rolling features added
        """
        if windows is None:
            windows = self.rolling_windows
        
        result_df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for window in windows:
                min_periods = max(1, int(window * self.min_periods_pct))
                
                # Rolling mean (lagged by 1 to prevent leakage)
                mean_col = f"{col}_roll{window}_mean"
                result_df[mean_col] = result_df.groupby('artist_id')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
                )
                
                # Rolling std (momentum indicator)
                std_col = f"{col}_roll{window}_std"
                result_df[std_col] = result_df.groupby('artist_id')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).std()
                )
                
                logger.debug(f"Created {mean_col} and {std_col}")
        
        return result_df
    
    def create_growth_features(
        self,
        df: pd.DataFrame,
        cols: List[str]
    ) -> pd.DataFrame:
        """
        Create growth rate features for specified columns.
        
        Args:
            df: DataFrame with time-series data
            cols: Columns to calculate growth for
        
        Returns:
            DataFrame with growth features added
        """
        result_df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            # Izabel This was the first model I have used but shown to be very unstable with outliers and zeros.
            # Keeping for comparison.
            # Week-over-week growth
            growth_col = f"{col}_growth_1w"
            result_df[growth_col] = self.calculate_growth_rate(result_df, col, periods=1)
            
            # 4-week growth (monthly)
            growth_col_4w = f"{col}_growth_4w"
            result_df[growth_col_4w] = self.calculate_growth_rate(result_df, col, periods=4)
            
            logger.debug(f"Created growth features for {col}")
        
        return result_df
    
    def create_momentum_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create momentum and trend features.
        
        Args:
            df: DataFrame with time-series data
            target_col: Column to create momentum for (uses self.target_col if None)
        
        Returns:
            DataFrame with momentum features added
        """
        if target_col is None:
            target_col = self.target_col
        
        result_df = df.copy()
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return result_df
        
        # Acceleration (change in growth rate)
        growth_1w = self.calculate_growth_rate(result_df, target_col, periods=1)
        result_df[f'{target_col}_acceleration'] = growth_1w.groupby(result_df['artist_id']).diff()
        
        # Momentum score (recent growth vs historical average)
        for window in [8, 12]:
            recent_growth = result_df.groupby('artist_id')[target_col].transform(
                lambda x: x.pct_change(periods=1).shift(1).rolling(4, min_periods=2).mean()
            )
            historical_growth = result_df.groupby('artist_id')[target_col].transform(
                lambda x: x.pct_change(periods=1).shift(1).rolling(window, min_periods=int(window*0.5)).mean()
            )
            result_df[f'{target_col}_momentum_{window}w'] = recent_growth - historical_growth
        
        logger.debug("Created momentum features")
        return result_df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Main feature engineering pipeline.
        
        Args:
            df: Raw dataset with artist, week, and metrics
        
        Returns:
            Tuple of (engineered DataFrame, list of feature column names)
        """
        logger.info("Engineering features...")
        logger.info(f"Input shape: {df.shape}")
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Identify numeric columns for feature engineering
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID, week, and target from feature engineering
        exclude_cols = ['artist_id', 'week', self.target_col]
        
        # IMPORTANT: If using cumulative mode, exclude cumulative column from features
        # to prevent data leakage (cumulative_t is trivially predictable from cumulative_{t-1})
        # UNLESS user explicitly requests it for comparison analysis
        if self.use_cumulative and 'cleaned_cumulative_spotify_streams' in result_df.columns:
            if not self.include_cumulative_features:
                exclude_cols.append('cleaned_cumulative_spotify_streams')
                logger.info("  Excluding cumulative streams from features (prevents leakage)")
            else:
                logger.warning("  ⚠️  Including cumulative streams in features (DATA LEAKAGE!)")
                logger.warning("     R² will be artificially high. Use only for comparison!")
        
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        logger.info(f"Engineering features for {len(numeric_cols)} numeric columns")
        
        # 1. Create target variables
        if self.use_cumulative:
            # Use cumulative streams (if available) - better for avoiding outliers
            cumulative_col = 'cleaned_cumulative_spotify_streams'
            if cumulative_col in result_df.columns:
                logger.info("Using cumulative streams as target")
                # Future cumulative value (1 week ahead)
                result_df[f'{self.target_col}_target'] = result_df.groupby('artist_id')[cumulative_col].shift(-1)
                logger.info(f"Created target: {self.target_col}_target (predicting cumulative streams 1 week ahead)")
            else:
                logger.warning(f"cumulative column not found, falling back to growth rates")
                self.use_cumulative = False
        
        if not self.use_cumulative:
            # Use growth rates (original behavior)
            logger.info("Using growth rates as target")
            result_df[f'{self.target_col}_target'] = self.calculate_growth_rate(
                result_df, self.target_col, periods=1
            )
            logger.info(f"Created target: {self.target_col}_target (predicting 1-week growth)")
        
        # 2. Create lagged features (prevent leakage)
        key_cols = [c for c in numeric_cols if any(
            keyword in c.lower() for keyword in ['stream', 'follower', 'engagement', 'ticket', 'sales']
        )]
        if key_cols:
            result_df = self.create_lagged_features(result_df, key_cols)
            logger.info(f"Created lagged features for {len(key_cols)} columns")
        
        # 3. Create rolling aggregates
        if key_cols:
            result_df = self.create_rolling_features(result_df, key_cols[:5])  # Limit to top 5 to save time
            logger.info("Created rolling window features")
        
        # 4. Create growth features for key metrics
        growth_cols = [c for c in numeric_cols if 'stream' in c.lower() or 'follower' in c.lower()][:3]
        if growth_cols:
            result_df = self.create_growth_features(result_df, growth_cols)
            logger.info(f"Created growth features for {len(growth_cols)} columns")
        
        # 5. Create momentum features
        result_df = self.create_momentum_features(result_df)
        logger.info("Created momentum features")
        
        # Identify feature columns (exclude original data and target)
        # IMPORTANT: Exclude any features derived from the target variable itself
        # to prevent temporal leakage. Only use external predictors.
        feature_cols = []
        for c in result_df.columns:
            # Include lagged, rolling, growth, momentum features
            is_feature = any([
                c.endswith('_lag1'), c.endswith('_lag2'), c.endswith('_lag4'),
                '_roll' in c, '_growth' in c, '_momentum' in c, '_acceleration' in c
            ])
            
            # EXCLUDE features derived from the target column itself
            is_target_derived = self.target_col in c
            
            if is_feature and not is_target_derived:
                feature_cols.append(c)
        
        logger.info(f"Output shape: {result_df.shape}")
        logger.info(f"Total features created: {len(feature_cols)}")
        logger.info(f"Excluded target-derived features to prevent leakage")
        logger.info(f"Missing values after feature engineering: {result_df[feature_cols].isnull().sum().sum():,}")
        
        return result_df, feature_cols
    
    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for modeling by separating X, y and handling missing values.
        
        Args:
            df: DataFrame with engineered features
            feature_cols: List of feature column names
            drop_na: Whether to drop rows with missing values
        
        Returns:
            Tuple of (X, y, metadata) where metadata includes artist_id and week
        """
        target_col = f'{self.target_col}_target'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        metadata = df[['artist_id', 'week']].copy()
        
        logger.info(f"Preparing for modeling:")
        logger.info(f"  Features shape: {X.shape}")
        logger.info(f"  Target shape: {y.shape}")
        logger.info(f"  Missing in features: {X.isnull().sum().sum():,}")
        logger.info(f"  Missing in target: {y.isnull().sum():,}")
        
        if drop_na:
            # Drop rows where target or any feature is missing
            valid_idx = (~y.isnull()) & (~X.isnull().any(axis=1))
            X = X[valid_idx]
            y = y[valid_idx]
            metadata = metadata[valid_idx]
            logger.info(f"  After dropping NA: {X.shape[0]:,} samples")
        
        return X, y, metadata
