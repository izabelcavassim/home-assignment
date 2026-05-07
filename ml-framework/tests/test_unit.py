"""Unit tests for ML Framework components - Streaming Analysis Focus."""

import pytest
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

from src.streaming_data import StreamingDataLoader
from src.streaming_features import StreamingGrowthFeatures
from src.model import ModelRegistry, ScikitLearnModel
from src.utils import setup_logging


class TestStreamingDataLoader:
    """Tests for StreamingDataLoader."""

    @pytest.fixture
    def sample_streaming_data(self):
        """Create sample streaming data for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            # Generate dates (use actual dates, not ISO week strings)
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=30, freq='W-MON')
            
            # Create artist_mstreams_week.csv
            streams_data = []
            for artist_id in [1, 2, 3, 4, 5]:
                for date in dates:
                    streams_data.append({
                        'artist_id': artist_id,
                        'week_start_date': date.strftime('%Y-%m-%d'),  # Use date format
                        'number_of_streams': np.random.randint(1000000, 10000000),
                        'platform_name': 'Spotify'
                    })
            pd.DataFrame(streams_data).to_csv(data_dir / "artist_mstreams_week.csv", index=False)
            
            # Create artist_social_week.csv
            social_data = []
            for artist_id in [1, 2, 3, 4, 5]:
                for date in dates:
                    social_data.append({
                        'artist_id': artist_id,
                        'week_start_date': date.strftime('%Y-%m-%d'),  # Use date format
                        'max_following': np.random.randint(10000, 100000),
                        'platform_name': 'Spotify'
                    })
            pd.DataFrame(social_data).to_csv(data_dir / "artist_social_week.csv", index=False)
            
            # Create minimal Instagram data
            instagram_data = [{'artist_id': i, 'engagement_rate': 0.05} for i in [1, 2, 3, 4, 5]]
            pd.DataFrame(instagram_data).to_csv(data_dir / "artist_instagram.csv", index=False)
            
            # Create empty DMA and ticket files (required by loader)
            pd.DataFrame(columns=['artist_id', 'week_start_date', 'dma_id']).to_csv(
                data_dir / "artist_mstreams_dma_week.csv", index=False
            )
            pd.DataFrame(columns=['artist_id', 'start_date']).to_csv(
                data_dir / "lsecondaryticket_artist_week.csv", index=False
            )
            pd.DataFrame(columns=['artist_id', 'start_date']).to_csv(
                data_dir / "tsecondaryticket_artist_week.csv", index=False
            )
            pd.DataFrame(columns=['artist_id', 'dma_id']).to_csv(
                data_dir / "lsecondaryticket_artist_dma_week.csv", index=False
            )
            pd.DataFrame(columns=['artist_id', 'dma_id']).to_csv(
                data_dir / "tsecondaryticket_artist_dma_week.csv", index=False
            )
            
            yield str(data_dir)

    def test_streaming_data_loader_initialization(self, sample_streaming_data):
        """Test StreamingDataLoader initialization."""
        loader = StreamingDataLoader(
            data_dir=sample_streaming_data,
            artist_sample_size=3,
            min_weeks_per_artist=10,
            random_state=42
        )
        assert loader.artist_sample_size == 3
        assert loader.min_weeks_per_artist == 10
        assert loader.random_state == 42

    def test_create_joined_dataset(self, sample_streaming_data):
        """Test creating joined dataset."""
        loader = StreamingDataLoader(
            data_dir=sample_streaming_data,
            artist_sample_size=3,
            min_weeks_per_artist=10,
            random_state=42
        )
        
        df = loader.create_joined_dataset()
        
        # Check dataset structure
        assert df is not None
        assert len(df) > 0
        assert 'artist_id' in df.columns
        assert 'week' in df.columns
        assert 'number_of_streams' in df.columns
        
        # Check sampled artists
        assert len(loader.sampled_artists) <= 3

    def test_artist_sampling(self, sample_streaming_data):
        """Test artist sampling by volume."""
        loader = StreamingDataLoader(
            data_dir=sample_streaming_data,
            artist_sample_size=2,
            min_weeks_per_artist=10,
            random_state=42
        )
        
        df = loader.create_joined_dataset()
        
        # Should only have 2 artists
        assert df['artist_id'].nunique() <= 2
        assert len(loader.sampled_artists) <= 2

    def test_week_filtering(self, sample_streaming_data):
        """Test filtering by minimum weeks."""
        loader = StreamingDataLoader(
            data_dir=sample_streaming_data,
            artist_sample_size=5,
            min_weeks_per_artist=20,  # High threshold
            random_state=42
        )
        
        df = loader.create_joined_dataset()
        
        # Each artist should have >= 20 weeks
        for artist_id in df['artist_id'].unique():
            artist_weeks = df[df['artist_id'] == artist_id]['week'].nunique()
            assert artist_weeks >= 20


class TestStreamingGrowthFeatures:
    """Tests for StreamingGrowthFeatures."""

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time-series data."""
        np.random.seed(42)
        
        dates = pd.date_range('2024-01-01', periods=20, freq='W-MON')
        weeks = [f"{d.year}-W{d.isocalendar()[1]:02d}" for d in dates]
        
        data = []
        for artist_id in [1, 2]:
            for week in weeks:
                data.append({
                    'artist_id': artist_id,
                    'week': week,
                    'number_of_streams': np.random.randint(1000000, 5000000),
                    'max_following': np.random.randint(10000, 50000),
                    'engagement_rate': np.random.uniform(0.01, 0.1)
                })
        
        return pd.DataFrame(data)

    def test_feature_engineering_initialization(self):
        """Test StreamingGrowthFeatures initialization."""
        feature_eng = StreamingGrowthFeatures(
            target_col='number_of_streams',
            lags=[1, 2, 4],
            rolling_windows=[4, 8]
        )
        assert feature_eng.target_col == 'number_of_streams'
        assert feature_eng.lags == [1, 2, 4]
        assert feature_eng.rolling_windows == [4, 8]

    def test_calculate_growth_rate(self, sample_time_series):
        """Test growth rate calculation."""
        feature_eng = StreamingGrowthFeatures(target_col='number_of_streams')
        growth = feature_eng.calculate_growth_rate(
            sample_time_series, 
            'number_of_streams', 
            periods=1
        )
        
        # Growth rate should exist and have some non-null values
        assert growth is not None
        assert growth.notna().sum() > 0
        # First row of each artist should be NaN
        assert growth.isna().sum() >= 2

    def test_create_lagged_features(self, sample_time_series):
        """Test lagged feature creation."""
        feature_eng = StreamingGrowthFeatures(
            target_col='number_of_streams',
            lags=[1, 2]
        )
        
        result = feature_eng.create_lagged_features(
            sample_time_series.copy(), 
            ['max_following']
        )
        
        # Check lagged columns exist
        assert 'max_following_lag1' in result.columns
        assert 'max_following_lag2' in result.columns
        
        # Check lag1 values shifted correctly (for first artist)
        artist1 = result[result['artist_id'] == 1].sort_values('week').reset_index(drop=True)
        if len(artist1) > 1:
            # lag1 at row 1 should equal original at row 0
            assert pd.isna(artist1.loc[0, 'max_following_lag1'])  # First row should be NaN

    def test_create_rolling_features(self, sample_time_series):
        """Test rolling window feature creation."""
        feature_eng = StreamingGrowthFeatures(
            target_col='number_of_streams',
            rolling_windows=[4]
        )
        
        result = feature_eng.create_rolling_features(
            sample_time_series.copy(), 
            ['max_following']
        )
        
        # Check rolling columns exist
        assert 'max_following_roll4_mean' in result.columns
        assert 'max_following_roll4_std' in result.columns

    def test_engineer_features(self, sample_time_series):
        """Test full feature engineering pipeline."""
        feature_eng = StreamingGrowthFeatures(
            target_col='number_of_streams',
            lags=[1, 2],
            rolling_windows=[4]
        )
        
        df_features, feature_cols = feature_eng.engineer_features(sample_time_series.copy())
        
        # Check output structure
        assert df_features is not None
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        
        # Check target was created
        assert f'{feature_eng.target_col}_target' in df_features.columns
        
        # Check no target-derived features in feature_cols
        for col in feature_cols:
            assert feature_eng.target_col not in col

    def test_prepare_for_modeling(self, sample_time_series):
        """Test data preparation for modeling."""
        feature_eng = StreamingGrowthFeatures(
            target_col='number_of_streams',
            lags=[1]
        )
        
        # Engineer features first
        df_features, feature_cols = feature_eng.engineer_features(sample_time_series.copy())
        
        # Prepare for modeling
        X, y, metadata = feature_eng.prepare_for_modeling(
            df_features, 
            feature_cols, 
            drop_na=True
        )
        
        # Check shapes match
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == metadata.shape[0]
        assert X.shape[1] == len(feature_cols)
        
        # Check no NaN values after drop_na=True
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_create_momentum_features(self, sample_time_series):
        """Test momentum feature creation."""
        feature_eng = StreamingGrowthFeatures(target_col='number_of_streams')
        
        result = feature_eng.create_momentum_features(sample_time_series.copy())
        
        # Check momentum columns created
        momentum_cols = [c for c in result.columns if 'momentum' in c or 'acceleration' in c]
        assert len(momentum_cols) > 0


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_create_linear_regression(self):
        """Test creating linear regression model."""
        model = ModelRegistry.create("linear_regression")
        assert model is not None
        assert model.name == "linear_regression"
        assert model.model_type == "sklearn"

    def test_create_random_forest_regressor(self):
        """Test creating random forest regressor."""
        model = ModelRegistry.create("random_forest_regressor", random_state=42, n_estimators=50)
        assert model is not None
        assert model.name == "random_forest_regressor"

    def test_create_gradient_boosting_regressor(self):
        """Test creating gradient boosting regressor."""
        model = ModelRegistry.create("gradient_boosting_regressor", random_state=42, n_estimators=50)
        assert model is not None
        assert model.name == "gradient_boosting_regressor"

    def test_invalid_model_type(self):
        """Test creating invalid model type."""
        with pytest.raises(ValueError):
            ModelRegistry.create("invalid_model")

    def test_model_fit_and_predict(self):
        """Test model training and prediction."""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 3)
        
        model = ModelRegistry.create("linear_regression")
        model.fit(X_train, y_train)
        
        assert model.is_trained
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_model_save_and_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            
            # Create and train simple model
            np.random.seed(42)
            X_train = np.random.randn(50, 3)
            y_train = np.random.randn(50)
            
            model = ModelRegistry.create("linear_regression")
            model.fit(X_train, y_train)
            
            # Save model
            model.save(str(model_path))
            assert model_path.exists()
            
            # Load model
            model2 = ModelRegistry.create("linear_regression")
            model2.load(str(model_path))
            assert model2.is_trained
            
            # Compare predictions
            X_test = np.random.randn(10, 3)
            pred1 = model.predict(X_test)
            pred2 = model2.predict(X_test)
            np.testing.assert_array_almost_equal(pred1, pred2)

    def test_model_get_params(self):
        """Test getting model parameters."""
        model = ModelRegistry.create("random_forest_regressor", n_estimators=100, max_depth=5)
        params = model.get_params()
        
        assert 'n_estimators' in params
        assert params['n_estimators'] == 100


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
            
            # Check log file contains message
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
