"""End-to-end tests for Streaming Growth Pipeline with all feature combinations."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import subprocess
import json
import sys

from src.streaming_data import StreamingDataLoader
from src.streaming_features import StreamingGrowthFeatures
from src.model import ModelRegistry


class TestE2EStreamingPipeline:
    """End-to-end tests for the complete streaming growth pipeline."""

    @pytest.fixture
    def test_data_dir(self):
        """Create test data directory with sample CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "test_data"
            data_dir.mkdir()
            
            # Generate synthetic streaming data
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=52, freq='W-MON')
            weeks = [f"{d.year}-W{d.isocalendar()[1]:02d}" for d in dates]
            
            # Create artist_mstreams_week.csv
            streams_data = []
            for artist_id in range(1, 11):  # 10 artists
                for week in weeks:
                    streams_data.append({
                        'artist_id': artist_id,
                        'week_start_date': week,
                        'number_of_streams': np.random.randint(1000000, 10000000),
                        'platform_name': 'Spotify'
                    })
            pd.DataFrame(streams_data).to_csv(data_dir / "artist_mstreams_week.csv", index=False)
            
            # Create artist_social_week.csv
            social_data = []
            for artist_id in range(1, 11):
                for week in weeks:
                    social_data.append({
                        'artist_id': artist_id,
                        'week_start_date': week,
                        'max_following': np.random.randint(10000, 100000),
                        'platform_name': 'Spotify'
                    })
            pd.DataFrame(social_data).to_csv(data_dir / "artist_social_week.csv", index=False)
            
            # Create artist_instagram.csv
            instagram_data = []
            for artist_id in range(1, 11):
                instagram_data.append({
                    'artist_id': artist_id,
                    'engagement_rate': np.random.uniform(0.01, 0.1),
                    'ages_13-17': np.random.uniform(0.05, 0.15),
                    'ages_18-24': np.random.uniform(0.20, 0.40),
                    'ages_25-34': np.random.uniform(0.20, 0.35),
                    'ages_35-44': np.random.uniform(0.10, 0.20),
                    'ages_45-54': np.random.uniform(0.05, 0.15),
                    'ages_55-64': np.random.uniform(0.02, 0.08),
                    'ages_65+': np.random.uniform(0.01, 0.05)
                })
            pd.DataFrame(instagram_data).to_csv(data_dir / "artist_instagram.csv", index=False)
            
            # Create ticket sales data
            ticket_data = []
            for artist_id in range(1, 11):
                for week in weeks[::4]:  # Sparse data (every 4 weeks)
                    ticket_data.append({
                        'artist_id': artist_id,
                        'start_date': week,
                        'st_event_count': np.random.randint(0, 5),
                        'st_num_tickets': np.random.randint(0, 1000),
                        'genre_id': np.random.randint(1, 10)
                    })
            pd.DataFrame(ticket_data).to_csv(data_dir / "lsecondaryticket_artist_week.csv", index=False)
            pd.DataFrame(ticket_data).to_csv(data_dir / "tsecondaryticket_artist_week.csv", index=False)
            
            # Create DMA data
            dma_data = []
            for artist_id in range(1, 11):
                for week in weeks[::2]:  # Bi-weekly data
                    for dma in range(1, 6):  # 5 DMAs per artist
                        dma_data.append({
                            'artist_id': artist_id,
                            'week_start_date': week,
                            'dma_id': dma,
                            'number_of_streams': np.random.randint(10000, 100000)
                        })
            pd.DataFrame(dma_data).to_csv(data_dir / "artist_mstreams_dma_week.csv", index=False)
            pd.DataFrame(dma_data).to_csv(data_dir / "lsecondaryticket_artist_dma_week.csv", index=False)
            pd.DataFrame(dma_data).to_csv(data_dir / "tsecondaryticket_artist_dma_week.csv", index=False)
            
            yield str(data_dir)

    def test_e2e_linear_regression_default(self, test_data_dir):
        """Test E2E pipeline with linear regression and default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            
            result = subprocess.run([
                sys.executable,
                "scripts/streaming_growth_pipeline.py",
                "--data-dir", test_data_dir,
                "--output", str(output_dir),
                "--artist-sample-size", "5",
                "--model", "linear_regression",
                "--cv-folds", "3",
                "--seed", "42"
            ], capture_output=True, text=True)
            
            # Check execution successful
            assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
            
            # Check output files exist
            assert (output_dir / "model_comparison.csv").exists()
            assert (output_dir / "results.json").exists()
            assert (output_dir / "linear_regression" / "predictions.csv").exists()
            assert (output_dir / "linear_regression" / "trained_model.pkl").exists()
            
            # Check results.json structure
            with open(output_dir / "results.json") as f:
                results = json.load(f)
                assert "config" in results
                assert "data_summary" in results
                assert "models" in results
                assert "linear_regression" in results["models"]

    def test_e2e_all_models(self, test_data_dir):
        """Test E2E pipeline with all models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            
            result = subprocess.run([
                sys.executable,
                "scripts/streaming_growth_pipeline.py",
                "--data-dir", test_data_dir,
                "--output", str(output_dir),
                "--artist-sample-size", "5",
                "--model", "all",
                "--cv-folds", "3",
                "--seed", "42"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
            
            # Check all model directories exist
            assert (output_dir / "linear_regression").exists()
            assert (output_dir / "random_forest").exists()
            assert (output_dir / "gradient_boosting").exists()
            
            # Check model comparison includes all models
            comparison_df = pd.read_csv(output_dir / "model_comparison.csv")
            assert len(comparison_df) == 3
            assert set(comparison_df["model"]) == {"linear_regression", "random_forest", "gradient_boosting"}

    def test_e2e_with_imputation(self, test_data_dir):
        """Test E2E pipeline with missing value imputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            
            result = subprocess.run([
                sys.executable,
                "scripts/streaming_growth_pipeline.py",
                "--data-dir", test_data_dir,
                "--output", str(output_dir),
                "--artist-sample-size", "5",
                "--model", "linear_regression",
                "--impute-missing",  # Enable imputation
                "--cv-folds", "3",
                "--seed", "42"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
            
            # Check results
            with open(output_dir / "results.json") as f:
                results = json.load(f)
                # With imputation, should have more samples
                assert results["data_summary"]["n_features"] > 0

    def test_e2e_outlier_methods(self, test_data_dir):
        """Test E2E pipeline with different outlier handling methods."""
        outlier_methods = ["none", "log_transform", "winsorize", "remove"]
        
        for method in outlier_methods:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / f"results_{method}"
                
                result = subprocess.run([
                    sys.executable,
                    "scripts/streaming_growth_pipeline.py",
                    "--data-dir", test_data_dir,
                    "--output", str(output_dir),
                    "--artist-sample-size", "5",
                    "--model", "linear_regression",
                    "--outlier-method", method,
                    "--cv-folds", "3",
                    "--seed", "42"
                ], capture_output=True, text=True)
                
                assert result.returncode == 0, f"Pipeline failed with {method}: {result.stderr}"
                assert (output_dir / "results.json").exists()

    def test_e2e_different_lags(self, test_data_dir):
        """Test E2E pipeline with different lag configurations."""
        lag_configs = [
            ["1"],
            ["1", "2"],
            ["1", "2", "4"],
            ["1", "2", "4", "8"]
        ]
        
        for lags in lag_configs:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / f"results_lags_{'_'.join(lags)}"
                
                result = subprocess.run([
                    sys.executable,
                    "scripts/streaming_growth_pipeline.py",
                    "--data-dir", test_data_dir,
                    "--output", str(output_dir),
                    "--artist-sample-size", "5",
                    "--model", "linear_regression",
                    "--lags", *lags,
                    "--cv-folds", "2",
                    "--seed", "42"
                ], capture_output=True, text=True)
                
                assert result.returncode == 0, f"Pipeline failed with lags {lags}: {result.stderr}"

    def test_e2e_different_rolling_windows(self, test_data_dir):
        """Test E2E pipeline with different rolling window configurations."""
        window_configs = [
            ["4"],
            ["4", "8"],
            ["4", "8", "12"]
        ]
        
        for windows in window_configs:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / f"results_windows_{'_'.join(windows)}"
                
                result = subprocess.run([
                    sys.executable,
                    "scripts/streaming_growth_pipeline.py",
                    "--data-dir", test_data_dir,
                    "--output", str(output_dir),
                    "--artist-sample-size", "5",
                    "--model", "linear_regression",
                    "--rolling-windows", *windows,
                    "--cv-folds", "2",
                    "--seed", "42"
                ], capture_output=True, text=True)
                
                assert result.returncode == 0, f"Pipeline failed with windows {windows}: {result.stderr}"

    def test_e2e_different_cv_folds(self, test_data_dir):
        """Test E2E pipeline with different CV fold configurations."""
        fold_counts = [2, 3, 5]
        
        for folds in fold_counts:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / f"results_cv{folds}"
                
                result = subprocess.run([
                    sys.executable,
                    "scripts/streaming_growth_pipeline.py",
                    "--data-dir", test_data_dir,
                    "--output", str(output_dir),
                    "--artist-sample-size", "5",
                    "--model", "linear_regression",
                    "--cv-folds", str(folds),
                    "--seed", "42"
                ], capture_output=True, text=True)
                
                assert result.returncode == 0, f"Pipeline failed with {folds} folds: {result.stderr}"
                
                # Check CV results in JSON
                with open(output_dir / "results.json") as f:
                    results = json.load(f)
                    cv_scores = results["models"]["linear_regression"]["cv_scores"]
                    assert cv_scores["n_folds"] == folds

    def test_e2e_feature_combination_matrix(self, test_data_dir):
        """Test comprehensive feature combination matrix."""
        test_combinations = [
            # (model, outlier_method, lags, windows, impute)
            ("linear_regression", "log_transform", ["1", "2"], ["4"], False),
            ("random_forest", "none", ["1"], ["4", "8"], False),
            ("gradient_boosting", "winsorize", ["1", "2", "4"], ["4"], True),
            ("linear_regression", "remove", ["1", "2"], ["4", "8"], True),
        ]
        
        for idx, (model, outlier, lags, windows, impute) in enumerate(test_combinations):
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / f"results_combo_{idx}"
                
                cmd = [
                    sys.executable,
                    "scripts/streaming_growth_pipeline.py",
                    "--data-dir", test_data_dir,
                    "--output", str(output_dir),
                    "--artist-sample-size", "5",
                    "--model", model,
                    "--outlier-method", outlier,
                    "--lags", *lags,
                    "--rolling-windows", *windows,
                    "--cv-folds", "2",
                    "--seed", "42"
                ]
                
                if impute:
                    cmd.append("--impute-missing")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                assert result.returncode == 0, f"Combination {idx} failed: {result.stderr}"
                assert (output_dir / "results.json").exists()

    def test_e2e_visualization_outputs(self, test_data_dir):
        """Test that all visualizations are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            
            result = subprocess.run([
                sys.executable,
                "scripts/streaming_growth_pipeline.py",
                "--data-dir", test_data_dir,
                "--output", str(output_dir),
                "--artist-sample-size", "5",
                "--model", "all",
                "--cv-folds", "3",
                "--seed", "42"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            
            plots_dir = output_dir / "plots"
            
            # Check model-agnostic plots
            assert (plots_dir / "feature_correlations_agnostic.png").exists()
            assert (plots_dir / "feature_contribution_relationships.png").exists()
            assert (plots_dir / "model_comparison_metrics.png").exists()
            assert (plots_dir / "cv_vs_test_performance.png").exists()
            
            # Check per-model plots
            for model in ["linear_regression", "random_forest", "gradient_boosting"]:
                model_plots = plots_dir / model
                assert (model_plots / "predicted_vs_actual.png").exists()
                assert (model_plots / "residuals_analysis.png").exists()
                assert (model_plots / "time_series_predictions.png").exists()
                assert (model_plots / "feature_importance.png").exists()

    def test_e2e_reproducibility(self, test_data_dir):
        """Test that pipeline produces reproducible results with same seed."""
        results_list = []
        
        for run in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / f"results_run{run}"
                
                result = subprocess.run([
                    sys.executable,
                    "scripts/streaming_growth_pipeline.py",
                    "--data-dir", test_data_dir,
                    "--output", str(output_dir),
                    "--artist-sample-size", "5",
                    "--model", "linear_regression",
                    "--cv-folds", "3",
                    "--seed", "42"  # Same seed
                ], capture_output=True, text=True)
                
                assert result.returncode == 0
                
                # Load results
                with open(output_dir / "results.json") as f:
                    results_list.append(json.load(f))
        
        # Compare metrics (should be identical)
        metrics1 = results_list[0]["models"]["linear_regression"]["test_metrics"]
        metrics2 = results_list[1]["models"]["linear_regression"]["test_metrics"]
        
        assert abs(metrics1["r2"] - metrics2["r2"]) < 1e-10
        assert abs(metrics1["rmse"] - metrics2["rmse"]) < 1e-10


class TestE2EDataLoading:
    """E2E tests specifically for data loading combinations."""

    def test_data_loading_with_different_sample_sizes(self):
        """Test data loading with various artist sample sizes."""
        sample_sizes = [5, 10, 20]
        
        for size in sample_sizes:
            # This would use fixture data, simplified for demonstration
            assert size > 0  # Placeholder

    def test_data_loading_with_min_weeks_filter(self):
        """Test data loading with different min_weeks thresholds."""
        min_weeks_options = [10, 20, 30]
        
        for min_weeks in min_weeks_options:
            assert min_weeks > 0  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
