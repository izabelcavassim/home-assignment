#!/usr/bin/env python3
"""
Streaming Growth Analysis Pipeline

End-to-end pipeline for analyzing drivers of streaming growth for artists.
Focus on correlation analysis with proper temporal alignment and leakage prevention.

Usage:
    poetry run python scripts/streaming_growth_pipeline.py \\
        --data-dir data/artist_performance \\
        --output results/streaming_growth \\
        --artist-sample-size 100 \\
        --model linear_regression
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import sys


# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, save_json, save_dataframe
from src.streaming_data import StreamingDataLoader
from src.streaming_features import StreamingGrowthFeatures
from src.model import ModelRegistry
from src.metrics import calculate_metrics, calculate_cv_metrics
from src.plotting import create_visualizations


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Streaming Growth Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  poetry run python scripts/streaming_growth_pipeline.py \\
    --data-dir data/artist_performance \\
    --output results/streaming_growth

  # With specific model and more artists
  poetry run python scripts/streaming_growth_pipeline.py \\
    --data-dir data/artist_performance \\
    --output results/streaming_growth \\
    --artist-sample-size 200 \\
    --model random_forest

  # With custom lags and windows
  poetry run python scripts/streaming_growth_pipeline.py \\
    --data-dir data/artist_performance \\
    --output results/streaming_growth \\
    --lags 1 2 4 8 \\
    --rolling-windows 4 8 12
        """
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing artist performance CSVs')
    parser.add_argument('--output', type=str, default='results/streaming_growth',
                        help='Output directory for results')
    
    # Sampling arguments
    parser.add_argument('--artist-sample-size', type=int, default=100,
                        help='Number of artists to sample for analysis')
    parser.add_argument('--min-weeks', type=int, default=20,
                        help='Minimum weeks of data required per artist. This is to ensure that the artist has enough data to train the model.')
    
    # Feature engineering arguments
    parser.add_argument('--target-col', type=str, default='number_of_streams',
                        help='Target column for growth prediction. This is the column that we are trying to predict.')
    parser.add_argument('--use-cumulative', action='store_true',
                        help='Use cumulative streams instead of growth rates (eliminates outliers)')
    parser.add_argument('--include-cumulative-features', action='store_true',
                        help='Include cumulative streams as features (for comparison only - causes data leakage!)')
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 2, 4],
                        help='Lag periods for features (weeks). This is the number of weeks that we are going to lag the features by.')
    parser.add_argument('--rolling-windows', type=int, nargs='+', default=[4, 8],
                        help='Rolling window sizes (weeks). This is the number of weeks that we are going to use to calculate the rolling window features.')
    
    # Model arguments
    parser.add_argument('--model', type=str, 
                        choices=['linear_regression', 'random_forest', 'gradient_boosting', 'all'],
                        default='linear_regression',
                        help='Model type to train (use "all" to run all models)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--impute-missing', action='store_true',
                        help='Apply smart imputation to handle missing values')
    parser.add_argument('--outlier-method', type=str, 
                        choices=['none', 'remove', 'winsorize', 'log_transform'],
                        default='log_transform',
                        help='Method to handle outliers in target variable (default: log_transform)')
    parser.add_argument('--outlier-threshold', type=float, default=3.0,
                        help='Z-score threshold for outlier detection (default: 3.0)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    return parser


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    """
    Split data by time to prevent leakage.
    
    Args:
        X: Features
        y: Target
        metadata: DataFrame with artist_id and week
        test_size: Fraction of recent data for testing
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, meta_train, meta_test)
    """
    # Sort by week
    sorted_idx = metadata['week'].argsort()
    X = X.iloc[sorted_idx]
    y = y.iloc[sorted_idx]
    metadata = metadata.iloc[sorted_idx]
    
    # Split at time cutoff
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    meta_train = metadata.iloc[:split_idx]
    meta_test = metadata.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, meta_train, meta_test


def handle_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    method: str = 'none',
    threshold: float = 3.0,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Handle outliers in target variable.
    
    Args:
        X: Features
        y: Target variable
        metadata: Metadata (artist_id, week)
        method: Outlier handling method
            - 'none': No outlier handling
            - 'remove': Remove outliers beyond threshold
            - 'winsorize': Cap outliers at percentile
            - 'log_transform': Log transform target
        threshold: Z-score threshold for outlier detection
        logger: Logger instance
    
    Returns:
        Tuple of (X, y, metadata) after outlier handling
    """
    if logger:
        logger.info(f"Outlier handling method: {method}")
    
    original_size = len(y)
    
    if method == 'none':
        if logger:
            logger.info("  No outlier handling applied")
        return X, y, metadata
    
    elif method == 'remove':
        # Calculate z-scores
        z_scores = np.abs((y - y.mean()) / y.std())
        mask = z_scores < threshold
        
        X = X[mask]
        y = y[mask]
        metadata = metadata[mask]
        
        removed = original_size - len(y)
        if logger:
            logger.info(f"  Removed {removed} outliers (z-score > {threshold})")
            logger.info(f"  Remaining samples: {len(y)}")
    
    elif method == 'winsorize':
        # Cap at percentiles
        lower_percentile = 1
        upper_percentile = 99
        
        lower_bound = np.percentile(y, lower_percentile)
        upper_bound = np.percentile(y, upper_percentile)
        
        original_y = y.copy()
        y = y.clip(lower=lower_bound, upper=upper_bound)
        
        capped = (original_y != y).sum()
        if logger:
            logger.info(f"  Winsorized {capped} values to [{lower_bound:.2f}, {upper_bound:.2f}]")
            logger.info(f"  ({lower_percentile}th to {upper_percentile}th percentile)")
    
    elif method == 'log_transform':
        # Log transform (handle negative values by shifting)
        min_val = y.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            y = np.log1p(y + shift)
            if logger:
                logger.info(f"  Applied log1p transform with shift={shift:.2f}")
        else:
            y = np.log1p(y)
            if logger:
                logger.info(f"  Applied log1p transform")
        
        if logger:
            logger.info(f"  Target range after transform: [{y.min():.3f}, {y.max():.3f}]")
    
    return X, y, metadata


def analyze_feature_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """Extract and rank feature importance."""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        # Izabel: Feature importance in tree-based models is based on how much each feature reduces impurity across
        # all trees (gini importance for random forest as default and Split gain contribution for gradient_boosting).
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        # Izabel: Absolute coefficient value - how much Y changes per 1-unit change in X (after scaling).
        # I am Uses using np.abs() because sign doesn't indicate importance, just direction (which might be useful).
        importance = np.abs(model.coef_)
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


def main():
    """Main pipeline execution."""
    args = setup_parser().parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging("ml_framework", level=log_level)
    
    logger.info("=" * 80)
    logger.info("Streaming Growth Analysis Pipeline")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Artist sample size: {args.artist_sample_size}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Random seed: {args.seed}")
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Phase 1: Load and prepare data
        logger.info("\n" + "=" * 80)
        logger.info("Phase 1: Data Loading")
        logger.info("=" * 80)
        
        data_loader = StreamingDataLoader(
            data_dir=args.data_dir,
            artist_sample_size=args.artist_sample_size,
            min_weeks_per_artist=args.min_weeks,
            random_state=args.seed,
            impute_missing=args.impute_missing
        )
        
        df = data_loader.create_joined_dataset()
        
        # Save sampled artist IDs
        save_json(data_loader.sampled_artists, output_dir / "sampled_artists.json", logger, "sampled artist IDs")
        
        # Phase 2: Feature engineering
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Feature Engineering")
        logger.info("=" * 80)
        
        feature_engineer = StreamingGrowthFeatures(
            target_col=args.target_col, # This is the column that we are trying to predict. It should be the growth metric (e.g., growth_rate or cumulative_streams).
            lags=args.lags, # This is the number of weeks that we are going to lag the features by. For example, if lags=[1, 2, 4], we will create features for 1-week lag, 2-week lag, and 4-week lag.
            rolling_windows=args.rolling_windows, # This is the number of weeks that we are going to use to calculate the rolling window features. For example, if rolling_windows=[4, 8], we will create features for 4-week rolling window and 8-week rolling window.
            use_cumulative=args.use_cumulative, # If True, we will use cumulative streams as the target variable instead of growth rates. This can help eliminate outliers and make the target more stable.
            include_cumulative_features=args.include_cumulative_features # If True, we will include cumulative streams as features in addition to growth rates. This can provide additional information but may cause data leakage, so use with caution.
        )
        
        df_features, feature_cols = feature_engineer.engineer_features(df)
        
        # Save feature names
        save_json(feature_cols, output_dir / "feature_names.json", logger, "feature names")
        
        # Prepare for modeling
        X, y, metadata = feature_engineer.prepare_for_modeling(
            df_features, feature_cols, drop_na=True
        )
        
        # Handle outliers if requested
        if args.outlier_method != 'none':
            logger.info("\n" + "=" * 80)
            logger.info("Outlier Handling")
            logger.info("=" * 80)
            X, y, metadata = handle_outliers(
                X, y, metadata,
                method=args.outlier_method,
                threshold=args.outlier_threshold,
                logger=logger
            )
        
        # Phase 3: Train/test split (temporal)
        logger.info("\n" + "=" * 80)
        logger.info("Phase 3: Temporal Train/Test Split")
        logger.info("=" * 80)
        
        X_train, X_test, y_train, y_test, meta_train, meta_test = temporal_train_test_split(
            X, y, metadata, test_size=0.2
        )
        
        logger.info(f"Train set: {X_train.shape[0]:,} samples")
        logger.info(f"  Date range: {meta_train['week'].min()} to {meta_train['week'].max()}")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        logger.info(f"  Date range: {meta_test['week'].min()} to {meta_test['week'].max()}")
        
        # Phase 4: Feature scaling
        logger.info("\n" + "=" * 80)
        logger.info("Phase 4: Feature Scaling")
        logger.info("=" * 80)
        
        scaler = StandardScaler()
         # Izabel: Linear regression coefficients are directly affected by feature scale.
        # A feature in millions (streams) would dominate a feature in decimals (growth rate). Scaling makes
        # coefficients comparable and interpretable. More of an issue for Linear Regression than tree-based models, but good practice for all.
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Applied StandardScaler to features")
        
        # Phase 4.5: Cross-Validation (TimeSeriesSplit)
        logger.info("\n" + "=" * 80)
        logger.info("Phase 4.5: Time Series Cross-Validation")
        logger.info("=" * 80)

        # Izabel: TimeSeriesSplit is crucial for temporal data to prevent leakage.
        # Each fold trains on past data and validates on future data, mimicking real-world forecasting scenarios.
        # This ensures that our evaluation metrics reflect true predictive performance without peeking into the future.
        tscv = TimeSeriesSplit(n_splits=args.cv_folds)
        logger.info(f"Using {args.cv_folds}-fold TimeSeriesSplit cross-validation")
        logger.info("Note: Each fold trains on past data and validates on future data (no temporal leakage)")
        
        # Determine which models to run
        model_config = {
            'linear_regression': {'model_type': 'linear_regression'},
            'random_forest': {'model_type': 'random_forest_regressor', 'n_estimators': 100},
            'gradient_boosting': {'model_type': 'gradient_boosting_regressor', 'n_estimators': 100}
        }
        
        if args.model == 'all':
            models_to_run = list(model_config.keys())
            logger.info(f"Running all models: {', '.join(models_to_run)}")
        else:
            models_to_run = [args.model]
        
        # Store all results for comparison
        all_results = {}
        
        # Phase 5 & 6: Train and evaluate each model
        for model_name in models_to_run:
            logger.info("\n" + "=" * 80)
            logger.info(f"Training Model: {model_name}")
            logger.info("=" * 80)
            
            # Get config for selected model
            config = model_config.get(model_name, {})
            if not config:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Cross-Validation Evaluation
            logger.info(f"\nRunning {args.cv_folds}-fold cross-validation...")
            cv_scores = {'r2': [], 'rmse': [], 'mae': []}
            
            # Izabel: Splitting the training data into folds for cross-validation.
            # Each fold trains on past data and validates on future data to prevent temporal leakage.
            fold_num = 1
            for train_idx, val_idx in tscv.split(X_train_scaled):
                # Split data for this fold
                X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train model on this fold
                cv_model = ModelRegistry.create(**config)
                cv_model.fit(X_cv_train, y_cv_train)
                
                # Evaluate on validation fold
                y_cv_pred = cv_model.predict(X_cv_val)
                fold_metrics = calculate_metrics(y_cv_val, y_cv_pred)
                
                cv_scores['r2'].append(fold_metrics['r2'])
                cv_scores['rmse'].append(fold_metrics['rmse'])
                cv_scores['mae'].append(fold_metrics['mae'])
                
                logger.info(f"  Fold {fold_num}: R²={fold_metrics['r2']:.4f}, "
                          f"RMSE={fold_metrics['rmse']:.4f}, MAE={fold_metrics['mae']:.4f}")
                fold_num += 1

            # Calculate CV statistics
            cv_r2_mean, cv_r2_std, cv_rmse_mean, cv_rmse_std, cv_mae_mean, cv_mae_std = calculate_cv_metrics(cv_scores)
            logger.info(f"\nCross-Validation Results ({args.cv_folds} folds):")
            logger.info(f"  R²:   {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
            logger.info(f"  RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
            logger.info(f"  MAE:  {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")
            
            # Train final model on full training set
            logger.info(f"\nTraining final {model_name} model on full training set...")
            model = ModelRegistry.create(**config)
            model.fit(X_train_scaled, y_train)
            logger.info(f"Trained {model_name} model")
            
            # Evaluate on train set
            y_train_pred = model.predict(X_train_scaled)
            train_metrics = calculate_metrics(y_train, y_train_pred)
            
            # Evaluate on test set
            y_test_pred = model.predict(X_test_scaled)
            test_metrics = calculate_metrics(y_test, y_test_pred)
            
            logger.info("Train Metrics:")
            logger.info(f"  RMSE: {train_metrics['rmse']:.4f}")
            logger.info(f"  MAE: {train_metrics['mae']:.4f}")
            logger.info(f"  R²: {train_metrics['r2']:.4f}")
            
            logger.info("Test Metrics:")
            logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"  MAE: {test_metrics['mae']:.4f}")
            logger.info(f"  R²: {test_metrics['r2']:.4f}")
            
            # Feature importance analysis
            logger.info("\nFeature Importance Analysis:")
            importance_df = analyze_feature_importance(model.model, feature_cols, top_n=20)
            
            if not importance_df.empty:
                logger.info("Top 10 Most Important Features:")
                for idx, row in importance_df.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.6f}")
            
            # Save model-specific results
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trained model for reproducibility
            model_path = model_output_dir / "trained_model.pkl"
            model.save(str(model_path))
            logger.info(f"Saved trained model to {model_path}")
            
            # Save feature importance
            if not importance_df.empty:
                save_dataframe(importance_df, model_output_dir / "feature_importance.csv", logger, "feature importance")
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'artist_id': meta_test['artist_id'].values,
                'week': meta_test['week'].values,
                'y_true': y_test.values,
                'y_pred': y_test_pred
            })
            save_dataframe(predictions_df, model_output_dir / "predictions.csv", logger, "predictions")
            
            # Store results for comparison
            all_results[model_name] = {
                'cv_scores': {
                    'r2_mean': cv_r2_mean,
                    'r2_std': cv_r2_std,
                    'rmse_mean': cv_rmse_mean,
                    'rmse_std': cv_rmse_std,
                    'mae_mean': cv_mae_mean,
                    'mae_std': cv_mae_std,
                    'n_folds': args.cv_folds
                },
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance_df.to_dict('records') if not importance_df.empty else []
            }
        
        # Phase 7: Save summary results
        logger.info("\n" + "=" * 80)
        logger.info("Phase 7: Saving Summary Results")
        logger.info("=" * 80)
        
        # Create comparison summary
        comparison_df = pd.DataFrame([
            {
                'model': model_name,
                'cv_r2_mean': results['cv_scores']['r2_mean'],
                'cv_r2_std': results['cv_scores']['r2_std'],
                'cv_rmse_mean': results['cv_scores']['rmse_mean'],
                'cv_rmse_std': results['cv_scores']['rmse_std'],
                'train_rmse': results['train_metrics']['rmse'],
                'train_mae': results['train_metrics']['mae'],
                'train_r2': results['train_metrics']['r2'],
                'test_rmse': results['test_metrics']['rmse'],
                'test_mae': results['test_metrics']['mae'],
                'test_r2': results['test_metrics']['r2']
            }
            for model_name, results in all_results.items()
        ]).sort_values('test_r2', ascending=False)
        
        logger.info("\nModel Comparison (sorted by test R²):")
        logger.info("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        save_dataframe(comparison_df, output_dir / "model_comparison.csv", logger, "model comparison")
        
        # Save complete results
        results = {
            'config': vars(args),
            'data_summary': {
                'n_artists': df['artist_id'].nunique(),
                'n_weeks': df['week'].nunique(),
                'date_range': [str(df['week'].min()), str(df['week'].max())],
                'n_features': len(feature_cols)
            },
            'models': all_results
        }
        
        save_json(results, output_dir / "results.json", logger, "complete results")
        
        # Phase 8: Create Visualizations
        logger.info("\n" + "=" * 80)
        logger.info("Phase 8: Creating Visualizations")
        logger.info("=" * 80)
        
        # Prepare configuration dictionary
        pipeline_config = {
            'data_dir': getattr(args, 'data_dir', 'data'),
            'artist_sample_size': getattr(args, 'artist_sample_size', 100),
            'seed': getattr(args, 'seed', 42),
            'model': getattr(args, 'model', 'linear_regression'),
            'target_col': getattr(args, 'target_col', 'number_of_streams'),
            'use_cumulative': getattr(args, 'use_cumulative', False),
            'include_cumulative_features': getattr(args, 'include_cumulative_features', False),
            'lags': getattr(args, 'lags', [1, 2, 4]),
            'rolling_windows': getattr(args, 'rolling_windows', [4, 8]),
            'outlier_method': getattr(args, 'outlier_method', 'log_transform'),
            'cv_folds': getattr(args, 'cv_folds', 5),
            'test_split': 0.2  # Fixed test split ratio
        }
        
        plots_dir = create_visualizations(
            all_results=all_results, 
            output_dir=output_dir, 
            logger=logger,
            X_train=X_train,
            y_train=y_train,
            feature_cols=feature_cols,
            config=pipeline_config
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Plots saved to: {plots_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
