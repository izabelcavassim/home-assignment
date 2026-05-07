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
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
import os


# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging
from src.streaming_data import StreamingDataLoader
from src.streaming_features import StreamingGrowthFeatures
from src.model import ModelRegistry
from src.train import Trainer


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
                        help='Minimum weeks of data required per artist')
    
    # Feature engineering arguments
    parser.add_argument('--target-col', type=str, default='number_of_streams',
                        help='Target column for growth prediction')
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 2, 4],
                        help='Lag periods for features (weeks)')
    parser.add_argument('--rolling-windows', type=int, nargs='+', default=[4, 8],
                        help='Rolling window sizes (weeks)')
    
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


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'n_samples': len(y_true)
    }


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
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_)
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


def create_visualizations(
    all_results: Dict,
    output_dir: Path,
    logger: logging.Logger,
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None,
    feature_cols: List[str] = None
):
    """
    Create comprehensive visualizations for model results.
    
    Args:
        all_results: Dictionary with results for each model
        output_dir: Directory to save plots
        logger: Logger instance
        X_train: Training features (optional, for contribution analysis)
        y_train: Training target (optional, for contribution analysis)
        feature_cols: Feature names (optional)
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating visualizations in {plots_dir}")
    
    # 0. Feature Contribution Analysis (if data provided)
    if X_train is not None and y_train is not None and feature_cols is not None:
        logger.info("Creating feature contribution analysis...")
        
        # Combine features and target for correlation analysis
        analysis_df = pd.DataFrame(X_train, columns=feature_cols).copy()
        analysis_df['number_of_streams_target'] = y_train.values
        
        # Calculate correlations with target
        correlations = analysis_df.corr()['number_of_streams_target'].drop('number_of_streams_target')
        correlations = correlations.sort_values(key=abs, ascending=False)
        
        # 0a. Model-Agnostic Correlation Bar Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in correlations.head(15)]
        bars = ax.barh(range(len(correlations.head(15))), correlations.head(15), color=colors, alpha=0.7)
        ax.set_yticks(range(len(correlations.head(15))))
        ax.set_yticklabels(correlations.head(15).index)
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
        ax.set_title('Top 15 Features: Correlation with Target (Model-Agnostic)\n' +
                    'Linear relationship strength - same for all models\n' +
                    '(Green=Positive, Red=Negative)', 
                    fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, correlations.head(15))):
            label_x = val + (0.01 if val > 0 else -0.01)
            ha = 'left' if val > 0 else 'right'
            ax.text(label_x, i, f'{val:.3f}', va='center', ha=ha, fontsize=9)
        
        # Add note about model-specific importance
        note_text = ("Note: This shows linear correlation (Pearson r). Model-specific feature importance\n"
                    "varies by algorithm - see individual model plots for each model's actual weighting.")
        ax.text(0.5, -0.12, note_text, transform=ax.transAxes, 
               ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_correlations_agnostic.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Created: feature_correlations_agnostic.png (model-agnostic)")
        
        # 0b. Top Features Scatter Plots (showing actual relationships)
        top_features = correlations.abs().nlargest(6).index.tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Top 6 Features: Relationship with Stream Count', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            
            # Sample points if too many for visualization
            if len(analysis_df) > 5000:
                sample_df = analysis_df.sample(n=5000, random_state=42)
            else:
                sample_df = analysis_df
            
            # Create scatter plot
            scatter = ax.scatter(sample_df[feature], sample_df['number_of_streams_target'], 
                               alpha=0.3, s=10)
            
            # Add trend line
            z = np.polyfit(sample_df[feature].dropna(), 
                          sample_df.loc[sample_df[feature].notna(), 'number_of_streams_target'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample_df[feature].min(), sample_df[feature].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Trend')
            
            corr = correlations[feature]
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Stream Count (Target)', fontsize=10)
            ax.set_title(f'{feature.replace("_", " ").title()}\nCorr: {corr:.3f}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_contribution_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Created: feature_contribution_relationships.png")
        
        # 0c. Feature Contribution Summary Table
        contribution_summary = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'abs_correlation': correlations.abs().values,
            'contribution_type': ['Positive' if x > 0 else 'Negative' for x in correlations.values]
        }).sort_values('abs_correlation', ascending=False)
        
        contribution_file = plots_dir / 'feature_contributions.csv'
        contribution_summary.to_csv(contribution_file, index=False)
        logger.info(f"  Saved feature contributions to {contribution_file}")
    
    # 1. Model Comparison - Metrics Bar Chart
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(all_results.keys())
        metrics_data = {
            'RMSE': [all_results[m]['test_metrics']['rmse'] for m in models],
            'MAE': [all_results[m]['test_metrics']['mae'] for m in models],
            'R²': [all_results[m]['test_metrics']['r2'] for m in models]
        }
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx]
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(f'Test {metric_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Created: model_comparison_metrics.png")
    
    # 2. Per-Model Visualizations
    for model_name, results in all_results.items():
        model_plots_dir = plots_dir / model_name
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions data
        predictions_file = output_dir / model_name / "predictions.csv"
        if predictions_file.exists():
            pred_df = pd.read_csv(predictions_file)
            
            # 2a. Predicted vs Actual Scatter Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(pred_df['y_true'], pred_df['y_pred'], alpha=0.5, s=20)
            
            # Add perfect prediction line
            min_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min())
            max_val = max(pred_df['y_true'].max(), pred_df['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Streams', fontsize=12)
            ax.set_ylabel('Predicted Streams', fontsize=12)
            ax.set_title(f'{model_name.replace("_", " ").title()}\nPredicted vs Actual', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add R² score to plot
            r2 = results['test_metrics']['r2']
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(model_plots_dir / 'predicted_vs_actual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2b. Residuals Plot
            residuals = pred_df['y_true'] - pred_df['y_pred']
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'{model_name.replace("_", " ").title()} - Residual Analysis', 
                        fontsize=14, fontweight='bold')
            
            # Residuals vs Predicted
            axes[0].scatter(pred_df['y_pred'], residuals, alpha=0.5, s=20)
            axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0].set_xlabel('Predicted Streams', fontsize=11)
            axes[0].set_ylabel('Residuals', fontsize=11)
            axes[0].set_title('Residuals vs Predicted', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Residuals Distribution
            axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
            axes[1].set_xlabel('Residuals', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].set_title('Residuals Distribution', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(model_plots_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2c. Time Series of Predictions (sample of artists)
            sample_artists = pred_df['artist_id'].unique()[:5]  # Top 5 artists
            
            fig, axes = plt.subplots(len(sample_artists), 1, 
                                    figsize=(12, 3*len(sample_artists)))
            if len(sample_artists) == 1:
                axes = [axes]
            
            fig.suptitle(f'{model_name.replace("_", " ").title()} - Sample Artist Predictions', 
                        fontsize=14, fontweight='bold')
            
            for idx, artist_id in enumerate(sample_artists):
                artist_data = pred_df[pred_df['artist_id'] == artist_id].sort_values('week')
                
                axes[idx].plot(range(len(artist_data)), artist_data['y_true'], 
                             'o-', label='Actual', alpha=0.7, linewidth=2)
                axes[idx].plot(range(len(artist_data)), artist_data['y_pred'], 
                             's-', label='Predicted', alpha=0.7, linewidth=2)
                axes[idx].set_ylabel('Streams', fontsize=10)
                axes[idx].set_title(f'Artist ID: {artist_id}', fontsize=11)
                axes[idx].legend(loc='best')
                axes[idx].grid(True, alpha=0.3)
                
                if idx == len(sample_artists) - 1:
                    axes[idx].set_xlabel('Week Index', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(model_plots_dir / 'time_series_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2d. Model-Specific Feature Importance Plot
        if results['feature_importance']:
            importance_df = pd.DataFrame(results['feature_importance']).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(importance_df)), importance_df['importance'])
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Feature Importance (Model-Specific)', fontsize=12)
            ax.set_title(f'{model_name.replace("_", " ").title()}\n' +
                        f'Top 15 Feature Importance - {model_name.upper()} Specific Weighting', 
                        fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
                ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
            
            # Add explanation based on model type
            if 'linear' in model_name:
                note = "Linear Regression: Importance = |coefficient| (linear effect size)"
            elif 'forest' in model_name:
                note = "Random Forest: Importance = Gini importance (split contribution)"
            elif 'boosting' in model_name:
                note = "Gradient Boosting: Importance = split gain contribution"
            else:
                note = ""
            
            if note:
                ax.text(0.5, -0.10, note, transform=ax.transAxes, 
                       ha='center', fontsize=9, style='italic', color='gray')
            
            plt.tight_layout()
            plt.savefig(model_plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"  Created visualizations for {model_name}")
    
    logger.info(f"All visualizations saved to {plots_dir}")
    return plots_dir


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
        sampled_artists_file = output_dir / "sampled_artists.json"
        with open(sampled_artists_file, 'w') as f:
            json.dump(data_loader.sampled_artists, f, indent=2)
        logger.info(f"Saved sampled artist IDs to {sampled_artists_file}")
        
        # Phase 2: Feature engineering
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Feature Engineering")
        logger.info("=" * 80)
        
        feature_engineer = StreamingGrowthFeatures(
            target_col=args.target_col,
            lags=args.lags,
            rolling_windows=args.rolling_windows
        )
        
        df_features, feature_cols = feature_engineer.engineer_features(df)
        
        # Save feature names
        features_file = output_dir / "feature_names.json"
        with open(features_file, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        logger.info(f"Saved feature names to {features_file}")
        
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
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Applied StandardScaler to features")
        
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
            
            # Train model
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
            importance_df = analyze_feature_importance(model, feature_cols, top_n=20)
            
            if not importance_df.empty:
                logger.info("Top 10 Most Important Features:")
                for idx, row in importance_df.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.6f}")
            
            # Save model-specific results
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save feature importance
            if not importance_df.empty:
                importance_file = model_output_dir / "feature_importance.csv"
                importance_df.to_csv(importance_file, index=False)
                logger.info(f"Saved feature importance to {importance_file}")
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'artist_id': meta_test['artist_id'].values,
                'week': meta_test['week'].values,
                'y_true': y_test.values,
                'y_pred': y_test_pred
            })
            predictions_file = model_output_dir / "predictions.csv"
            predictions_df.to_csv(predictions_file, index=False)
            logger.info(f"Saved predictions to {predictions_file}")
            
            # Store results for comparison
            all_results[model_name] = {
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
        comparison_file = output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"\nSaved model comparison to {comparison_file}")
        
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
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved complete results to {results_file}")
        
        # Phase 8: Create Visualizations
        logger.info("\n" + "=" * 80)
        logger.info("Phase 8: Creating Visualizations")
        logger.info("=" * 80)
        
        plots_dir = create_visualizations(
            all_results=all_results, 
            output_dir=output_dir, 
            logger=logger,
            X_train=X_train,
            y_train=y_train,
            feature_cols=feature_cols
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
