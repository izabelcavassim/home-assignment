import pandas as pd
from typing import Dict, List
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.html_report import generate_html_report

def create_visualizations(
        all_results: Dict,
        output_dir: Path,
        logger: logging.Logger,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None,
        feature_cols: List[str] = None,
        config: Dict = None,
        meta_train: pd.DataFrame = None
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
        config: Pipeline configuration settings (optional)
        meta_train: Training metadata with artist_id (optional, for coloring by artist)
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating visualizations in {plots_dir}")

    # Flatten multi-horizon results for visualization
    flattened_results = {}
    for model_name, results in all_results.items():
        if results.get('is_multihorizon', False):
            # Multi-horizon: create separate entries for each horizon
            for horizon, horizon_results in results['horizons'].items():
                flattened_results[f"{model_name}_{horizon}"] = horizon_results
        else:
            # Regular: use as-is
            flattened_results[model_name] = results

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
        # Basically here we are showing the correlation between the features and the target variable
        # It is agnostic of the model because it just shows the linear relationship strength, which is the same regardless of the model used
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in correlations.head(15)]
        bars = ax.barh(range(len(correlations.head(15))), correlations.head(15), color=colors, alpha=0.7)
        ax.set_yticks(range(len(correlations.head(15))))
        ax.set_yticklabels(correlations.head(15).index)
        ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12)
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
        fig.suptitle('Top 6 Features: Relationship with Stream Count (colored by artist)', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for idx, feature in enumerate(top_features):
            ax = axes[idx]

            # Sample points if too many for visualization
            if len(analysis_df) > 5000:
                sample_idx = analysis_df.sample(n=5000, random_state=42).index
                sample_df = analysis_df.loc[sample_idx]
            else:
                sample_idx = analysis_df.index
                sample_df = analysis_df

            # Create scatter plot with artist coloring if available
            if meta_train is not None and 'artist_id' in meta_train.columns:
                artist_ids = meta_train.loc[sample_idx, 'artist_id'].astype('category').cat.codes
                scatter = ax.scatter(sample_df[feature], sample_df['number_of_streams_target'],
                    c=artist_ids, cmap='tab20', alpha=0.4, s=15)
            else:
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
            ax.set_ylabel('Stream Growth Target (log-transformed)', fontsize=10)
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
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        models = list(all_results.keys())

        # Test metrics (top row)
        test_metrics_data = {
            'Test RMSE': [all_results[m]['test_metrics']['rmse'] for m in models],
            'Test MAE': [all_results[m]['test_metrics']['mae'] for m in models],
            'Test R²': [all_results[m]['test_metrics']['r2'] for m in models]
        }

        # CV metrics (bottom row)
        cv_metrics_data = {
            'CV RMSE': [all_results[m]['cv_scores']['rmse_mean'] for m in models],
            'CV MAE': [all_results[m]['cv_scores']['mae_mean'] for m in models],
            'CV R²': [all_results[m]['cv_scores']['r2_mean'] for m in models]
        }

        cv_std_data = {
            'CV RMSE': [all_results[m]['cv_scores']['rmse_std'] for m in models],
            'CV MAE': [all_results[m]['cv_scores']['mae_std'] for m in models],
            'CV R²': [all_results[m]['cv_scores']['r2_std'] for m in models]
        }

        # Plot test metrics (top row)
        for idx, (metric_name, values) in enumerate(test_metrics_data.items()):
            ax = axes[0, idx]
            bars = ax.bar(models, values, alpha=0.7, color='steelblue')
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name.split()[1])
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9)

        # Plot CV metrics with error bars (bottom row)
        for idx, (metric_name, values) in enumerate(cv_metrics_data.items()):
            ax = axes[1, idx]
            stds = cv_std_data[metric_name]
            bars = ax.bar(models, values, yerr=stds, alpha=0.7, color='coral',
                capsize=5, error_kw={'linewidth': 2})
            ax.set_title(f'{metric_name} (± std)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name.split()[1])
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, val, std in zip(bars, values, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{val:.4f}\n±{std:.4f}',
                    ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(plots_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Created: model_comparison_metrics.png (with CV)")
        
        # Create separate plots for linear vs non-linear models
        linear_models = [m for m in models if 'linear' in m.lower() or 'ridge' in m.lower() or 'lasso' in m.lower()]
        nonlinear_models = [m for m in models if m not in linear_models]
        
        # Helper function to create comparison plot for a subset of models
        def create_model_subset_plot(model_subset, title_prefix, filename, color):
            if len(model_subset) < 1:
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(max(12, 5 * len(model_subset)), 10))
            fig.suptitle(f'{title_prefix} Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # Test metrics (top row)
            test_metrics = {
                'Test RMSE': [all_results[m]['test_metrics']['rmse'] for m in model_subset],
                'Test MAE': [all_results[m]['test_metrics']['mae'] for m in model_subset],
                'Test R²': [all_results[m]['test_metrics']['r2'] for m in model_subset]
            }
            
            # CV metrics (bottom row)
            cv_metrics = {
                'CV RMSE': [all_results[m]['cv_scores']['rmse_mean'] for m in model_subset],
                'CV MAE': [all_results[m]['cv_scores']['mae_mean'] for m in model_subset],
                'CV R²': [all_results[m]['cv_scores']['r2_mean'] for m in model_subset]
            }
            
            cv_stds = {
                'CV RMSE': [all_results[m]['cv_scores']['rmse_std'] for m in model_subset],
                'CV MAE': [all_results[m]['cv_scores']['mae_std'] for m in model_subset],
                'CV R²': [all_results[m]['cv_scores']['r2_std'] for m in model_subset]
            }
            
            # Plot test metrics (top row)
            for idx, (metric_name, values) in enumerate(test_metrics.items()):
                ax = axes[0, idx]
                x_labels = [m.replace('_', '\n') for m in model_subset]
                bars = ax.bar(x_labels, values, alpha=0.8, color=color)
                ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric_name.split()[1])
                
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Plot CV metrics with error bars (bottom row)
            for idx, (metric_name, values) in enumerate(cv_metrics.items()):
                ax = axes[1, idx]
                stds = cv_stds[metric_name]
                x_labels = [m.replace('_', '\n') for m in model_subset]
                bars = ax.bar(x_labels, values, yerr=stds, alpha=0.8, color=color,
                    capsize=5, error_kw={'linewidth': 2})
                ax.set_title(f'{metric_name} (± std)', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric_name.split()[1])
                
                for bar, val, std in zip(bars, values, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + abs(std),
                        f'{val:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create linear models comparison
        if len(linear_models) >= 1:
            create_model_subset_plot(
                linear_models, 
                'Linear', 
                'model_comparison_linear.png',
                '#3498db'  # Blue
            )
            logger.info("  Created: model_comparison_linear.png")
        
        # Create non-linear (tree-based) models comparison
        if len(nonlinear_models) >= 1:
            create_model_subset_plot(
                nonlinear_models, 
                'Tree-Based (Non-Linear)', 
                'model_comparison_nonlinear.png',
                '#27ae60'  # Green
            )
            logger.info("  Created: model_comparison_nonlinear.png")
    elif len(all_results) == 1:
        # Single model: show CV fold variance
        model_name = list(all_results.keys())[0]
        cv_scores = all_results[model_name]['cv_scores']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{model_name.replace("_", " ").title()} - Cross-Validation Performance',
            fontsize=16, fontweight='bold')

        metrics = [
            ('R²', cv_scores['r2_mean'], cv_scores['r2_std']),
            ('RMSE', cv_scores['rmse_mean'], cv_scores['rmse_std']),
            ('MAE', cv_scores['mae_mean'], cv_scores['mae_std'])
        ]

        for idx, (metric_name, mean_val, std_val) in enumerate(metrics):
            ax = axes[idx]
            bar = ax.bar([model_name], [mean_val], yerr=[std_val],
                alpha=0.7, color='coral', capsize=10, error_kw={'linewidth': 2})
            ax.set_title(f'CV {metric_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels
            ax.text(0, mean_val + std_val, f'{mean_val:.4f}\n±{std_val:.4f}',
                ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(plots_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Created: model_comparison_metrics.png (single model CV)")

    # 1.5 CV Consistency Plot
    if len(all_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(all_results.keys())
        x_pos = np.arange(len(models))

        # Plot test R² vs CV R² mean
        test_r2 = [all_results[m]['test_metrics']['r2'] for m in models]
        cv_r2_mean = [all_results[m]['cv_scores']['r2_mean'] for m in models]
        cv_r2_std = [all_results[m]['cv_scores']['r2_std'] for m in models]

        ax.scatter(x_pos, test_r2, s=100, label='Test R²', color='steelblue', marker='o', zorder=3)
        ax.errorbar(x_pos, cv_r2_mean, yerr=cv_r2_std, fmt='o', label='CV R² (mean ± std)',
            color='coral', capsize=5, capthick=2, markersize=8, zorder=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
        ax.set_ylabel('R² Score (coefficient of determination)', fontsize=12)
        ax.set_title('Model Stability: Test R² vs Cross-Validation R²\n' +
                     '(Good models have test score within CV error bars)',
            fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add note
        note = ("Interpretation: If test R² falls within CV error bars, the model generalizes well.\n"
                "Large gap suggests overfitting or distributional shift in test period.")
        ax.text(0.5, -0.25, note, transform=ax.transAxes,
            ha='center', fontsize=9, style='italic', color='gray')

        plt.tight_layout()
        plt.savefig(plots_dir / 'cv_vs_test_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Created: cv_vs_test_performance.png")

    # 2. Per-Model Visualizations
    for model_name, results in all_results.items():
        model_plots_dir = plots_dir / model_name
        model_plots_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions data
        predictions_file = output_dir / model_name / "predictions.csv"
        if predictions_file.exists():
            pred_df = pd.read_csv(predictions_file)

            # 2a. Predicted vs Actual Scatter Plot (colored by artist)
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color by artist if artist_id is available
            if 'artist_id' in pred_df.columns:
                artist_codes = pred_df['artist_id'].astype('category').cat.codes
                scatter = ax.scatter(pred_df['y_true'], pred_df['y_pred'], 
                    c=artist_codes, cmap='tab20', alpha=0.5, s=20)
            else:
                ax.scatter(pred_df['y_true'], pred_df['y_pred'], alpha=0.5, s=20)

            # Add perfect prediction line
            min_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min())
            max_val = max(pred_df['y_true'].max(), pred_df['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            ax.set_xlabel('Actual Streams (log-transformed growth rate)', fontsize=12)
            ax.set_ylabel('Predicted Streams (log-transformed growth rate)', fontsize=12)
            ax.set_title(f'{model_name.replace("_", " ").title()}\nPredicted vs Actual (colored by artist)',
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

            # 2b. Residuals Plot (colored by artist)
            residuals = pred_df['y_true'] - pred_df['y_pred']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'{model_name.replace("_", " ").title()} - Residual Analysis (colored by artist)',
                fontsize=14, fontweight='bold')

            # Residuals vs Predicted (colored by artist)
            if 'artist_id' in pred_df.columns:
                artist_codes = pred_df['artist_id'].astype('category').cat.codes
                axes[0].scatter(pred_df['y_pred'], residuals, c=artist_codes, cmap='tab20', alpha=0.5, s=20)
            else:
                axes[0].scatter(pred_df['y_pred'], residuals, alpha=0.5, s=20)
            axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0].set_xlabel('Predicted Streams (log-transformed growth rate)', fontsize=11)
            axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
            axes[0].set_title('Residuals vs Predicted', fontsize=12)
            axes[0].grid(True, alpha=0.3)

            # Residuals Distribution
            axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
            axes[1].set_xlabel('Residuals (Actual - Predicted)', fontsize=11)
            axes[1].set_ylabel('Frequency (count)', fontsize=11)
            axes[1].set_title('Residuals Distribution', fontsize=12)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(model_plots_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2c. Time Series of Predictions (sample of artists)
            # Select artists with most test data points for better visualization
            artist_counts = pred_df['artist_id'].value_counts()
            sample_artists = artist_counts.head(8).index.tolist()  # Top 8 artists by data points

            fig, axes = plt.subplots(len(sample_artists), 1,
                figsize=(14, 3*len(sample_artists)))
            if len(sample_artists) == 1:
                axes = [axes]

            fig.suptitle(f'{model_name.replace("_", " ").title()} - Time Series Predictions per Artist\n' +
                         f'(Top {len(sample_artists)} Artists by Test Data Points)',
                fontsize=14, fontweight='bold')

            for idx, artist_id in enumerate(sample_artists):
                artist_data = pred_df[pred_df['artist_id'] == artist_id].sort_values('week')

                # Calculate per-artist R²
                from sklearn.metrics import r2_score
                artist_r2 = r2_score(artist_data['y_true'], artist_data['y_pred'])
                artist_mae = np.abs(artist_data['y_true'] - artist_data['y_pred']).mean()

                # Plot with different styles
                axes[idx].plot(range(len(artist_data)), artist_data['y_true'],
                    'o-', label='Actual', alpha=0.8, linewidth=2.5,
                    markersize=6, color='#2E86AB')
                axes[idx].plot(range(len(artist_data)), artist_data['y_pred'],
                    's--', label='Predicted', alpha=0.8, linewidth=2,
                    markersize=5, color='#A23B72')

                # Shade the error region
                axes[idx].fill_between(range(len(artist_data)),
                    artist_data['y_true'],
                    artist_data['y_pred'],
                    alpha=0.2, color='gray')

                axes[idx].set_ylabel('Stream Growth (log-transformed)', fontsize=10, fontweight='bold')
                axes[idx].set_title(f'Artist {artist_id} | R²: {artist_r2:.4f} | MAE: {artist_mae:.4f} | n={len(artist_data)} weeks',
                    fontsize=11, fontweight='bold')
                axes[idx].legend(loc='best', fontsize=9)
                axes[idx].grid(True, alpha=0.3, linestyle='--')

                # Add week labels if available
                if 'week' in artist_data.columns and len(artist_data) <= 20:
                    # Show week labels if not too many points
                    axes[idx].set_xticks(range(len(artist_data)))
                    axes[idx].set_xticklabels(artist_data['week'].values,
                        rotation=45, ha='right', fontsize=8)
                    axes[idx].set_xlabel('Week', fontsize=10)
                else:
                    axes[idx].set_xlabel('Week Index', fontsize=10)

                # Add horizontal line at y=0 if data crosses zero
                if artist_data['y_true'].min() < 0 or artist_data['y_pred'].min() < 0:
                    axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

            plt.tight_layout()
            plt.savefig(model_plots_dir / 'time_series_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2c2. Create an aggregated view across all artists
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f'{model_name.replace("_", " ").title()} - Aggregate Time Series View',
                fontsize=14, fontweight='bold')

            # Top panel: All predictions scatter
            axes[0].scatter(pred_df['y_true'], pred_df['y_pred'],
                alpha=0.4, s=30, c=pred_df['artist_id'].astype('category').cat.codes,
                cmap='tab20')

            # Perfect prediction line
            min_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min())
            max_val = max(pred_df['y_true'].max(), pred_df['y_pred'].max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
                label='Perfect Prediction', alpha=0.8)

            axes[0].set_xlabel('Actual Stream Growth (log-transformed)', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Predicted Stream Growth (log-transformed)', fontsize=11, fontweight='bold')
            axes[0].set_title('All Test Predictions (colored by artist)', fontsize=12)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

            # Bottom panel: Prediction errors over time
            if 'week' in pred_df.columns:
                # Group by week and calculate metrics
                weekly_metrics = pred_df.groupby('week', group_keys=False).apply(
                    lambda x: pd.Series({
                        'mae': np.abs(x['y_true'] - x['y_pred']).mean(),
                        'rmse': np.sqrt(((x['y_true'] - x['y_pred'])**2).mean()),
                        'count': len(x)
                    }), include_groups=False
                ).reset_index()

                axes[1].plot(range(len(weekly_metrics)), weekly_metrics['mae'],
                    'o-', label='MAE', linewidth=2, markersize=6, color='#2E86AB')
                axes[1].plot(range(len(weekly_metrics)), weekly_metrics['rmse'],
                    's-', label='RMSE', linewidth=2, markersize=6, color='#A23B72')

                axes[1].set_xlabel('Week (time period)', fontsize=11, fontweight='bold')
                axes[1].set_ylabel('Error Magnitude (log-transformed units)', fontsize=11, fontweight='bold')
                axes[1].set_title(f'Prediction Error by Week (n={len(pred_df)} predictions across {len(weekly_metrics)} weeks)',
                    fontsize=12)
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3)

                # Add week labels if not too many
                if len(weekly_metrics) <= 20:
                    axes[1].set_xticks(range(len(weekly_metrics)))
                    axes[1].set_xticklabels(weekly_metrics['week'].values,
                        rotation=45, ha='right', fontsize=9)

            plt.tight_layout()
            plt.savefig(model_plots_dir / 'time_series_aggregate.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2d. Model-Specific Feature Importance Plot
        if results['feature_importance']:
            importance_df = pd.DataFrame(results['feature_importance']).head(15)

            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(importance_df)), importance_df['importance'])
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Feature Importance (|coefficient| or Gini importance)', fontsize=12)
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

    # Create combined feature importance comparison across all models
    if len(flattened_results) > 0:
        logger.info("\nCreating combined feature importance comparison...")

        # Collect feature importance from all models
        all_feature_importance = {}
        for model_name, results in flattened_results.items():
            feature_importance = results.get('feature_importance')
            if feature_importance is not None:
                # Handle both DataFrame and list formats
                if isinstance(feature_importance, pd.DataFrame):
                    fi_df = feature_importance
                elif isinstance(feature_importance, list) and len(feature_importance) > 0:
                    fi_df = pd.DataFrame(feature_importance)
                else:
                    fi_df = None

                if fi_df is not None and not fi_df.empty:
                    all_feature_importance[model_name] = fi_df

        if len(all_feature_importance) > 0:
            # Get top N features across all models
            top_n = 15
            all_features = set()
            for fi_df in all_feature_importance.values():
                all_features.update(fi_df['feature'].head(top_n).tolist())

            # Create comparison dataframe
            comparison_data = []
            for feature in all_features:
                row = {'feature': feature}
                for model_name, fi_df in all_feature_importance.items():
                    # Find importance for this feature in this model
                    feature_row = fi_df[fi_df['feature'] == feature]
                    if not feature_row.empty:
                        row[model_name] = feature_row['importance'].values[0]
                    else:
                        row[model_name] = 0.0
                comparison_data.append(row)

            comparison_df = pd.DataFrame(comparison_data)

            # Sort by average importance across models
            model_cols = [col for col in comparison_df.columns if col != 'feature']
            comparison_df['avg_importance'] = comparison_df[model_cols].mean(axis=1)
            comparison_df = comparison_df.sort_values('avg_importance', ascending=False).head(top_n)

            # Create grouped bar plot
            fig, ax = plt.subplots(figsize=(14, 10))

            features = comparison_df['feature'].tolist()
            x = np.arange(len(features))
            width = 0.8 / len(model_cols) if len(model_cols) > 1 else 0.6

            colors = plt.cm.Set3(np.linspace(0, 1, len(model_cols)))

            for idx, model_name in enumerate(model_cols):
                values = comparison_df[model_name].tolist()
                offset = (idx - len(model_cols)/2) * width + width/2
                bars = ax.barh(x + offset, values, width, label=model_name.replace('_', ' ').title(),
                    alpha=0.8, color=colors[idx])

                # Add value labels for significant values
                for i, (bar, val) in enumerate(zip(bars, values)):
                    if val > 0.01:  # Only label if importance > 0.01
                        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                            ha='left', va='center', fontsize=7, alpha=0.7)

            ax.set_yticks(x)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Feature Importance (normalized)', fontsize=12, fontweight='bold')
            ax.set_title(f'Feature Importance Comparison Across Models\n' +
                         f'(Top {len(features)} Features by Average Importance)',
                fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')

            # Add interpretation note
            note = ("Features shown consistently across models indicate robust predictors.\n"
                    "Model-specific importance differences reveal algorithmic preferences.")
            ax.text(0.5, -0.12, note, transform=ax.transAxes,
                ha='center', fontsize=9, style='italic', color='gray')

            plt.tight_layout()
            plt.savefig(plots_dir / 'combined_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("  Created: combined_feature_importance.png")

            # Save comparison table
            comparison_df.drop('avg_importance', axis=1).to_csv(
                plots_dir / 'feature_importance_comparison.csv', index=False
            )
            logger.info("  Saved: feature_importance_comparison.csv")
            
            # Create side-by-side feature importance comparison (one subplot per model)
            n_models = len(all_feature_importance)
            if n_models > 1:
                # Determine grid layout
                n_cols = min(n_models, 3)  # Max 3 columns
                n_rows = (n_models + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
                fig.suptitle('Feature Importance Comparison: Side-by-Side View\n(Top 10 Features per Model)', 
                    fontsize=16, fontweight='bold', y=1.02)
                
                # Flatten axes for easy iteration
                if n_models == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                
                # Color palette for models
                model_colors = plt.cm.Set2(np.linspace(0, 1, n_models))
                
                for idx, (model_name, fi_df) in enumerate(all_feature_importance.items()):
                    ax = axes[idx]
                    
                    # Get top 10 features for this model
                    top_10 = fi_df.head(10)
                    
                    # Create horizontal bar plot
                    y_pos = np.arange(len(top_10))
                    bars = ax.barh(y_pos, top_10['importance'], color=model_colors[idx], alpha=0.8)
                    
                    # Labels
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_10['feature'], fontsize=9)
                    ax.invert_yaxis()  # Top feature at top
                    ax.set_xlabel('Importance', fontsize=10)
                    ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for bar, val in zip(bars, top_10['importance']):
                        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{val:.3f}', va='center', fontsize=8)
                
                # Hide empty subplots if any
                for idx in range(n_models, len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'feature_importance_side_by_side.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("  Created: feature_importance_side_by_side.png")
        else:
            logger.info("  No feature importance data available for comparison")

    # Generate HTML Report with all settings and results
    if config:
        logger.info("\nGenerating HTML summary report...")
        html_file = generate_html_report(
            all_results=all_results,
            flattened_results=flattened_results,
            output_dir=output_dir,
            plots_dir=plots_dir,
            config=config,
            feature_cols=feature_cols,
            X_train=X_train
        )
        logger.info(f"  Created: analysis_report.html")
        logger.info(f"  Open in browser: file://{html_file.absolute()}")

    logger.info(f"All visualizations saved to {plots_dir}")
    return plots_dir