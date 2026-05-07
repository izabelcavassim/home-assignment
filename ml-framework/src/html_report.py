"""
HTML Report Generator for Streaming Growth Analysis Pipeline

Generates a comprehensive, self-contained HTML report with:
- Pipeline configuration settings
- Model performance metrics
- Feature engineering details
- Output files inventory
- Reproducibility command
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd


def generate_html_report(
    all_results: Dict,
    flattened_results: Dict,
    output_dir: Path,
    plots_dir: Path,
    config: Dict,
    feature_cols: List[str] = None,
    X_train: pd.DataFrame = None
) -> Path:
    """
    Generate HTML analysis report with all pipeline settings and results.
    
    Args:
        all_results: Dictionary with results for each model
        flattened_results: Flattened results (multi-horizon expanded)
        output_dir: Output directory path
        plots_dir: Plots directory path
        config: Pipeline configuration dictionary
        feature_cols: List of feature names
        X_train: Training features DataFrame
        
    Returns:
        Path to generated HTML file
    """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streaming Growth Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 20px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .config-item {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        .config-label {{
            font-weight: bold;
            color: #495057;
            display: block;
            margin-bottom: 5px;
        }}
        .config-value {{
            color: #212529;
            font-family: 'Courier New', monospace;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .metrics-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        .metrics-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .metrics-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .metrics-table tr:hover {{
            background: #e9ecef;
        }}
        .best-value {{
            background: #d4edda;
            font-weight: bold;
            color: #155724;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .warning-icon {{
            color: #ffc107;
            font-weight: bold;
            margin-right: 10px;
        }}
        .info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .info-icon {{
            color: #17a2b8;
            font-weight: bold;
            margin-right: 10px;
        }}
        .feature-list {{
            columns: 3;
            column-gap: 20px;
            margin-top: 10px;
        }}
        .feature-list li {{
            break-inside: avoid;
            margin-bottom: 5px;
        }}
        .timestamp {{
            text-align: center;
            color: #6c757d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
        code {{
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e83e8c;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Streaming Growth Analysis Report</h1>
        <p><strong>Output Directory:</strong> {output_dir}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📋 Pipeline Configuration</h2>
        <div class="config-grid">
            <div class="config-item">
                <span class="config-label">Data Directory</span>
                <span class="config-value">{config.get('data_dir', 'N/A')}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Artist Sample Size</span>
                <span class="config-value">{config.get('artist_sample_size', 'N/A')}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Random Seed</span>
                <span class="config-value">{config.get('seed', 'N/A')}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Models Trained</span>
                <span class="config-value">{config.get('model', 'N/A')}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Target Column</span>
                <span class="config-value">{config.get('target_col', 'N/A')}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Use Cumulative</span>
                <span class="config-value">{'✓ Yes' if config.get('use_cumulative', False) else '✗ No'}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Include Cumulative Features</span>
                <span class="config-value">{'✓ Yes (DATA LEAKAGE!)' if config.get('include_cumulative_features', False) else '✗ No'}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Multi-Horizon</span>
                <span class="config-value">{'✓ Yes' if config.get('multi_horizon', False) else '✗ No'}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Prediction Horizons</span>
                <span class="config-value">{', '.join(map(str, config.get('prediction_horizons', []))) + ' weeks' if config.get('prediction_horizons') else 'N/A'}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Lag Features</span>
                <span class="config-value">{', '.join(map(str, config.get('lags', [])))}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Rolling Windows</span>
                <span class="config-value">{', '.join(map(str, config.get('rolling_windows', [])))}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Outlier Handling</span>
                <span class="config-value">{config.get('outlier_method', 'N/A')}</span>
            </div>
            <div class="config-item">
                <span class="config-label">CV Folds</span>
                <span class="config-value">{config.get('cv_folds', 'N/A')}</span>
            </div>
            <div class="config-item">
                <span class="config-label">Test Split Ratio</span>
                <span class="config-value">{config.get('test_split', 'N/A')}</span>
            </div>
        </div>
        
        {'<div class="warning"><span class="warning-icon">⚠️</span><strong>Data Leakage Warning:</strong> Cumulative features are included in the model! R² values will be artificially high. Use only for comparison analysis.</div>' if config.get('include_cumulative_features', False) else ''}
    </div>
    
    <div class="section">
        <h2>📊 Model Performance Summary</h2>
"""
    
    # Add model comparison table
    if len(all_results) > 0:
        html_content += """
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Test R²</th>
                    <th>Test RMSE</th>
                    <th>Test MAE</th>
                    <th>CV R² (mean ± std)</th>
                    <th>CV RMSE (mean ± std)</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Find best models
        best_r2_model = max(flattened_results.items(), key=lambda x: x[1]['test_metrics']['r2'])[0]
        best_rmse_model = min(flattened_results.items(), key=lambda x: x[1]['test_metrics']['rmse'])[0]
        
        for model_name, results in sorted(flattened_results.items(), 
                                         key=lambda x: x[1]['test_metrics']['r2'], 
                                         reverse=True):
            test_r2 = results['test_metrics']['r2']
            test_rmse = results['test_metrics']['rmse']
            test_mae = results['test_metrics']['mae']
            cv_r2_mean = results['cv_scores']['r2_mean']
            cv_r2_std = results['cv_scores']['r2_std']
            cv_rmse_mean = results['cv_scores']['rmse_mean']
            cv_rmse_std = results['cv_scores']['rmse_std']
            
            r2_class = 'best-value' if model_name == best_r2_model else ''
            rmse_class = 'best-value' if model_name == best_rmse_model else ''
            
            html_content += f"""
                <tr>
                    <td><strong>{model_name.replace('_', ' ').title()}</strong></td>
                    <td class="{r2_class}">{test_r2:.4f}</td>
                    <td class="{rmse_class}">{test_rmse:.4f}</td>
                    <td>{test_mae:.4f}</td>
                    <td>{cv_r2_mean:.4f} ± {cv_r2_std:.4f}</td>
                    <td>{cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <div class="info">
            <span class="info-icon">ℹ️</span>
            <strong>Interpretation:</strong> Higher R² and lower RMSE/MAE indicate better predictions. 
            Small CV std suggests model stability across time periods.
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>🔍 Feature Engineering Details</h2>
"""
    
    if feature_cols:
        html_content += f"""
        <p><strong>Total Features Created:</strong> {len(feature_cols)}</p>
        <p><strong>Training Samples:</strong> {len(X_train) if X_train is not None else 'N/A'}</p>
        <p><strong>Feature Types:</strong></p>
        <ul>
            <li><strong>Lag Features:</strong> Past values (t-{', t-'.join(map(str, config.get('lags', [])))})</li>
            <li><strong>Rolling Windows:</strong> Moving averages & std ({', '.join(map(str, config.get('rolling_windows', [])))} weeks)</li>
            <li><strong>Growth Features:</strong> Percentage changes</li>
            <li><strong>Momentum Features:</strong> Trend indicators</li>
        </ul>
        
        <h3>All Features Used ({len(feature_cols)}):</h3>
        <ul class="feature-list">
"""
        for feat in sorted(feature_cols):
            html_content += f"            <li><code>{feat}</code></li>\n"
        
        html_content += """
        </ul>
"""
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>📁 Output Files</h2>
        <h3>Visualizations</h3>
        <ul>
            <li><code>plots/model_comparison_metrics.png</code> - Performance comparison</li>
            <li><code>plots/combined_feature_importance.png</code> - Cross-model feature comparison</li>
            <li><code>plots/cv_vs_test_performance.png</code> - Validation analysis</li>
            <li><code>plots/feature_correlations_agnostic.png</code> - Feature correlations</li>
        </ul>
        
        <h3>Per-Model Outputs</h3>
        <p>For each model (<code>{', '.join([m.replace('_', ' ').title() for m in all_results.keys()])}</code>):</p>
        <ul>
            <li><code>{'{model}'}/predictions.csv</code> - Test set predictions</li>
            <li><code>{'{model}'}/feature_importance.csv</code> - Feature rankings</li>
            <li><code>{'{model}'}/trained_model.pkl</code> - Saved model</li>
            <li><code>plots/{'{model}'}/predicted_vs_actual.png</code></li>
            <li><code>plots/{'{model}'}/residuals_analysis.png</code></li>
            <li><code>plots/{'{model}'}/time_series_predictions.png</code></li>
            <li><code>plots/{'{model}'}/time_series_aggregate.png</code></li>
            <li><code>plots/{'{model}'}/feature_importance.png</code></li>
        </ul>
        
        <h3>Summary Files</h3>
        <ul>
            <li><code>model_comparison.csv</code> - Performance metrics table</li>
            <li><code>results.json</code> - Complete results in JSON format</li>
            <li><code>feature_names.json</code> - List of all features</li>
            <li><code>sampled_artists.json</code> - Artist IDs used in analysis</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🚀 Reproduce This Run</h2>
        <p>To reproduce this exact analysis, run:</p>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;"><code>poetry run python scripts/streaming_growth_pipeline.py \\
    --data-dir {config.get('data_dir', 'data')} \\
    --output {output_dir} \\
    --model {config.get('model', 'all')} \\
    --artist-sample-size {config.get('artist_sample_size', 100)} \\
    --seed {config.get('seed', 42)} \\
    --target-col {config.get('target_col', 'number_of_streams')} \\
    --lags {' '.join(map(str, config.get('lags', [1, 2, 4])))} \\
    --rolling-windows {' '.join(map(str, config.get('rolling_windows', [4, 8])))} \\
    --outlier-method {config.get('outlier_method', 'log_transform')} \\
    --cv-folds {config.get('cv_folds', 5)}{' \\' if config.get('use_cumulative') or config.get('multi_horizon') else ''}
    {'--use-cumulative \\' if config.get('use_cumulative') else ''}
    {'--include-cumulative-features \\' if config.get('include_cumulative_features') else ''}
    {'--multi-horizon \\' if config.get('multi_horizon') else ''}
    {'--prediction-horizons ' + ' '.join(map(str, config.get('prediction_horizons', []))) if config.get('multi_horizon') else ''}</code></pre>
    </div>
    
    <div class="timestamp">
        <p>Report generated by Streaming Growth Analysis Pipeline</p>
        <p>Framework Version: 1.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    html_file = plots_dir / 'analysis_report.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_file
