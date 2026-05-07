#!/usr/bin/env python3
"""
Generate HTML Report for Streaming Growth Analysis (Hybrid Mode)

Creates HTML reports in two modes:
- light: Lightweight report with external image links (fast, small file size)
- full: Comprehensive report with embedded base64 images and detailed interpretations

Usage:
    # Lightweight report (default)
    poetry run python scripts/generate_report.py --results-dir results/my_run
    
    # Lightweight report (explicit)
    poetry run python scripts/generate_report.py --results-dir results/my_run --mode light
    
    # Full comprehensive report with embedded images
    poetry run python scripts/generate_report.py --results-dir results/my_run --mode full
"""

import argparse
import json
import base64
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def encode_image(image_path: Path) -> str:
    """Encode image as base64 string for embedding in HTML"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def interpret_metrics(model_name: str, metrics: dict) -> str:
    """Generate interpretation of model metrics"""
    r2 = metrics.get('r2', 0)
    rmse = metrics.get('rmse', 0)
    mae = metrics.get('mae', 0)
    
    # Interpret R² score
    if r2 >= 0.7:
        r2_interp = "excellent - explains >70% of variance in streaming growth"
    elif r2 >= 0.5:
        r2_interp = "good - explains >50% of variance"
    elif r2 >= 0.3:
        r2_interp = "moderate - captures some patterns but significant unexplained variance"
    else:
        r2_interp = "weak - struggles to explain streaming patterns"
    
    return f"""
            <p><strong>R² Score: {r2:.4f}</strong> - {r2_interp}</p>
            <p><strong>RMSE: {rmse:.4f}</strong> - Average prediction error magnitude</p>
            <p><strong>MAE: {mae:.4f}</strong> - Typical absolute prediction error</p>
            <p>Since we're predicting log-transformed streaming values, these errors are in log-space. 
            An RMSE of ~{rmse:.2f} translates to predictions typically within {100*(10**rmse - 1):.1f}% of actual values.</p>
    """


def generate_full_report(results_dir: Path, output_file: Path):
    """Generate comprehensive HTML report with embedded images"""
    
    # Load results
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file) as f:
        full_results = json.load(f)
    
    # Handle nested structure
    if 'models' in full_results:
        models = full_results['models']
        config = full_results.get('config', {})
        data_summary = full_results.get('data_summary', {})
    else:
        models = full_results
        config = full_results.get('config', {})
        data_summary = {}
    
    plots_dir = results_dir / "plots"
    
    # Extract data summary with fallbacks
    n_artists = data_summary.get('n_artists', config.get('artist_sample_size', 'N/A'))
    n_weeks = data_summary.get('n_weeks', 'N/A')
    n_features = data_summary.get('n_features', 'N/A')
    date_range = data_summary.get('date_range', ['N/A', 'N/A'])
    
    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streaming Growth Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .plot-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        .metric-card {{
            display: inline-block;
            background: #f8f9fa;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        .key-finding {{
            background: #e7f3ff;
            padding: 20px;
            border-left: 4px solid #667eea;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .interpretation {{
            background: #f0f9ff;
            padding: 15px;
            border-left: 4px solid #3b82f6;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .methodology {{
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Streaming Growth Analysis Report</h1>
        <p>Quantifying Factors that Drive Artist Streaming Success</p>
        <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="key-finding">
            <h3>Key Findings</h3>
            <ul>
                <li><strong>Social Engagement Matters:</strong> Follower count and engagement rate from previous weeks are among the strongest predictors of streaming growth.</li>
                <li><strong>Geographic Diversity:</strong> Artists with activity across multiple markets (DMAs) show stronger growth patterns.</li>
                <li><strong>Lagged Features Critical:</strong> Past week's metrics are more predictive than current week, emphasizing the importance of temporal momentum.</li>
                <li><strong>Log Transformation Essential:</strong> Streaming data is right-skewed; log transformation significantly improves model performance.</li>
            </ul>
        </div>
        
        <h3>Dataset Overview</h3>
        <div class="metric-card">
            <div class="label">Artists Analyzed</div>
            <div class="value">{n_artists}</div>
        </div>
        <div class="metric-card">
            <div class="label">Time Period</div>
            <div class="value">{n_weeks} weeks</div>
        </div>
        <div class="metric-card">
            <div class="label">Features Created</div>
            <div class="value">{n_features}</div>
        </div>
        <div class="metric-card">
            <div class="label">Date Range</div>
            <div class="value">{' to '.join(date_range)}</div>
        </div>
    </div>

    <div class="section">
        <h2>Methodology</h2>
        <div class="methodology">
            <h3>Approach</h3>
            <p><strong>Problem Statement:</strong> Predict week-over-week streaming growth using external factors (social engagement, touring activity, demographics) while preventing temporal leakage.</p>
            
            <p><strong>Key Methodological Decisions:</strong></p>
            <ul>
                <li><strong>Target Variable:</strong> Log-transformed week-over-week streaming growth rate</li>
                <li><strong>Feature Engineering:</strong> Lagged features (1, 2, 4 weeks) and rolling aggregates (4, 8 week windows)</li>
                <li><strong>Temporal Leakage Prevention:</strong> Excluded all features derived from the target variable itself</li>
                <li><strong>Train/Test Split:</strong> Temporal split (80/20) - trained on older data, tested on recent data</li>
                <li><strong>Outlier Handling:</strong> {config.get('outlier_method', 'log_transform')} applied to normalize distribution</li>
                <li><strong>Missing Values:</strong> {'Smart imputation applied' if config.get('impute_missing', False) else 'Complete case analysis only'}</li>
            </ul>
        </div>
        
        <h3>Feature Categories</h3>
        <ul>
            <li><strong>Social Engagement:</strong> Follower counts, engagement rate, likes, comments (Instagram/Spotify)</li>
            <li><strong>Geographic Diversity:</strong> Number of unique DMAs with streaming/touring activity</li>
            <li><strong>Touring Activity:</strong> Ticket sales, event counts, average prices</li>
            <li><strong>Demographics:</strong> Age distribution, gender, ethnicity, language of fanbase</li>
            <li><strong>Temporal Features:</strong> All features lagged (1-4 weeks) and rolled (4-8 week windows)</li>
        </ul>
    </div>
"""

    # Model comparison section
    if models:
        html += """
    <div class="section">
        <h2>Model Comparison</h2>
"""
        
        # Add comparison plot if exists
        comparison_plot = plots_dir / "model_comparison_metrics.png"
        if comparison_plot.exists():
            img_data = encode_image(comparison_plot)
            html += f"""
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Model Comparison">
            <div class="plot-caption">Figure 1: Performance comparison across all models</div>
        </div>
"""
        
        # Create comparison table
        html += """
        <h3>Performance Metrics</h3>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Train R²</th>
                    <th>Test R²</th>
                    <th>Train RMSE</th>
                    <th>Test RMSE</th>
                    <th>Train MAE</th>
                    <th>Test MAE</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for model_name, model_results in sorted(models.items(), 
                                                key=lambda x: x[1]['test_metrics'].get('r2', 0), 
                                                reverse=True):
            train = model_results['train_metrics']
            test = model_results['test_metrics']
            html += f"""
                <tr>
                    <td><strong>{model_name.replace('_', ' ').title()}</strong></td>
                    <td>{train.get('r2', 0):.4f}</td>
                    <td>{test.get('r2', 0):.4f}</td>
                    <td>{train.get('rmse', 0):.4f}</td>
                    <td>{test.get('rmse', 0):.4f}</td>
                    <td>{train.get('mae', 0):.4f}</td>
                    <td>{test.get('mae', 0):.4f}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <div class="interpretation">
            <h3>Interpretation</h3>
            <p><strong>Best Performing Model:</strong> """ + max(models.items(), 
                key=lambda x: x[1]['test_metrics'].get('r2', 0))[0].replace('_', ' ').title() + """</p>
            <p>The table above compares model performance on both training and test sets. 
            The <strong>Test R²</strong> is the most important metric as it indicates how well 
            the model generalizes to unseen data. A higher R² (closer to 1.0) indicates better 
            predictive power.</p>
        </div>
    </div>
"""

    # Individual model sections
    for model_name, model_results in models.items():
        model_plots_dir = plots_dir / model_name
        
        html += f"""
    <div class="section">
        <h2>{model_name.replace('_', ' ').title()} - Detailed Analysis</h2>
        
        <div class="interpretation">
            {interpret_metrics(model_name, model_results['test_metrics'])}
        </div>
"""
        
        # Predicted vs Actual
        pred_plot = model_plots_dir / "predicted_vs_actual.png"
        if pred_plot.exists():
            img_data = encode_image(pred_plot)
            html += f"""
        <h3>Predicted vs Actual Performance</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="{model_name} Predictions">
            <div class="plot-caption">Scatter plot showing predicted vs actual streaming values. 
            Points closer to the diagonal line indicate better predictions.</div>
        </div>
"""
        
        # Residuals
        resid_plot = model_plots_dir / "residuals_analysis.png"
        if resid_plot.exists():
            img_data = encode_image(resid_plot)
            html += f"""
        <h3>Residual Analysis</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="{model_name} Residuals">
            <div class="plot-caption">Left: Residuals should be randomly distributed around zero. 
            Right: Histogram shows distribution of prediction errors.</div>
        </div>
"""
        
        # Time series predictions
        ts_plot = model_plots_dir / "time_series_predictions.png"
        if ts_plot.exists():
            img_data = encode_image(ts_plot)
            html += f"""
        <h3>Time Series Predictions</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="{model_name} Time Series">
            <div class="plot-caption">Model predictions over time for selected artists. 
            Good predictions should closely follow the actual values.</div>
        </div>
"""
        
        # Feature importance
        feat_plot = model_plots_dir / "feature_importance.png"
        if feat_plot.exists():
            img_data = encode_image(feat_plot)
            html += f"""
        <h3>Feature Importance</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="{model_name} Features">
            <div class="plot-caption">Top features driving {model_name.replace('_', ' ')} predictions. 
            Longer bars indicate more influential features.</div>
        </div>
"""
        
        html += """
    </div>
"""

    # Combined feature importance
    combined_feat_plot = plots_dir / "combined_feature_importance.png"
    if combined_feat_plot.exists():
        img_data = encode_image(combined_feat_plot)
        html += f"""
    <div class="section">
        <h2>Combined Feature Importance</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Combined Feature Importance">
            <div class="plot-caption">Comparison of feature importance across all models. 
            Features consistently important across models are more robust indicators of streaming growth.</div>
        </div>
    </div>
"""

    # Footer
    html += f"""
    <div class="footer">
        <p>Generated by Streaming Growth ML Pipeline</p>
        <p>{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        <p>Report Mode: <strong>Full (Embedded Images)</strong></p>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✅ Full HTML report generated: {output_file}")
    print(f"   File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"   Open in browser: file://{output_file.absolute()}")


def generate_light_report(results_dir: Path, output_file: Path):
    """Generate lightweight HTML report with external image references"""
    
    # Import the existing light report generator
    from src.html_report import generate_html_report
    
    # Load results
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file) as f:
        full_results = json.load(f)
    
    # Handle nested structure
    if 'models' in full_results:
        all_results = full_results['models']
        config = full_results.get('config', {})
    else:
        all_results = full_results
        config = full_results.get('config', {})
    
    # Flatten multi-horizon results
    flattened_results = {}
    for model_name, results in all_results.items():
        if isinstance(results, dict) and results.get('is_multihorizon', False):
            for horizon, horizon_results in results.get('horizons', {}).items():
                flattened_results[f"{model_name}_{horizon}"] = horizon_results
        else:
            flattened_results[model_name] = results
    
    # Load feature names if available
    feature_file = results_dir / "feature_names.json"
    feature_cols = None
    if feature_file.exists():
        with open(feature_file) as f:
            feature_data = json.load(f)
            # Handle both dict and list formats
            if isinstance(feature_data, dict):
                feature_cols = feature_data.get('features', [])
            elif isinstance(feature_data, list):
                feature_cols = feature_data
    
    plots_dir = results_dir / 'plots'
    
    # Generate HTML report using the existing function
    html_file = generate_html_report(
        all_results=all_results,
        flattened_results=flattened_results,
        output_dir=results_dir,
        plots_dir=plots_dir,
        config=config,
        feature_cols=feature_cols,
        X_train=None
    )
    
    print(f"✅ Light HTML report generated: {html_file}")
    print(f"   File size: {html_file.stat().st_size / 1024:.1f} KB")
    print(f"   Open in browser: file://{html_file.absolute()}")
    
    return html_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML report from pipeline results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lightweight report (default, fast generation, small file)
  python scripts/generate_report.py --results-dir results/my_run
  
  # Full report with embedded images (comprehensive, large file)
  python scripts/generate_report.py --results-dir results/my_run --mode full
        """
    )
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing pipeline results')
    parser.add_argument('--output', 
                        help='Output HTML file path (default: results_dir/report.html)')
    parser.add_argument('--mode', choices=['light', 'full'], default='light',
                        help='Report mode: light (external images, fast) or full (embedded images, comprehensive)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        print(f"   Run the pipeline first to generate results.")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = results_dir / "report.html"
    
    # Generate report based on mode
    print(f"\n{'='*60}")
    print(f"Generating {args.mode.upper()} mode HTML report from {results_dir}...")
    print(f"{'='*60}\n")
    
    try:
        if args.mode == 'full':
            generate_full_report(results_dir, output_file)
        else:
            generate_light_report(results_dir, output_file)
        
        print(f"\n{'='*60}")
        print(f"Report generation complete!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
