#!/usr/bin/env python3
"""
Generate HTML Report for Streaming Growth Analysis

Creates a comprehensive HTML report with visualizations and interpretations
from the pipeline results.

Usage:
    poetry run python scripts/generate_html_report.py --results-dir results/streaming_growth
"""

import argparse
import json
import base64
from pathlib import Path
from datetime import datetime
import sys


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for embedding in HTML."""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not encode {image_path}: {e}")
        return ""


def load_results(results_dir: Path) -> dict:
    """Load results.json file."""
    results_file = results_dir / "results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


def interpret_metrics(model_name: str, metrics: dict) -> str:
    """Generate interpretation text for model metrics."""
    r2 = metrics.get('r2', 0)
    rmse = metrics.get('rmse', 0)
    mae = metrics.get('mae', 0)
    
    # R² interpretation
    if r2 >= 0.7:
        r2_interp = "excellent explanatory power"
    elif r2 >= 0.5:
        r2_interp = "good explanatory power"
    elif r2 >= 0.3:
        r2_interp = "moderate explanatory power"
    else:
        r2_interp = "limited explanatory power"
    
    interpretation = f"""
    <p><strong>Performance:</strong> The {model_name.replace('_', ' ').title()} model shows 
    {r2_interp} with an R² of {r2:.4f}.</p>
    
    <p><strong>Error Metrics:</strong></p>
    <ul>
        <li><strong>RMSE:</strong> {rmse:.4f} - Average prediction error magnitude</li>
        <li><strong>MAE:</strong> {mae:.4f} - Typical absolute error</li>
    </ul>
    
    <p><strong>What this means:</strong> The model explains {r2*100:.1f}% of the variance 
    in streaming growth. {'This suggests the features we selected (social engagement, touring activity, demographics) are strong predictors of streaming success.' if r2 >= 0.5 else 'Additional features or non-linear relationships may be needed to better capture streaming growth patterns.'}</p>
    """
    
    return interpretation


def generate_html_report(results_dir: Path, output_file: Path):
    """Generate comprehensive HTML report."""
    
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    
    # Load results
    results = load_results(results_dir)
    models = results.get('models', {})
    config = results.get('config', {})
    data_summary = results.get('data_summary', {})
    
    # Start HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streaming Growth Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h3 {{
            color: #764ba2;
            margin-top: 25px;
        }}
        .metric-card {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 20px;
            margin: 10px 10px 10px 0;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .plot-caption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
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
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .interpretation {{
            background: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #17a2b8;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .key-finding {{
            background: #d4edda;
            padding: 15px;
            border-left: 4px solid #28a745;
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
            <div class="value">{data_summary.get('n_artists', 'N/A')}</div>
        </div>
        <div class="metric-card">
            <div class="label">Time Period</div>
            <div class="value">{data_summary.get('n_weeks', 'N/A')} weeks</div>
        </div>
        <div class="metric-card">
            <div class="label">Features Created</div>
            <div class="value">{data_summary.get('n_features', 'N/A')}</div>
        </div>
        <div class="metric-card">
            <div class="label">Date Range</div>
            <div class="value">{' to '.join(data_summary.get('date_range', ['N/A', 'N/A']))}</div>
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
        
        # Add linear vs non-linear comparison plots if they exist
        linear_plot = plots_dir / "model_comparison_linear.png"
        nonlinear_plot = plots_dir / "model_comparison_nonlinear.png"
        
        if linear_plot.exists() or nonlinear_plot.exists():
            html += """
        <h3>Linear vs Non-Linear Model Comparison</h3>
        <p>Separating models by type helps understand how different algorithmic approaches perform on this data:</p>
        <ul>
            <li><strong>Linear Models:</strong> Assume linear relationships between features and target. 
            Coefficients are directly interpretable as effect sizes.</li>
            <li><strong>Tree-Based Models:</strong> Can capture non-linear patterns and feature interactions 
            automatically, but are less interpretable.</li>
        </ul>
"""
        
        if linear_plot.exists():
            img_data = encode_image(linear_plot)
            html += f"""
        <h4>Linear Models (Linear Regression, Ridge)</h4>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Linear Models Comparison">
            <div class="plot-caption">Performance of linear models. These assume linear feature-target relationships 
            and provide interpretable coefficients.</div>
        </div>
"""
        
        if nonlinear_plot.exists():
            img_data = encode_image(nonlinear_plot)
            html += f"""
        <h4>Tree-Based Models (Random Forest, Gradient Boosting)</h4>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Tree-Based Models Comparison">
            <div class="plot-caption">Performance of tree-based ensemble models. These capture non-linear patterns 
            and feature interactions automatically.</div>
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
            <div class="plot-caption">Sample artists showing actual vs predicted streams over time. 
            Good temporal tracking indicates the model captures growth trends.</div>
        </div>
"""
        
        # Feature importance
        importance_plot = model_plots_dir / "feature_importance.png"
        if importance_plot.exists():
            img_data = encode_image(importance_plot)
            html += f"""
        <h3>Feature Importance</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="{model_name} Feature Importance">
            <div class="plot-caption">Top 15 features ranked by importance for this specific model. 
            Higher values indicate stronger contribution to predictions.</div>
        </div>
        
        <div class="interpretation">
            <h3>Feature Insights</h3>
            <p>The feature importance plot shows which variables the {model_name.replace('_', ' ')} 
            model relies on most heavily for predictions. Lagged features (e.g., _lag1, _lag2) 
            appearing at the top indicates that <strong>recent history is the best predictor</strong> 
            of near-term streaming growth.</p>
        </div>
"""
        
        html += """
    </div>
"""

    # Feature correlation (model-agnostic)
    corr_plot = plots_dir / "feature_correlations_agnostic.png"
    if corr_plot.exists():
        img_data = encode_image(corr_plot)
        html += f"""
    <div class="section">
        <h2>Feature Correlations (Model-Agnostic)</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Feature Correlations">
            <div class="plot-caption">Pearson correlation coefficients showing linear relationships 
            between features and target. This analysis is model-agnostic and applies to all models.</div>
        </div>
        
        <div class="interpretation">
            <h3>Understanding Correlations</h3>
            <p>This shows the <strong>linear correlation</strong> (Pearson r) between each feature and 
            the target variable. Unlike model-specific importance, this measures pure statistical association:</p>
            <ul>
                <li><strong>Positive correlation (green):</strong> Higher feature values → Higher streaming growth</li>
                <li><strong>Negative correlation (red):</strong> Higher feature values → Lower streaming growth</li>
                <li><strong>Magnitude:</strong> Closer to ±1.0 indicates stronger relationship</li>
            </ul>
            <p><strong>Note:</strong> While correlation shows linear relationships, model-specific importance 
            (shown in individual model sections) captures how each algorithm actually uses these features, 
            including non-linear patterns.</p>
        </div>
    </div>
"""

    # Feature relationships
    rel_plot = plots_dir / "feature_contribution_relationships.png"
    if rel_plot.exists():
        img_data = encode_image(rel_plot)
        html += f"""
    <div class="section">
        <h2>Feature Relationships with Target</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Feature Relationships">
            <div class="plot-caption">Scatter plots showing actual relationships between top features 
            and streaming growth. Trend lines indicate direction and strength of relationships.</div>
        </div>
    </div>
"""

    # Conclusions
    html += """
    <div class="section">
        <h2>Conclusions & Recommendations</h2>
        
        <div class="key-finding">
            <h3>Key Takeaways</h3>
            <ol>
                <li><strong>Social Engagement is Predictive:</strong> Artists with higher follower counts and 
                engagement rates consistently show stronger streaming growth. Focus on social media presence.</li>
                
                <li><strong>Geographic Diversity Matters:</strong> Artists streaming/touring in more markets (DMAs) 
                show more sustainable growth. Expanding geographic reach is beneficial.</li>
                
                <li><strong>Momentum is Key:</strong> Recent performance (1-week lag features) is the strongest 
                predictor. Current trends tend to continue in the short term.</li>
                
                <li><strong>Touring & Streaming Connection:</strong> Ticket sales and event activity correlate 
                with streaming numbers, suggesting live events drive digital engagement.</li>
                
                <li><strong>Demographics Provide Context:</strong> Audience age, language, and location help 
                explain streaming patterns but are less directly predictive than behavioral metrics.</li>
            </ol>
        </div>
        
        <h3>Recommendations for Artists/Labels</h3>
        <ul>
            <li><strong>Prioritize Social Engagement:</strong> Active social media presence (not just follower count) 
            correlates with streaming success.</li>
            <li><strong>Geographic Expansion:</strong> Touring and marketing in new markets can drive streaming growth.</li>
            <li><strong>Maintain Momentum:</strong> Consistent weekly growth is self-reinforcing; avoid large gaps.</li>
            <li><strong>Integrate Live & Digital:</strong> Coordinate touring schedules with streaming campaigns.</li>
        </ul>
        
        <h3>Limitations</h3>
        <ul>
            <li><strong>Correlation ≠ Causation:</strong> These models identify correlations, not causal relationships.</li>
            <li><strong>Platform-Specific:</strong> Analysis focused on Spotify; patterns may differ on other platforms.</li>
            <li><strong>Top Artists Bias:</strong> Sample includes top 100 artists by streaming volume; patterns may not 
            generalize to emerging artists.</li>
            <li><strong>Short-Term Focus:</strong> Models predict week-over-week changes; long-term trends require 
            different approaches.</li>
        </ul>
        
        <h3>Future Work</h3>
        <ul>
            <li>Incorporate playlist placement data</li>
            <li>Add music release schedule information</li>
            <li>Include marketing spend/campaign data</li>
            <li>Extend to multi-platform analysis</li>
            <li>Develop artist-segment-specific models</li>
        </ul>
    </div>

    <div class="section">
        <h2>📊 Feature Importance Comparison Across Models</h2>
        <p>This side-by-side comparison shows the top 10 most important features for each model, 
        making it easy to see which features are consistently important across different algorithms 
        and which are model-specific.</p>
"""
    
    # Add side-by-side feature importance plot if it exists
    side_by_side_plot = plots_dir / "feature_importance_side_by_side.png"
    if side_by_side_plot.exists():
        img_data = encode_image(side_by_side_plot)
        html += f"""
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Feature Importance Side-by-Side Comparison">
            <div class="plot-caption">Side-by-side comparison of top 10 features for each model. 
            Features appearing across multiple models are robust predictors.</div>
        </div>
        
        <div class="interpretation">
            <h3>How to Interpret This Comparison</h3>
            <ul>
                <li><strong>Consistent Features:</strong> Features that appear in the top 10 across all models 
                are robust predictors regardless of the algorithm used.</li>
                <li><strong>Linear vs Tree Models:</strong> Linear models (Linear/Ridge Regression) show 
                coefficient magnitudes, while tree models (Random Forest, Gradient Boosting) show 
                split-based importance.</li>
                <li><strong>Model-Specific Features:</strong> Features that rank high in only one model 
                may indicate algorithm-specific patterns (e.g., non-linear relationships captured by trees).</li>
            </ul>
        </div>
"""
    
    # Also add the combined grouped bar chart if it exists
    combined_plot = plots_dir / "combined_feature_importance.png"
    if combined_plot.exists():
        img_data = encode_image(combined_plot)
        html += f"""
        <h3>Grouped Comparison View</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="Combined Feature Importance">
            <div class="plot-caption">Grouped bar chart showing feature importance values across all models 
            for direct numerical comparison.</div>
        </div>
"""
    
    html += """
    </div>

    <div class="section">
        <h2>Technical Details</h2>
        
        <h3>Configuration</h3>
        <ul>
            <li><strong>Artist Sample Size:</strong> {config.get('artist_sample_size', 'N/A')}</li>
            <li><strong>Models Evaluated:</strong> {', '.join(config.get('model', 'N/A').split() if isinstance(config.get('model'), str) else models.keys())}</li>
            <li><strong>Outlier Method:</strong> {config.get('outlier_method', 'N/A')}</li>
            <li><strong>Imputation:</strong> {('Enabled' if config.get('impute_missing', False) else 'Disabled')}</li>
            <li><strong>Lags Used:</strong> {', '.join(map(str, config.get('lags', [])))}</li>
            <li><strong>Rolling Windows:</strong> {', '.join(map(str, config.get('rolling_windows', [])))}</li>
            <li><strong>Random Seed:</strong> {config.get('seed', 42)}</li>
        </ul>
        
        <h3>Data Sources</h3>
        <ul>
            <li><code>artist_mstreams_week.csv</code> - Spotify streaming data (target variable)</li>
            <li><code>artist_social_week.csv</code> - Social media metrics (followers, engagement)</li>
            <li><code>artist_instagram.csv</code> - Instagram demographics & engagement</li>
            <li><code>lsecondaryticket_artist_week.csv</code> - Ticket sales (LiveNation)</li>
            <li><code>tsecondaryticket_artist_week.csv</code> - Ticket sales (Ticketmaster)</li>
            <li><code>artist_mstreams_dma_week.csv</code> - Geographic streaming data</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated by Streaming Growth Analysis Pipeline</p>
        <p>Built with Python, scikit-learn, pandas, matplotlib</p>
        <p>&copy; 2026 | For Educational Purposes</p>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML report generated: {output_file}")
    print(f"   Open in browser: file://{output_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from streaming growth analysis results"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/streaming_growth',
        help='Directory containing analysis results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HTML file path (default: results_dir/report.html)'
    )
    
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
    
    # Generate report
    print(f"Generating HTML report from {results_dir}...")
    generate_html_report(results_dir, output_file)


if __name__ == "__main__":
    main()