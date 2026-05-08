# Streaming Growth Analysis Pipeline
## Software Architecture & Design Document (SAD/SDD)

**Version:** 1.0  
**Date:** May 2026  
**Author:** ML Framework Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Module Structure](#3-module-structure)
4. [Data Flow](#4-data-flow)
5. [Feature Engineering](#5-feature-engineering)
6. [Machine Learning Approach](#6-machine-learning-approach)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Configuration Options](#8-configuration-options)
9. [Output Artifacts](#9-output-artifacts)
10. [Usage Examples](#10-usage-examples)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Executive Summary

### 1.1 Problem Statement

Predict future streaming growth for music artists based on historical performance metrics, enabling data-driven decisions for artist management, marketing, and resource allocation.

### 1.2 Solution Overview

A modular Python ML pipeline that:
- **Loads and joins** multiple data sources (streaming, social media, ticket sales, geographic data)
- **Engineers temporal features** (lags, rolling windows, growth rates) with leakage prevention
- **Trains and evaluates** multiple ML models (Linear Regression, Random Forest, Gradient Boosting)
- **Generates comprehensive** visualizations and HTML reports for stakeholder communication

### 1.3 Key Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error - penalizes large errors |
| **MAE** | Mean Absolute Error - average prediction error |
| **R²** | Coefficient of determination - variance explained |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ Streaming Data  │ Social Media    │ Ticket Sales    │ Geographic (DMA)      │
│ (Spotify)       │ (Instagram)     │ (Secondary)     │                       │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         └─────────────────┴─────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PHASES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Data Loading          │  StreamingDataLoader                      │
│  Phase 2: Feature Engineering   │  StreamingGrowthFeatures                  │
│  Phase 3: Train/Test Split      │  Temporal split (80/20)                   │
│  Phase 4: Feature Scaling       │  StandardScaler                           │
│  Phase 4.5: Cross-Validation    │  TimeSeriesSplit (5 folds)                │
│  Phase 5-6: Model Training      │  ModelRegistry → fit/predict              │
│  Phase 7: Results Saving        │  JSON, CSV, PKL artifacts                 │
│  Phase 8: Visualization         │  Plots + HTML Report                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUTS                                            │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ trained_model   │ predictions     │ feature         │ analysis_report       │
│ .pkl            │ .csv            │ _importance.csv │ .html                 │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Package Management** | Poetry |

---

## 3. Module Structure

### 3.1 Directory Layout

```
ml-framework/
├── scripts/
│   └── streaming_growth_pipeline.py   # Main entry point
├── src/
│   ├── streaming_data.py              # Data loading & joining
│   ├── streaming_features.py          # Feature engineering
│   ├── model.py                       # Model registry & wrappers
│   ├── metrics.py                     # Evaluation metrics
│   ├── plotting.py                    # Visualization generation
│   ├── html_report.py                 # HTML report generation
│   └── utils.py                       # Logging, I/O utilities
├── data/
│   └── *.csv                          # Input data files
├── results/
│   └── {experiment}/                  # Output artifacts
├── tests/
│   ├── test_unit.py
│   └── test_e2e.py
└── docs/
    └── DESIGN.md                      # This document
```

### 3.2 Module Responsibilities

| Module | Class/Function | Responsibility |
|--------|----------------|----------------|
| `streaming_data.py` | `StreamingDataLoader` | Load CSVs, filter Spotify, sample artists, join data sources |
| `streaming_features.py` | `StreamingGrowthFeatures` | Create lagged/rolling/growth features, prepare X/y |
| `model.py` | `ModelRegistry` | Factory pattern for model creation |
| `model.py` | `ScikitLearnModel` | Unified wrapper for sklearn models |
| `metrics.py` | `calculate_metrics()` | Compute RMSE, MAE, R² |
| `metrics.py` | `calculate_cv_metrics()` | Aggregate CV fold statistics |
| `plotting.py` | `create_visualizations()` | Generate all plots and HTML report |
| `utils.py` | `save_json()`, `save_dataframe()` | File I/O with logging |

---

## 4. Data Flow

### 4.1 Input Data Sources

| File | Description | Key Columns |
|------|-------------|-------------|
| `artist_mstreams_week.csv` | Weekly streaming counts | `artist_id`, `number_of_streams`, `week_start_date` |
| `artist_social_week.csv` | Social media followers | `artist_id`, `max_following` |
| `artist_instagram.csv` | Instagram demographics | Age, gender, ethnicity, engagement metrics |
| `lsecondaryticket_artist_week.csv` | Ticket sales (L source) | `st_num_tickets`, `st_order_value` |
| `tsecondaryticket_artist_week.csv` | Ticket sales (T source) | Same as above |
| `artist_mstreams_dma_week.csv` | Geographic streaming | DMA-level metrics |

### 4.2 Data Processing Pipeline

```
1. LOAD RAW DATA
   ├── Filter to Spotify platform only
   ├── Convert dates to ISO week format (YYYY-WW)
   └── Create cumulative streams per artist

2. SAMPLE ARTISTS
   ├── Rank by total streaming volume
   ├── Filter: minimum 20 weeks of data
   └── Select top N artists (default: 100)

3. JOIN DATA SOURCES
   └── Streaming (base) ← Social ← Tickets ← DMA ← Instagram

4. ENGINEER FEATURES
   ├── Lagged features (1, 2, 4 weeks)
   ├── Rolling windows (4, 8 weeks) - mean & std
   ├── Growth rates (1-week, 4-week)
   └── Momentum & acceleration

5. PREPARE FOR MODELING
   ├── Create target variable (growth rate or cumulative)
   ├── Handle outliers (log_transform default)
   ├── Temporal train/test split (80/20)
   └── Scale features (StandardScaler)
```

---

## 5. Feature Engineering

### 5.1 Feature Types

| Type | Example | Formula | Purpose |
|------|---------|---------|---------|
| **Lag** | `streams_lag1` | `value[t-1]` | Point-in-time historical value |
| **Lag** | `streams_lag4` | `value[t-4]` | Value from 4 weeks ago |
| **Rolling Mean** | `streams_roll4_mean` | `mean(value[t-4:t-1])` | Smoothed trend |
| **Rolling Std** | `streams_roll4_std` | `std(value[t-4:t-1])` | Volatility indicator |
| **Growth Rate** | `streams_growth_1w` | `(v[t] - v[t-1]) / v[t-1]` | Week-over-week change |
| **Momentum** | `streams_momentum_8w` | `recent_growth - historical_growth` | Trend acceleration |

### 5.2 Leakage Prevention

**Critical Design Principle:** All features must use only information available at prediction time.

| Safeguard | Implementation |
|-----------|----------------|
| **Temporal lagging** | All features use `shift(1)` or greater |
| **Rolling window offset** | `shift(1).rolling()` - excludes current period |
| **Target exclusion** | Features derived from target column are excluded |
| **Temporal split** | Train on past, test on future (no shuffling) |
| **TimeSeriesSplit CV** | Each fold respects temporal ordering |

### 5.3 Target Variable

Two modes supported:

| Mode | Target | Use Case |
|------|--------|----------|
| **Growth Rate** (default) | `(streams[t+1] - streams[t]) / streams[t]` | Relative change prediction |
| **Cumulative** (`--use-cumulative`) | `cumulative_streams[t+1]` | Absolute value prediction |

---

## 6. Machine Learning Approach

### 6.1 Available Models

| Model | Type | Strengths | Feature Importance |
|-------|------|-----------|-------------------|
| **Linear Regression** | Linear | Interpretable coefficients | Absolute coefficients |
| **Random Forest** | Ensemble | Handles non-linearity, robust | Gini importance |
| **Gradient Boosting** | Ensemble | Often best accuracy | Split gain |

### 6.2 Model Registry Pattern

```python
# Usage
model = ModelRegistry.create(model_type='random_forest_regressor', n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model.save('model.pkl')
```

### 6.3 Validation Strategy

**TimeSeriesSplit Cross-Validation** (default: 5 folds)

```
Fold 1: Train [----]     Validate [--]
Fold 2: Train [------]   Validate [--]
Fold 3: Train [--------] Validate [--]
Fold 4: Train [----------] Validate [--]
Fold 5: Train [------------] Validate [--]
```

- Each fold trains on chronologically earlier data
- Validates on later data (mimics real forecasting)
- Reports mean ± std across folds for stability assessment

### 6.4 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | `√(mean((y - ŷ)²))` | Lower is better; penalizes large errors |
| **MAE** | `mean(\|y - ŷ\|)` | Lower is better; average error magnitude |
| **R²** | `1 - SS_res/SS_tot` | Higher is better; 1.0 = perfect fit |

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Spotify-only filtering** | Ensures consistent data source; Spotify is primary streaming platform |
| **Temporal train/test split** | Prevents future data leaking into training; mimics production use |
| **StandardScaler** | Normalizes features for linear models; doesn't hurt tree models |
| **Log transform for outliers** | Growth rates can be extreme (+10000%); log compresses scale |
| **Minimum 20 weeks per artist** | Ensures sufficient history for lag/rolling features |
| **ModelRegistry pattern** | Decouples model creation from usage; easy to add new models |
| **HTML report generation** | Self-contained deliverable for non-technical stakeholders |

---

## 8. Configuration Options

### 8.1 Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | *required* | Directory containing input CSV files |
| `--output` | `results/streaming_growth` | Output directory for all artifacts |
| `--artist-sample-size` | `100` | Number of artists to analyze |
| `--min-weeks` | `20` | Minimum weeks of data required per artist |
| `--target-col` | `number_of_streams` | Column to predict |
| `--use-cumulative` | `False` | Predict cumulative instead of growth rate |
| `--lags` | `[1, 2, 4]` | Lag periods in weeks |
| `--rolling-windows` | `[4, 8]` | Rolling window sizes in weeks |
| `--model` | `linear_regression` | Model type (`linear_regression`, `random_forest`, `gradient_boosting`, `all`) |
| `--cv-folds` | `5` | Number of cross-validation folds |
| `--outlier-method` | `log_transform` | Outlier handling (`none`, `remove`, `winsorize`, `log_transform`) |
| `--outlier-threshold` | `3.0` | Z-score threshold for outlier detection |
| `--seed` | `42` | Random seed for reproducibility |
| `--impute-missing` | `False` | Apply smart imputation for missing values |

---

## 9. Output Artifacts

### 9.1 Directory Structure

```
results/{experiment_name}/
├── sampled_artists.json        # List of artist IDs analyzed
├── feature_names.json          # List of feature columns
├── model_comparison.csv        # Performance metrics for all models
├── results.json                # Complete results with config
├── {model_name}/
│   ├── trained_model.pkl       # Serialized model
│   ├── predictions.csv         # Test set predictions
│   └── feature_importance.csv  # Feature rankings
└── plots/
    ├── analysis_report.html    # Comprehensive HTML report
    ├── model_comparison_metrics.png
    ├── cv_vs_test_performance.png
    ├── combined_feature_importance.png
    ├── feature_correlations_agnostic.png
    └── {model_name}/
        ├── predicted_vs_actual.png
        ├── residuals_analysis.png
        ├── time_series_predictions.png
        └── feature_importance.png
```

### 9.2 Key Output Files

| File | Description | Use Case |
|------|-------------|----------|
| `model_comparison.csv` | Side-by-side model metrics | Model selection |
| `analysis_report.html` | Interactive HTML report | Stakeholder presentation |
| `predictions.csv` | Actual vs predicted values | Error analysis |
| `feature_importance.csv` | Ranked feature list | Feature selection, insights |
| `trained_model.pkl` | Serialized model | Production deployment |

---

## 10. Usage Examples

### 10.1 Basic Usage

```bash
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir data \
    --output results/baseline
```

### 10.2 Compare All Models

```bash
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir data \
    --output results/model_comparison \
    --model all \
    --artist-sample-size 200
```

### 10.3 Custom Feature Engineering

```bash
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir data \
    --output results/extended_features \
    --lags 1 2 4 8 12 \
    --rolling-windows 4 8 12 16 \
    --model gradient_boosting
```

### 10.4 Cumulative Prediction Mode

```bash
poetry run python scripts/streaming_growth_pipeline.py \
    --data-dir data \
    --output results/cumulative_mode \
    --use-cumulative \
    --outlier-method none
```

---

## 11. Future Enhancements

### 11.1 Short-Term

- [ ] Multi-horizon prediction (predict multiple weeks ahead simultaneously)
- [ ] Hyperparameter tuning integration (GridSearchCV, Optuna)
- [ ] Additional models (XGBoost, LightGBM)
- [ ] Feature selection automation (RFE, importance thresholding)

### 11.2 Medium-Term

- [ ] Real-time prediction API (FastAPI/Flask)
- [ ] Model versioning and experiment tracking (MLflow)
- [ ] Automated retraining pipeline
- [ ] A/B testing framework for model comparison

### 11.3 Long-Term

- [ ] Deep learning models (LSTM, Transformer)
- [ ] Causal inference for feature impact analysis
- [ ] Multi-task learning (predict multiple metrics)
- [ ] Explainability dashboard (SHAP values)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **DMA** | Designated Market Area - geographic region for media measurement |
| **Lag feature** | Value from a previous time period |
| **Rolling window** | Aggregate statistic over a sliding time window |
| **Data leakage** | Using future information to predict the past |
| **TimeSeriesSplit** | Cross-validation that respects temporal ordering |
| **Feature importance** | Measure of how much a feature contributes to predictions |

---

## Appendix B: References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [pandas Documentation](https://pandas.pydata.org/)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
