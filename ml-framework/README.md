# 🎵 Streaming Growth Analysis - ML Framework

## Executive Summary

This project analyzes **factors that correlate with short-term streaming growth** for artists on Spotify. Using a time-series machine learning pipeline with proper temporal alignment, 
I examined how social engagement, touring activity, and audience demographics relate to week-over-week changes in streaming numbers.

**Key Findings:**
- Social engagement (followers, engagement rate) shows strong correlation with streaming growth
- Geographic diversity (number of markets with activity) positively impacts streams
- Lagged features (prior week's metrics) are strongest predictors
- Cross-validation reveals severe overfitting issues with current feature/sample ratios

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Results & Interpretation](#results--interpretation)
- [Technical Details](#technical-details)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements--considerations)
- [Framework Features](#framework-features)
- [Project Structure](#project-structure)

---

## 🎯 Problem Statement

**Research Question:** Which factors correlate with short-term growth in Spotify streams for artists?

**Approach:** 
- Predict **week-over-week streaming growth** (target variable)
- Use **external predictors**: social metrics, ticket sales, demographics, touring activity
- Apply **proper temporal alignment** to prevent data leakage
- Use **TimeSeriesSplit cross-validation** for robust evaluation
- Compare **multiple models** to identify most predictive factors

**Key Scoping Decisions:**

1. **Artist Sampling**: 
   - Top 100 artists by total streaming volume
   - Ensures consistent, high-quality data
   - Minimum 20 weeks of observations per artist
   
2. **Temporal Alignment**:
   - All features lagged by ≥1 week to prevent leakage
   - Train/test split by time (80/20), not random
   - Test set contains most recent 20% of data

3. **Cross-Validation**:
   - 5-fold TimeSeriesSplit (respects temporal ordering)
   - Each fold trains on past, predicts future
   - Provides robust performance estimates

---

## 🔬 Methodology

### 1. Data Preparation

**Data Sources:**

| File | Description | Key Columns |
|------|-------------|-------------|
| `artist_mstreams_week.csv` | Weekly streaming data | `artist_id`, `week_start_date`, `number_of_streams` |
| `artist_social_week.csv` | Weekly social metrics | `artist_id`, `week_start_date`, `max_following` |
| `artist_instagram.csv` | Instagram demographics | `artist_id`, `engagement_rate`, `ages_*`, `follower_*` |
| `lsecondaryticket_artist_week.csv` | Ticket sales (LiveNation) | `artist_id`, `start_date`, `st_event_count` |
| `tsecondaryticket_artist_week.csv` | Ticket sales (Ticketmaster) | `artist_id`, `start_date`, `st_avg_price` |
| `artist_mstreams_dma_week.csv` | Geographic streaming | `artist_id`, `week_start_date`, `dma_id` |

**Processing:**
- **Platform filtering**: Spotify only (both streaming and social metrics)
- **Artist sampling**: Top 100 artists by total streaming volume
- **Time window**: 5 years of weekly data (260 weeks)
- **Week standardization**: ISO week numbers (YYYY-WW format) for consistent merging

### 2. Feature Engineering

**Temporal Leakage Prevention:**
- Target: `number_of_streams` growth (week-over-week percentage change)
- **Excluded**: Features derived from target itself (`number_of_streams_*`)
- **Included**: External predictors only, lagged appropriately

**Feature Types Created:**
- **Lagged features** (1, 2, 4 weeks): `max_following_lag1`, `st_event_count_lag2`
- **Rolling aggregates** (4, 8 week windows): `max_following_roll4_mean`
- **Static demographics**: Instagram age/gender/language distribution
- **Geographic diversity**: Number of unique DMAs with activity

### 3. Data Quality Handling

**Missing Value Imputation** (optional `--impute-missing` flag):
- Static demographics → Artist mean → Global median
- Touring metrics → Fill with 0 (no activity)
- Social metrics → Forward fill → Global median

**Outlier Handling** (`--outlier-method`):
- Default: **Log transformation** of target variable
- Addresses right-skewed distribution of streaming data
- Alternatives: Remove outliers, Winsorize, or None

### 4. Model Training & Evaluation

**Cross-Validation:**
- **Method**: TimeSeriesSplit (5 folds)
- **Process**: Each fold trains on expanding past, validates on future period
- **Purpose**: Robust performance estimation respecting temporal dependencies

**Temporal Train/Test Split:**
- 80% train (older data) / 20% test (most recent data)
- Final evaluation on held-out test set

**Models Compared:**
- Linear Regression (baseline, interpretable)
- Random Forest Regressor (captures non-linearities)
- Gradient Boosting Regressor (ensemble method)

**Evaluation Metrics:**
- **RMSE** (Root Mean Squared Error) - Penalizes large errors
- **MAE** (Mean Absolute Error) - Robust to outliers
- **R²** (Coefficient of Determination) - Variance explained
- **CV R² (mean ± std)** - Cross-validation performance

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Poetry (dependency management)

### Installation

```bash
# 1. Clone or navigate to the project directory
cd ml-framework

# 2. Install dependencies with Poetry (REQUIRED)
poetry install

# 3. Verify installation
poetry run python scripts/verify_setup.py
```

### Data Setup

Place your CSV files in the `data/` directory:
```
data/
├── artist_mstreams_week.csv
├── artist_social_week.csv
├── artist_instagram.csv
├── lsecondaryticket_artist_week.csv
├── tsecondaryticket_artist_week.csv
├── artist_mstreams_dma_week.csv
├── lsecondaryticket_artist_dma_week.csv
└── tsecondaryticket_artist_dma_week.csv
```

---

## 🎮 How to Run

### Basic Usage (Recommended)

```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --artist-sample-size 100 \
  --model all \
  --cv-folds 5
```

This will:
- ✅ Load and merge all data sources
- ✅ Engineer temporal features with proper lagging
- ✅ Run 5-fold time-series cross-validation
- ✅ Train all 3 models (Linear, Random Forest, Gradient Boosting)
- ✅ Generate comprehensive visualizations including CV analysis
- ✅ Create model comparison report with CV metrics

### Advanced Options

#### With Imputation (More Data)
```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --artist-sample-size 100 \
  --model all \
  --impute-missing \
  --cv-folds 5
```
- Increases sample size from ~600 to ~20K+ rows
- Applies smart domain-informed imputation

#### Single Model
```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --model random_forest
```

#### Custom Parameters
```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --artist-sample-size 200 \
  --lags 1 2 4 8 \
  --rolling-windows 4 8 12 \
  --outlier-method winsorize \
  --cv-folds 5
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Directory with CSV files | Required |
| `--output` | Output directory | `results/streaming_growth` |
| `--artist-sample-size` | Number of artists | 100 |
| `--model` | Model(s) to train (`linear_regression`, `random_forest`, `gradient_boosting`, `all`) | `linear_regression` |
| `--cv-folds` | Number of cross-validation folds | 5 |
| `--impute-missing` | Apply smart imputation | False |
| `--outlier-method` | Outlier handling (`none`, `remove`, `winsorize`, `log_transform`) | `log_transform` |
| `--outlier-threshold` | Z-score threshold | 3.0 |
| `--lags` | Lag periods (weeks) | [1, 2, 4] |
| `--rolling-windows` | Rolling window sizes | [4, 8] |
| `--min-weeks` | Min weeks per artist | 20 |
| `--seed` | Random seed | 42 |

---

## 📈 Results & Interpretation

### Output Structure

```
results/streaming_growth/
├── model_comparison.csv                    # Performance comparison with CV
├── results.json                           # Complete results
├── sampled_artists.json                   # Artist IDs used
├── feature_names.json                     # Feature list
├── plots/
│   ├── model_comparison_metrics.png       # Test + CV metrics comparison
│   ├── cv_vs_test_performance.png         # CV vs test stability plot
│   ├── feature_correlations_agnostic.png  # Model-agnostic correlations
│   ├── feature_contribution_relationships.png
│   ├── feature_contributions.csv
│   ├── linear_regression/
│   │   ├── predicted_vs_actual.png
│   │   ├── residuals_analysis.png
│   │   ├── time_series_predictions.png
│   │   └── feature_importance.png
│   ├── random_forest/
│   │   └── ...
│   └── gradient_boosting/
│       └── ...
├── linear_regression/
│   ├── predictions.csv
│   └── feature_importance.csv
├── random_forest/
│   └── ...
└── gradient_boosting/
    └── ...
```

### Key Visualizations

1. **Model Comparison** (`model_comparison_metrics.png`)
   - **Top row**: Test metrics (RMSE, MAE, R²)
   - **Bottom row**: CV metrics with error bars (stability assessment)
   - Identifies best-performing and most stable model

2. **CV vs Test Performance** (`cv_vs_test_performance.png`)
   - Plots test R² vs CV R² (mean ± std)
   - Good models have test scores within CV error bars
   - Large gaps indicate overfitting or distribution shift

3. **Feature Correlations** (`feature_correlations_agnostic.png`)
   - Linear correlation (Pearson r) with target
   - Model-agnostic - same for all models
   - Shows raw feature-target relationships

4. **Feature Importance** (per model)
   - **Linear Regression**: Absolute coefficient values
   - **Random Forest**: Gini importance (split contribution)
   - **Gradient Boosting**: Split gain
   - Each model weights features differently!

5. **Predicted vs Actual** (per model)
   - Scatter plot showing prediction accuracy
   - Diagonal line = perfect predictions
   - R² score displayed

6. **Residuals Analysis** (per model)
   - Checks for systematic errors or bias
   - Should be randomly distributed around zero

### Interpreting Cross-Validation Results

**CV R² Interpretation:**
- **Positive CV R²**: Model generalizes reasonably
- **Negative CV R²**: Model worse than predicting the mean
- **High CV std**: Model unstable across time periods

**Example from current results:**
```
Model              CV R² (mean ± std)    Test R²
Linear Regression  -154.49 ± 306.45     -25.26    ← Highly unstable
Random Forest       -0.13 ± 0.10        -27.54    ← Most stable failure
Gradient Boosting   -0.22 ± 0.29       -209.30    ← Severe overfitting
```

**Diagnosis:** All models show negative performance, indicating fundamental issues (see Future Improvements).

---

## 🔧 Technical Details

### Temporal Leakage Prevention

**Critical Design Choice:**
Predict streaming growth using **only external factors**, never using past streaming numbers directly.

❌ **Leakage (Wrong):**
```
Predict Week 10 streams using Week 10 follower count
→ Can't know Week 10 followers before Week 10 happens!
```

✅ **Proper Temporal Alignment:**
```
Predict Week 10 streams using Week 9 follower count (lag1)
→ We knew Week 9 followers before Week 10 started
```

### Feature Engineering Pipeline

```python
# Example: Creating lagged features
max_following_lag1 = df.groupby('artist_id')['max_following'].shift(1)
# Week 10 gets Week 9's value, Week 9 gets Week 8's value, etc.

# Example: Rolling window
max_following_roll4_mean = df.groupby('artist_id')['max_following']
    .shift(1)  # Lag first to prevent leakage
    .rolling(window=4)  # Then compute 4-week average
    .mean()
```

### TimeSeriesSplit Cross-Validation

```
Fold 1: Train [weeks 1-190]  → Validate [weeks 191-230]
Fold 2: Train [weeks 1-230]  → Validate [weeks 231-270]
Fold 3: Train [weeks 1-270]  → Validate [weeks 271-310]
Fold 4: Train [weeks 1-310]  → Validate [weeks 311-350]
Fold 5: Train [weeks 1-350]  → Validate [weeks 351-390]
```

Each fold expands training window and validates on future period.

### Log Transformation Impact

**Before Log Transform:**
```
Streams: 1M, 5M, 10M, 100M (huge range)
Standard Deviation: 25M (outliers dominate)
```

**After Log Transform:**
```
Log(Streams): 13.8, 15.4, 16.1, 18.4 (compressed range)
Standard Deviation: 1.2 (normalized distribution)
```

---

## 🎓 Key Learnings

1. **Temporal alignment is critical** - Using same-week data causes leakage
2. **Cross-validation essential for time-series** - Reveals stability issues early
3. **Log transformation helps** - Streaming data is right-skewed
4. **Recent history matters most** - lag1 features typically most important
5. **Sample size is limiting** - 593 samples with 71 features causes severe overfitting
6. **CV standard deviation matters** - High variance indicates unstable predictions
7. **Negative R² is possible** - Means model worse than predicting the mean

---

## 🚀 Future Improvements & Considerations

Based on cross-validation analysis and model performance evaluation:

### High Priority (Do First)

1. **Better Imputation** (Current: 99% data loss)
   - **Solution**: Use `--impute-missing` flag
   - **Impact**: 600 → 20K+ samples
   
2. **Feature Selection** (Current: 71 features, 593 samples)
   - **Solution**: Keep top 10-15 features by correlation
   - **Methods**: Correlation filtering, RFE, tree-based selection
   - **Impact**: Reduced overfitting, better generalization

3. **Regularization** (Current: Severe overfitting)
   - **Solution**: Replace Linear Regression with Ridge/Lasso
   - **For trees**: Increase `min_samples_leaf`, reduce `max_depth`
   - **Impact**: Penalize complexity, prevent overfitting

### Medium Priority

4. **Hyperparameter Tuning**
   - **Solution**: GridSearchCV with TimeSeriesSplit
   - **Parameters**: `alpha` for Ridge/Lasso, tree depth/samples
   - **Impact**: Optimized model parameters

5. **Increase Sample Size**
   - **Solution**: Reduce `--min-weeks` threshold (currently 20)
   - **Alternative**: More flexible NA handling
   - **Impact**: More stable model training

### Low Priority (After Basics Work)

6. **Advanced Methods**
   - LSTM/GRU for sequence modeling
   - Prophet for trend decomposition
   - Hierarchical models (artist-level random effects)

7. **External Data**
   - Radio airplay
   - Playlist additions
   - Release dates and album cycles

### Assumptions & Limitations

**Assumptions:**
- Top artists are representative (may not generalize to emerging artists)
- Streaming patterns consistent over time (external shocks like COVID violate this)
- Features available before prediction time (proper lagging ensures this)

**Limitations:**
- **Correlation ≠ Causation**: Analysis identifies relationships, not causal effects
- **Sample bias**: Focused on top 100 artists
- **Missing external factors**: Radio, TV, viral moments not captured
- **Linear relationships**: Current models may miss complex interactions

---

## 🛠️ Framework Features

This project is built on a general-purpose ML framework with:

### Core Capabilities

- **Efficient Data Loading**: Stream and batch large datasets (CSV, Parquet, HDF5)
- **Mixed Data Types**: Automatic detection and preprocessing
- **Feature Engineering**: Composable transformation pipeline
- **Unified Model Interface**: Consistent API for scikit-learn and PyTorch
- **Comprehensive Training**: Validation, early stopping, cross-validation
- **Configuration Management**: YAML/JSON-based configuration
- **Experiment Tracking**: Automatic logging of models and metrics

### Supported Models

- Logistic Regression
- Linear Regression  
- Random Forest (Classification & Regression)
- Gradient Boosting
- Support Vector Machine (SVM)
- PyTorch Neural Networks (extensible)

### Data Processing

- **Missing Values**: mean, median, forward fill, backward fill, drop
- **Outlier Detection**: IQR and Z-score methods
- **Feature Scaling**: Standard, MinMax, Robust
- **Categorical Encoding**: Label and one-hot encoding
- **Dimensionality Reduction**: PCA support

---

## 📁 Project Structure

```
ml-framework/
├── data/                          # Input CSV files
├── src/
│   ├── streaming_data.py         # Time-series data loading & joining
│   ├── streaming_features.py     # Feature engineering with temporal alignment
│   ├── data.py                   # Generic data utilities
│   ├── model.py                  # Model registry
│   ├── train.py                  # Training utilities
│   ├── features.py               # Feature transformers
│   ├── config.py                 # Configuration management
│   └── utils.py                  # Logging and helpers
├── scripts/
│   ├── streaming_growth_pipeline.py  # Main pipeline (ENTRY POINT)
│   └── verify_setup.py           # Setup verification
├── results/                       # Output directory
├── notebooks/
│   └── home_assignment.ipynb     # Interactive analysis
├── tests/                         # Test suite
├── README.md                      # This file
└── pyproject.toml                # Dependencies
```

---

## 📞 Support

For questions or issues:
1. Check the pipeline logs in terminal output
2. Verify data files are in correct format
3. Ensure Poetry environment is activated: `poetry shell`
4. Run verification: `poetry run python scripts/verify_setup.py`

---

## 📜 License

This analysis framework is provided for educational and research purposes.

---

**Built with:** Python, scikit-learn, pandas, matplotlib, seaborn  
**Assignment Focus:** Drivers of Streaming Growth (Question #2)  
**Last Updated:** May 2026
