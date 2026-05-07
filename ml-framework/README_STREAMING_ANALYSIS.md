# 🎵 Drivers of Streaming Growth Analysis

## Executive Summary

This project analyzes **factors that correlate with short-term streaming growth** for artists on Spotify. Using a machine learning pipeline, we examined how social engagement, touring activity, and audience demographics relate to week-over-week changes in streaming numbers.

**Key Findings:**
- Social engagement (followers, engagement rate) shows strong correlation with streaming growth
- Geographic diversity (number of markets with activity) positively impacts streams
- Lagged features (prior week's metrics) are strongest predictors
- Log-transformed models significantly outperform linear models due to streaming data's right-skewed distribution

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Data Sources](#data-sources)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Results & Interpretation](#results--interpretation)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)

---

## 🎯 Problem Statement

**Question:** Which factors correlate with short-term growth in Spotify streams for artists?

**Approach:** 
- Predict **week-over-week streaming growth** (target variable)
- Use **external predictors**: social metrics, ticket sales, demographics, touring activity
- Apply **proper temporal alignment** to prevent data leakage
- Compare **multiple models** to identify most predictive factors

---

## 🔬 Methodology

### 1. Data Preparation
- **Platform filtering**: Spotify only (both streaming and social metrics)
- **Artist sampling**: Top 100 artists by total streaming volume
- **Time window**: 5 years of weekly data (260 weeks)
- **Week standardization**: ISO week numbers (YYYY-WW format) for consistent merging

### 2. Feature Engineering

**Temporal Leakage Prevention:**
- Target: `number_of_streams` growth (week-over-week percentage change)
- **Excluded**: Features derived from target itself (`number_of_streams_acceleration`, etc.)
- **Included**: External predictors only

**Feature Types Created:**
- **Lagged features** (1, 2, 4 weeks ago): `max_following_lag1`, `st_event_count_lag2`
- **Rolling aggregates** (4, 8 week windows): `max_following_roll4_mean`
- **Static demographics**: Instagram age/gender/language distribution
- **Geographic diversity**: Number of unique DMAs with activity

### 3. Data Quality Handling

**Missing Value Imputation** (optional flag):
- Static demographics → Artist mean → Global median
- Touring metrics → Fill with 0 (no activity)
- Social metrics → Forward fill → Global median

**Outlier Handling**:
- Default: **Log transformation** of target variable
- Addresses right-skewed distribution of streaming data
- Alternatives: Remove outliers, Winsorize, or None

### 4. Model Training

**Temporal Train/Test Split:**
- 80% train (older data) / 20% test (recent data)
- Ensures we don't use future data to predict past

**Models Compared:**
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (coefficient of determination)

---

## 📊 Data Sources

### Input Files (CSV format)

| File | Description | Key Columns |
|------|-------------|-------------|
| `artist_mstreams_week.csv` | Weekly streaming data | `artist_id`, `week_start_date`, `number_of_streams`, `platform_name` |
| `artist_social_week.csv` | Weekly social metrics | `artist_id`, `week_start_date`, `max_following` |
| `artist_instagram.csv` | Instagram demographics | `artist_id`, `engagement_rate`, `ages_*`, `follower_*` |
| `lsecondaryticket_artist_week.csv` | Ticket sales (LiveNation) | `artist_id`, `start_date`, `st_event_count`, `st_num_tickets`, `genre_id` |
| `tsecondaryticket_artist_week.csv` | Ticket sales (Ticketmaster) | `artist_id`, `start_date`, `st_avg_price` |
| `artist_mstreams_dma_week.csv` | Geographic streaming | `artist_id`, `week_start_date`, `dma_id` |
| `*_dma_week.csv` | Geographic ticket sales | `artist_id`, `dma_id` |

**Data Scope:**
- 500 artists stratified by social media following
- 5 years of historical data
- Weekly granularity
- Spotify platform focus

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Poetry (dependency management)

### Installation

```bash
# 1. Clone or navigate to the project directory
cd ml-framework

# 2. Install dependencies with Poetry
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
  --model all
```

This will:
- ✅ Load and merge all data sources
- ✅ Apply log transformation (handles outliers)
- ✅ Train all 3 models (Linear, Random Forest, Gradient Boosting)
- ✅ Generate comprehensive visualizations
- ✅ Create model comparison report

### Advanced Options

#### With Imputation (More Data)
```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --artist-sample-size 100 \
  --model all \
  --impute-missing
```
- Increases sample size from ~7.5K to ~20K+ rows
- Applies smart domain-informed imputation

#### Single Model
```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --artist-sample-size 100 \
  --model random_forest
```

#### Without Log Transformation
```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --artist-sample-size 100 \
  --model all \
  --outlier-method none
```

#### Custom Parameters
```bash
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --artist-sample-size 200 \
  --lags 1 2 4 8 \
  --rolling-windows 4 8 12 \
  --outlier-method winsorize
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Directory with CSV files | Required |
| `--output` | Output directory | `results/streaming_growth` |
| `--artist-sample-size` | Number of artists | 100 |
| `--model` | Model(s) to train | `linear_regression` |
| `--impute-missing` | Apply smart imputation | False |
| `--outlier-method` | Outlier handling | `log_transform` |
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
├── model_comparison.csv                    # Performance comparison
├── results.json                           # Complete results
├── sampled_artists.json                   # Artist IDs used
├── feature_names.json                     # Feature list
├── plots/
│   ├── model_comparison_metrics.png       # Cross-model comparison
│   ├── feature_correlations_agnostic.png  # Generic correlations
│   ├── feature_contribution_relationships.png
│   ├── linear_regression/
│   │   ├── predicted_vs_actual.png
│   │   ├── residuals_analysis.png
│   │   ├── time_series_predictions.png
│   │   └── feature_importance.png         # Linear coefficients
│   ├── random_forest/
│   │   └── ...                            # Forest-specific importance
│   └── gradient_boosting/
│       └── ...                            # Boosting-specific importance
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
   - Compare RMSE, MAE, R² across all models
   - Identify best-performing model

2. **Feature Correlations** (`feature_correlations_agnostic.png`)
   - Shows linear correlation (Pearson r) with target
   - Model-agnostic - same for all models
   - Helps understand overall feature-target relationships

3. **Feature Importance** (per model)
   - **Linear Regression**: Absolute coefficient values
   - **Random Forest**: Gini importance (split contribution)
   - **Gradient Boosting**: Split gain
   - Each model weights features differently!

4. **Predicted vs Actual** (per model)
   - Scatter plot showing prediction accuracy
   - Perfect predictions would fall on diagonal line
   - R² score displayed

5. **Residuals Analysis** (per model)
   - Checks for prediction bias
   - Should be randomly distributed around zero

6. **Time Series Predictions** (per model)
   - Shows predictions for sample artists over time
   - Validates temporal prediction quality

### Interpreting Results

**Example Feature Importance Interpretation:**

If `max_following_lag1` has high importance:
> "An artist's follower count from the previous week is a strong predictor of current week's streaming growth. A 10% increase in followers tends to correlate with X% increase in streams."

If `dma_count_streams` is important:
> "Geographic diversity matters - artists streaming in more markets show stronger growth potential."

---

## 📁 Project Structure

```
ml-framework/
├── data/                          # Input CSV files
├── src/
│   ├── streaming_data.py         # Data loading & joining
│   ├── streaming_features.py     # Feature engineering
│   ├── model.py                  # Model registry
│   └── utils.py                  # Logging utilities
├── scripts/
│   └── streaming_growth_pipeline.py  # Main pipeline
├── results/                       # Output directory
├── notebooks/
│   └── home_assignment.ipynb     # Interactive analysis
├── README_STREAMING_ANALYSIS.md  # This file
└── pyproject.toml               # Dependencies
```

---

## 🔧 Technical Details

### Temporal Leakage Prevention

**Critical Design Choice:**
We predict streaming growth using **only external factors**, never using past streaming numbers directly.

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

### Model-Specific vs Model-Agnostic Metrics

| Metric Type | Scope | Example |
|-------------|-------|---------|
| **Correlation** | Model-agnostic | Pearson r = 0.65 (same for all models) |
| **Feature Importance** | Model-specific | Linear: 0.25, RF: 0.35, GB: 0.30 |

Different models weight features differently based on their algorithm!

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
Linear models work much better!
```

---

## 🎓 Key Learnings

1. **Temporal alignment is critical** - Using same-week data causes leakage
2. **Log transformation essential** - Streaming data is right-skewed
3. **Recent history matters most** - lag1 features typically most important
4. **Social engagement predictive** - Followers/engagement correlate with streams
5. **Geographic diversity helps** - Artists in more markets grow faster
6. **Model choice matters** - Tree-based models often outperform linear

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
**Maintained by:** [Your Name]
**Last Updated:** May 2026
