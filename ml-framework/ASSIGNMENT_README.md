# Streaming Growth Analysis - Take-Home Assignment

## Overview

This project analyzes **drivers of streaming growth** for artists using time-series analysis with proper temporal alignment and leakage prevention. The goal is to quantify which factors correlate with short-term growth in Spotify streams.

**Assignment Focus**: Question #2 - Drivers of streaming growth

## Problem Framing and Scope

### Research Question
What factors correlate with week-over-week growth in artist streaming numbers?

### Approach
- **Target Variable**: Percentage change in `number_of_streams` (week-over-week growth)
- **Predictive Features**: Lagged social metrics, demographics, and event signals
- **Temporal Scope**: Weekly granularity, focusing on recent years
- **Artist Scope**: Top 100 artists by total streams (ensuring data richness)

### Key Scoping Decisions

1. **Artist Sampling**: 
   - Selected top 100 artists by total streaming volume
   - Rationale: Focus on artists with consistent, high-quality data
   - Ensures minimum 20 weeks of observations per artist
   - Documented in `results/streaming_growth/sampled_artists.json`

2. **Feature Engineering**:
   - All features lagged by at least 1 week to prevent temporal leakage
   - Target is week t, features from week t-1 and earlier
   - Focus on social engagement, ticket sales, and streaming momentum
   - Rolling window aggregates for trend detection

3. **Temporal Split**:
   - Train/test split by time (80/20), not random
   - Respects time-series nature and prevents leakage
   - Test set contains most recent 20% of data

## Data Used

### Primary Datasets
- **artist_mstreams_week.csv**: Weekly streaming metrics (target variable)
- **artist_social_week.csv**: Weekly social media engagement metrics
- **artist_instagram.csv**: Instagram follower demographics
- **lsecondaryticket_artist_week.csv**: Secondary ticket sales (tour activity proxy)
- **tsecondaryticket_artist_week.csv**: Additional ticket sales data

### Data Joins
All datasets joined on `artist_id` and `week` with temporal alignment:
- Primary key: (`artist_id`, `week`)
- Left join from streaming data to preserve all streaming observations
- Missing social/ticket data filled as NA (handled in feature engineering)

### Data Quality
- **Completeness**: Focused on top artists with consistent reporting
- **Temporal Alignment**: All data aligned to ISO week boundaries
- **Missing Data**: Dropped observations with missing target or features (~5-10% typical)

## Pipeline Design

### Architecture

The pipeline follows a modular, class-based design with clear separation of concerns:

```
streaming_growth_pipeline.py (main orchestrator)
├── StreamingDataLoader (src/streaming_data.py)
│   ├── Load multiple CSV sources
│   ├── Sample top N artists
│   └── Join datasets with temporal alignment
│
├── StreamingGrowthFeatures (src/streaming_features.py)
│   ├── Calculate growth rates
│   ├── Create lagged features
│   ├── Engineer rolling aggregates
│   └── Prevent temporal leakage
│
└── ModelRegistry & Trainer (src/model.py, src/train.py)
    ├── Train regression models
    ├── Evaluate performance
    └── Extract feature importance
```

### Key Modules

1. **`src/streaming_data.py`**: Time-series data loading
   - Handles weekly granularity
   - Artist sampling strategy
   - Temporal join logic

2. **`src/streaming_features.py`**: Feature engineering
   - Growth rate calculations
   - Lagged feature creation (1, 2, 4 week lags)
   - Rolling window statistics (4, 8 week windows)
   - Momentum indicators

3. **`scripts/streaming_growth_pipeline.py`**: Main pipeline
   - Orchestrates end-to-end workflow
   - Temporal train/test split
   - Model training and evaluation
   - Results persistence

## How to Run

### Prerequisites

```bash
# Install dependencies
poetry install

# Verify setup
poetry run python scripts/verify_setup.py
```

### Basic Usage

```bash
# Run with default settings (top 100 artists, linear regression)
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data/artist_performance \
  --output results/streaming_growth
```

### Advanced Usage

```bash
# Custom configuration
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data/artist_performance \
  --output results/streaming_growth \
  --artist-sample-size 200 \
  --model random_forest \
  --lags 1 2 4 8 \
  --rolling-windows 4 8 12 \
  --cv-folds 5
```

### Configuration File

Alternatively, use the configuration file:

```bash
# See streaming_config.yaml for full configuration options
poetry run python scripts/streaming_growth_pipeline.py \
  --config streaming_config.yaml
```

## Modeling and Evaluation Approach

### Model Selection

**Primary Model**: Linear Regression
- **Rationale**: Interpretability is crucial for understanding feature relationships
- **Benefit**: Coefficient magnitudes directly indicate feature importance
- **Limitation**: Assumes linear relationships

**Secondary Models**: Random Forest, Gradient Boosting
- **Purpose**: Capture non-linear relationships and interactions
- **Benefit**: Feature importance rankings
- **Trade-off**: Less interpretable than linear models

### Evaluation Metrics

1. **RMSE (Root Mean Squared Error)**
   - Primary metric for prediction accuracy
   - Penalizes large errors
   - Interpretable in same units as target (growth rate)

2. **MAE (Mean Absolute Error)**
   - Robust to outliers
   - Average prediction error magnitude

3. **R² (Coefficient of Determination)**
   - Proportion of variance explained
   - Indicates overall model fit
   - Scale-independent comparison metric

### Cross-Validation Strategy

- **Method**: Time-series split (5 folds)
- **Rationale**: Respects temporal ordering
- **Process**: Train on past, predict future (no random shuffling)
- **Purpose**: Estimate generalization error with proper time dependencies

## Assumptions and Limitations

### Key Assumptions

1. **Temporal Leakage Prevention**
   - All features lagged by ≥1 week
   - Target is week t, features from week t-1 and earlier
   - Train/test split respects temporal ordering

2. **Artist Sampling**
   - Top N artists by volume are representative
   - Focus on data-rich cases improves signal quality
   - May not generalize to emerging artists

3. **Data Quality**
   - Streaming counts are accurate and complete
   - Social metrics reflect genuine engagement
   - Ticket sales proxy for tour activity

4. **Stationarity**
   - Growth patterns are somewhat consistent over time
   - External events (COVID, algorithm changes) may violate this

### Limitations

1. **Causality vs Correlation**
   - Analysis identifies correlations, not causal relationships
   - Cannot claim X causes Y without experimental design
   - Confounding variables may exist

2. **Sample Bias**
   - Focused on top artists (not representative of all artists)
   - Results may not apply to niche or emerging artists
   - Geographic bias (ticket sales primarily U.S.)

3. **Feature Engineering**
   - Limited to provided datasets
   - May miss important external factors (radio play, TV appearances, etc.)
   - Rolling windows chosen heuristically

4. **Model Complexity**
   - Linear models may miss non-linear effects
   - Feature interactions not explicitly modeled
   - Could benefit from polynomial features or deeper models

### Risks and Considerations

1. **Data Leakage**: Mitigated through strict temporal splits and lagging
2. **Overfitting**: Addressed via time-series CV and regularization
3. **Scale Differences**: Features standardized before modeling
4. **Missing Data**: Handled by dropping rows (may introduce bias)
5. **Outliers**: Growth rates clipped at ±1000% to prevent extreme values

## Findings

### Preliminary Results

(Note: Actual results depend on running the pipeline with real data)

**Expected Key Drivers**:
1. **Lagged Social Engagement**:
   - Instagram follower growth (lag 1-2 weeks)
   - Social media engagement metrics
   - Hypothesis: Viral social content drives streaming

2. **Tour Activity**:
   - Ticket sales volume (lag 1-4 weeks)
   - Tour announcement timing
   - Hypothesis: Live events generate streaming momentum

3. **Streaming Momentum**:
   - Recent growth rates (autocorrelation)
   - Rolling average trends
   - Hypothesis: Success begets success (momentum effects)

4. **Seasonal Patterns**:
   - Week-of-year effects
   - Holiday periods
   - Hypothesis: Systematic temporal patterns

### Interpretation Guidelines

- **Coefficient Magnitude**: Indicates feature impact strength
- **Feature Importance**: Rankings show relative predictive power
- **R² Score**: Overall model explanatory power
- **Residual Analysis**: Reveals systematic errors or patterns

## Next Steps with More Time

### Immediate Improvements (1-2 hours)

1. **Feature Enhancement**:
   - Add platform diversity (YouTube, TikTok if available)
   - Create interaction terms (social × tour activity)
   - Engineer genre/demographic features

2. **Model Refinement**:
   - Hyperparameter tuning via grid search
   - Ensemble methods (stacking, blending)
   - Regularization (Ridge, Lasso) for feature selection

3. **Validation**:
   - Per-artist model evaluation
   - Stratified analysis by popularity tiers
   - Residual diagnostics

### Medium-Term Extensions (1 week)

1. **Advanced Modeling**:
   - LSTM/GRU for sequence modeling
   - Prophet for trend decomposition
   - Hierarchical models (artist-level random effects)

2. **Causal Analysis**:
   - Difference-in-differences for tour impacts
   - Regression discontinuity for threshold effects
   - Propensity score matching for quasi-experiments

3. **Interactive Dashboard**:
   - Plotly/Streamlit visualization
   - Artist-level drill-down
   - Feature contribution waterfall charts

### Long-Term Research (1 month+)

1. **External Data Integration**:
   - Radio airplay data
   - Playlist addition events
   - Media mentions (news, blogs)
   - Release dates and album cycles

2. **Heterogeneous Effects**:
   - Genre-specific models
   - Career stage stratification (emerging vs established)
   - Geographic market analysis

3. **Prescriptive Analytics**:
   - Optimization for tour scheduling
   - Social media strategy recommendations
   - Budget allocation for promotion

## Project Structure

```
ml-framework/
├── src/
│   ├── streaming_data.py       # Time-series data loader
│   ├── streaming_features.py   # Feature engineering with lags
│   ├── data.py                 # Generic data utilities
│   ├── model.py                # Model registry
│   ├── train.py                # Training utilities
│   ├── config.py               # Configuration management
│   └── utils.py                # Logging and helpers
├── scripts/
│   ├── streaming_growth_pipeline.py  # Main pipeline (THIS IS THE ENTRY POINT)
│   └── verify_setup.py         # Setup verification
├── data/
│   └── artist_performance/     # Place datasets here
├── results/
│   └── streaming_growth/       # Output directory
├── streaming_config.yaml       # Configuration file
├── ASSIGNMENT_README.md        # This file
├── pyproject.toml              # Dependencies
└── README.md                   # General framework docs
```

## Dependencies

Core dependencies (see `pyproject.toml`):
- `pandas ^2.0.0` - Data manipulation
- `numpy ^1.24.0` - Numerical operations
- `scikit-learn ^1.3.0` - ML models and metrics
- `pyyaml ^6.0` - Configuration parsing

Install all dependencies with:
```bash
poetry install
```

## Reproducibility

All analyses are fully reproducible with:
- Fixed random seeds (`--seed 42`)
- Deterministic data splits
- Documented preprocessing steps
- Version-controlled configuration

## Contact and Support

For questions about the implementation:
- See inline code documentation
- Review `streaming_config.yaml` for parameters
- Check logs in `results/streaming_growth/`

---

**Time Investment**: ~2-3 hours for pipeline development and initial runs  
**Focus**: Clean, modular code with proper time-series handling  
**Goal**: Interpretable insights into streaming growth drivers
