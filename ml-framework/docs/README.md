# Documentation

This directory contains documentation for the Streaming Growth Analysis project.

## Available Documentation

### [POETRY_SETUP.md](POETRY_SETUP.md)
Complete guide to setting up Poetry for dependency management. Read this first if you're unfamiliar with Poetry or having setup issues.

**Key Topics:**
- Installing Poetry
- Managing dependencies
- Running commands with Poetry
- Troubleshooting common issues

## Main Documentation

For complete project documentation, see the root [README.md](../README.md) which covers:

- **Problem Statement** - What we're analyzing
- **Methodology** - How we approach the analysis
- **Setup Instructions** - Getting started
- **How to Run** - Running the pipeline with various options
- **Results & Interpretation** - Understanding the output
- **Technical Details** - Temporal leakage prevention, feature engineering
- **Future Improvements** - Recommended enhancements

## Quick Start

```bash
# 1. Install dependencies (REQUIRED first step)
poetry install

# 2. Run the streaming growth analysis
poetry run python scripts/streaming_growth_pipeline.py \
  --data-dir data \
  --output results/streaming_growth \
  --model all \
  --cv-folds 5

# 3. Run tests
poetry run pytest tests/test_unit.py -v
```

## Project Focus

This project analyzes **drivers of streaming growth** for artists on Spotify using:
- Time-series data with proper temporal alignment
- Multiple data sources (streaming, social, tickets, demographics)
- Cross-validation with TimeSeriesSplit
- Feature engineering with lagged and rolling window features
- Multiple regression models with comprehensive evaluation

See the main [README.md](../README.md) for full details.
