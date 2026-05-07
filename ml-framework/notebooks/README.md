# ML Framework - Notebooks

This directory contains Jupyter notebooks demonstrating the ML Framework for analyzing large datasets with mixed data types.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- Poetry (for dependency management)

### Installation & Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/izabelcavassim/ml-framework.git
cd ml-framework
```

#### 2. Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or using Homebrew (macOS):
```bash
brew install poetry
```

For other installation methods, see [Poetry Documentation](https://python-poetry.org/docs/#installation).

#### 3. Install Dependencies

```bash
poetry install
```

This will:
- Create a virtual environment
- Install all required dependencies from `pyproject.toml`
- Install development dependencies (pytest, black, flake8, mypy)

#### 4. Activate the Virtual Environment

```bash
poetry shell
```

Or run commands with `poetry run`:
```bash
poetry run jupyter notebook
```

### Running the Notebooks

#### Option A: Using Poetry Shell (Recommended)

```bash
# Activate the virtual environment
poetry shell

# Navigate to notebooks directory
cd notebooks

# Start Jupyter
jupyter notebook
```

#### Option B: Using Poetry Run

```bash
poetry run jupyter notebook notebooks/
```

### Available Notebooks

#### `example_usage.ipynb`
Comprehensive demonstration of the ML Framework including:
- Data loading and preprocessing
- Feature engineering and transformation
- Model training with multiple algorithms
- Model evaluation and comparison
- Visualization of results (confusion matrix, feature importance, etc.)
- Cross-validation analysis

**To run:**
1. Open `example_usage.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook automatically handles project path setup

#### `dblp_acm_entity_matching.ipynb`
Advanced example demonstrating entity matching/record linkage:
- DBLP-ACM dataset loading and exploration
- String similarity metrics (Levenshtein, Jaccard, Jaro-Winkler, Cosine)
- Pair-wise feature extraction
- Entity matching model training
- Performance evaluation

**To run:**
1. Open `dblp_acm_entity_matching.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook automatically handles project path setup

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:** The notebooks include automatic project root detection. If you still encounter this error:

1. Ensure you're running the notebook from within the project directory
2. Try restarting the Jupyter kernel (Kernel → Restart in Jupyter menu)
3. Verify Poetry environment is activated: `poetry shell`

### Issue: `ImportError: cannot import name 'print_section'`

**Solution:** Clear the module cache by restarting the Jupyter kernel:
1. In Jupyter, go to Kernel → Restart
2. Re-run the setup cells

### Issue: Missing dependencies (e.g., `ModuleNotFoundError: No module named 'seaborn'`)

**Solution:** Reinstall dependencies:
```bash
poetry install
```

### Issue: Jupyter not found

**Solution:** Install Jupyter through Poetry:
```bash
poetry add --group dev jupyter
poetry install
```

## Project Structure

```
ml-framework/
├── src/                          # Main framework code
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── data.py                  # Data loading and preprocessing
│   ├── entity_matching.py        # Entity matching utilities
│   ├── features.py              # Feature engineering
│   ├── model.py                 # Model definitions
│   ├── train.py                 # Training utilities
│   └── utils.py                 # Utility functions (including print_section)
├── notebooks/                    # Jupyter notebooks
│   ├── README.md                # This file
│   ├── example_usage.ipynb      # Basic usage example
│   └── dblp_acm_entity_matching.ipynb  # Entity matching example
├── scripts/                      # Utility scripts
│   ├── dblp_acm_data_prep.py    # Data preparation utilities
│   └── entity_matching_pipeline.py  # CLI pipeline
├── tests/                        # Unit tests
│   └── test_entity_matching.py  # Entity matching tests
├── pyproject.toml               # Poetry configuration
├── README.md                    # Main project README
└── ENTITY_MATCHING.md           # Entity matching documentation
```

## Key Features

### Data Layer
- Support for multiple data formats (CSV, JSON, Parquet)
- Automatic data type detection
- Handling of mixed data types (numerical, categorical, text)

### Feature Layer
- Composable feature transformers
- Scaling, encoding, and dimensionality reduction
- Text similarity metrics and embeddings
- Pair-wise feature extraction for entity matching

### Model Layer
- Multiple model implementations (Logistic Regression, Random Forest, SVM, etc.)
- Hyperparameter optimization with Optuna
- Cross-validation strategies

### Training Layer
- Flexible training pipeline
- Model evaluation and comparison
- Performance profiling

## Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **torch** - Deep learning
- **matplotlib** & **seaborn** - Visualization
- **pydantic** - Data validation
- **pyyaml** - Configuration files

## Usage Examples

### Basic Data Loading and Training

```python
from src.data import DataLoader
from src.features import create_default_transformer
from src.model import ModelRegistry
from src.train import Trainer

# Load data
loader = DataLoader()
X_train, X_test, y_train, y_test = loader.load_and_split('data.csv')

# Transform features
transformer = create_default_transformer()
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Train model
model = ModelRegistry.get_model('random_forest')
trainer = Trainer(model)
trainer.fit(X_train_transformed, y_train)

# Evaluate
results = trainer.evaluate(X_test_transformed, y_test)
print(results)
```

### Pretty Output Formatting

```python
from src.utils import print_section

print_section("Model Results", {
    "Accuracy": 0.95,
    "Precision": 0.92,
    "Recall": 0.88,
    "F1-Score": 0.90
})
```

## Documentation

For more detailed information:
- See `README.md` in the project root for framework overview
- See `ENTITY_MATCHING.md` for entity matching documentation
- See `DEVELOPMENT.md` for development guidelines

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the notebook comments and docstrings
3. Open an issue on [GitHub](https://github.com/izabelcavassim/ml-framework/issues)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Izabel Cavassim - [GitHub](https://github.com/izabelcavassim)
