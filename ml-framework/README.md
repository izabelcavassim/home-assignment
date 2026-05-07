# ML Framework - Large Dataset Analysis

A scalable, production-ready machine learning framework for analyzing large datasets with mixed data types. Built with support for both scikit-learn and PyTorch models.

## Features

### 🚀 Core Capabilities

- **Efficient Data Loading**: Stream and batch large datasets (CSV, Parquet, HDF5)
- **Mixed Data Types**: Automatic detection and preprocessing of numerical and categorical features
- **Feature Engineering**: Composable transformation pipeline with scaling, encoding, and dimensionality reduction
- **Unified Model Interface**: Consistent API for scikit-learn and PyTorch models
- **Comprehensive Training**: Built-in validation, early stopping, and cross-validation
- **Metrics & Evaluation**: Extensive metrics tracking and result visualization
- **Configuration Management**: YAML/JSON-based configuration with environment support
- **Experiment Tracking**: Automatic logging of models, predictions, and metrics

### 📊 Supported Models

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- PyTorch Neural Networks (extensible)

### 🔧 Data Processing

- **Missing Value Handling**: mean, median, forward fill, backward fill, drop
- **Outlier Detection**: IQR and Z-score methods
- **Feature Scaling**: Standard, MinMax, Robust scaling
- **Categorical Encoding**: Label encoding and one-hot encoding
- **Dimensionality Reduction**: PCA support

## Installation

### Prerequisites

- Python 3.9+
- Poetry (install from https://python-poetry.org/)

### Setup

**⚠️ IMPORTANT: Run `poetry install` as the first step after cloning the repository. This is required to set up the virtual environment and make the framework importable.**

1. Clone the repository:
```bash
git clone <repository-url>
cd izabels_project
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Install dependencies using Poetry (REQUIRED):**
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

Or run commands with Poetry directly:
```bash
poetry run python main.py --help
```

### Troubleshooting Installation

**If you get `ModuleNotFoundError: No module named 'src'`:**
- Ensure you've run `poetry install` in the project root
- Verify you're in the correct directory
- Run `python scripts/verify_setup.py` to diagnose setup issues
- See [POETRY_SETUP.md](docs/POETRY_SETUP.md) for detailed troubleshooting

## Quick Start

### Command Line Usage

Train a model on your data:

```bash
# Basic usage
python main.py --data data.csv --model random_forest --output results/

# With cross-validation
python main.py --data data.csv --model svm --cv 5 --cv-strategy stratified

# With custom configuration
python main.py --data data.csv --config config.yaml --model logistic_regression

# Full options
python main.py --help
```

### Python API

```python
from src.data import DataLoader
from src.features import create_default_transformer
from src.model import ModelRegistry
from src.train import Trainer

# Load data
data_loader = DataLoader(batch_size=32)
data_loader.load('data.csv')

# Split data
X_train, X_test, y_train, y_test = data_loader.split(test_size=0.2)

# Create feature transformer
transformer = create_default_transformer(scale_method='standard')

# Create and train model
model = ModelRegistry.create('random_forest', n_estimators=100)
trainer = Trainer(model, feature_transformer=transformer)
trainer.train(X_train, y_train)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Project Structure

```
izabels_project/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── README.md              # This file
├── .gitignore             # Git ignore rules
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── data.py            # Data loading and preprocessing
│   ├── features.py        # Feature engineering pipeline
│   ├── model.py           # Model definitions and wrappers
│   ├── train.py           # Training engine
│   └── utils.py           # Utilities and logging
│
├── tests/                 # Test suite
│   ├── test_unit.py       # Unit tests
│   └── test_e2e.py        # End-to-end tests
│
├── notebooks/             # Jupyter notebooks
│   └── example_usage.ipynb # Example usage notebook
│
├── data/                  # Data directory (not tracked)
├── results/               # Results and outputs (not tracked)
└── models/                # Saved models (not tracked)
```

## Configuration

### YAML Configuration Example

Create a `config.yaml` file:

```yaml
data:
  test_size: 0.2
  random_state: 42
  batch_size: 32
  normalize: true
  handle_missing: mean

model:
  model_type: random_forest
  random_state: 42
  n_jobs: -1

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  validation_split: 0.2

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
  cv_folds: 5
  cv_strategy: stratified

experiment:
  name: my_experiment
  save_model: true
  save_predictions: true
  output_dir: results
```

Then use it:

```bash
python main.py --data data.csv --config config.yaml
```

## API Reference

### DataLoader

```python
from src.data import DataLoader

# Initialize
loader = DataLoader(batch_size=32, random_state=42)

# Load data
loader.load('data.csv')  # Auto-detects format
loader.load('data.parquet', file_format='parquet')

# Or from DataFrame
loader.load_from_dataframe(df)

# Preprocessing
loader.handle_missing_values(strategy='mean')
loader.remove_outliers(method='iqr', threshold=1.5)

# Split data
X_train, X_test, y_train, y_test = loader.split(test_size=0.2)

# Or with validation set
X_train, X_val, X_test, y_train, y_val, y_test = loader.split(
    test_size=0.2, validation_size=0.1
)

# Get batches
for X_batch, y_batch in loader.get_batches(X, y, shuffle=True):
    # Process batch
    pass

# Get metadata
metadata = loader.get_metadata()
```

### FeatureTransformer

```python
from src.features import FeatureTransformer, create_default_transformer

# Create default pipeline
transformer = create_default_transformer(
    scale_method='standard',
    encode_method='label'
)

# Or build custom pipeline
transformer = FeatureTransformer()
transformer.add_scaling('standard')
transformer.add_encoding('label')
transformer.add_dimensionality_reduction('pca', n_components=10)

# Fit and transform
X_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Get feature importance
importance = transformer.get_feature_importance()
```

### ModelRegistry

```python
from src.model import ModelRegistry

# Create models
model = ModelRegistry.create('random_forest', n_estimators=100, random_state=42)
model = ModelRegistry.create('logistic_regression', max_iter=1000)
model = ModelRegistry.create('svm', kernel='rbf')
model = ModelRegistry.create('gradient_boosting', n_estimators=100)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test)

# Save/Load
model.save('model.pkl')
model.load('model.pkl')
```

### Trainer

```python
from src.train import Trainer

# Initialize
trainer = Trainer(model, feature_transformer=transformer, output_dir='results')

# Train
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = trainer.evaluate(X_test, y_test, save_results=True)

# Cross-validate
cv_results = trainer.cross_validate(X, y, cv=5, strategy='stratified')

# Save/Load
trainer.save_model('model.pkl')
trainer.load_model('model.pkl')

# Get history
history = trainer.get_training_history()
```

### ConfigManager

```python
from src.config import ConfigManager

# Initialize
config = ConfigManager(env='production')

# Load from file
config.load_from_yaml('config.yaml')
config.load_from_json('config.json')

# Update programmatically
config.update(
    data_test_size=0.3,
    model_type='svm',
    training_epochs=50
)

# Save configuration
config.save_to_yaml('config_output.yaml')
config.save_to_json('config_output.json')

# Access values
print(config.data.test_size)
print(config.model.model_type)
```

## Examples

### Example 1: Basic Classification

```python
from src.data import DataLoader
from src.model import ModelRegistry
from src.train import Trainer

# Load and prepare data
loader = DataLoader()
loader.load('iris.csv')
X_train, X_test, y_train, y_test = loader.split(test_size=0.2)

# Train model
model = ModelRegistry.create('random_forest')
trainer = Trainer(model)
trainer.train(X_train, y_train)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Example 2: With Feature Engineering

```python
from src.features import FeatureTransformer

# Create transformer
transformer = FeatureTransformer()
transformer.add_scaling('standard')
transformer.add_encoding('label')

# Apply transformations
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Train with transformed data
trainer = Trainer(model, feature_transformer=transformer)
trainer.train(X_train_transformed, y_train)
```

### Example 3: Cross-Validation

```python
# Perform 5-fold cross-validation
cv_results = trainer.cross_validate(
    X, y,
    cv=5,
    strategy='stratified'
)

# Print results
for metric in cv_results['test_scores']:
    score = cv_results['test_scores'][metric]
    print(f"{metric}: {score['mean']:.4f} (+/- {score['std']:.4f})")
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_unit.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Jupyter Notebooks

Start Jupyter and explore the example notebook:

```bash
jupyter notebook notebooks/example_usage.ipynb
```

The notebook demonstrates:
- Data loading and exploration
- Data preprocessing
- Feature engineering
- Model training and evaluation
- Cross-validation
- Model comparison
- Configuration management

## Performance Tips

1. **Batch Processing**: Use appropriate batch sizes for your hardware
2. **Feature Scaling**: Always scale features before training
3. **Cross-Validation**: Use stratified k-fold for imbalanced datasets
4. **Parallel Processing**: Set `n_jobs=-1` for scikit-learn models
5. **Memory Management**: Use DataLoader for large datasets to avoid loading everything into memory

## Troubleshooting

### Out of Memory Error

- Reduce batch size: `DataLoader(batch_size=16)`
- Use streaming for large files
- Process data in chunks

### Slow Training

- Reduce number of features with PCA
- Use fewer cross-validation folds
- Increase `n_jobs` for parallel processing
- Use a simpler model (e.g., Logistic Regression instead of SVM)

### Poor Model Performance

- Check data quality and handle missing values
- Try different feature scaling methods
- Perform feature engineering
- Tune hyperparameters with cross-validation
- Try different models

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ml_framework_2024,
  title={ML Framework: Large Dataset Analysis},
  author={Izabel},
  year={2024},
  url={https://github.com/example/ml-framework}
}
```

## Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check existing documentation
- Review example notebooks

## Roadmap

- [ ] GPU support for PyTorch models
- [ ] Distributed training with Dask
- [ ] AutoML capabilities
- [ ] Model interpretability tools
- [ ] Real-time prediction serving
- [ ] Advanced hyperparameter tuning
- [ ] Time series support
- [ ] NLP preprocessing utilities

## Changelog

### Version 0.1.0 (Initial Release)

- Core framework implementation
- Data loading and preprocessing
- Feature engineering pipeline
- Model training and evaluation
- Configuration management
- Comprehensive documentation
- Example notebooks and tests
