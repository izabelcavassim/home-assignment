# ML Framework - Quick Reference Guide

## Installation & Setup

**IMPORTANT: You must run `poetry install` before using the framework. This is required to set up the virtual environment and make the `src` module importable.**

```bash
# 1. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Install dependencies with Poetry (REQUIRED - do this first!)
poetry install

# 3. Activate virtual environment (optional - you can also use poetry run)
poetry shell

# 4. Or run commands directly with Poetry (without entering shell)
poetry run python main.py --help
```

### Troubleshooting Setup

If you get `ModuleNotFoundError: No module named 'src'`:
- Make sure you've run `poetry install` in the project root directory
- Verify you're in the correct project directory
- Run `python scripts/verify_setup.py` to diagnose the issue

## Command Line Usage

### Basic Training
```bash
python main.py --data data.csv --model random_forest
```

### With Cross-Validation
```bash
python main.py --data data.csv --model svm --cv 5 --cv-strategy stratified
```

### With Custom Configuration
```bash
python main.py --data data.csv --config config.yaml --model logistic_regression
```

### All Options
```bash
python main.py --help
```

## Python API Quick Start

### 1. Load Data
```python
from src.data import DataLoader

loader = DataLoader(batch_size=32)
loader.load('data.csv')
X_train, X_test, y_train, y_test = loader.split(test_size=0.2)
```

### 2. Prepare Features
```python
from src.features import create_default_transformer

transformer = create_default_transformer(scale_method='standard')
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)
```

### 3. Create Model
```python
from src.model import ModelRegistry

model = ModelRegistry.create('random_forest', n_estimators=100)
```

### 4. Train & Evaluate
```python
from src.train import Trainer

trainer = Trainer(model)
trainer.train(X_train, y_train)
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Common Tasks

### Handle Missing Values
```python
loader.handle_missing_values(strategy='mean')  # mean, median, drop, forward_fill
```

### Remove Outliers
```python
loader.remove_outliers(method='iqr', threshold=1.5)  # iqr or zscore
```

### Cross-Validation
```python
cv_results = trainer.cross_validate(X, y, cv=5, strategy='stratified')
```

### Save/Load Model
```python
trainer.save_model('model.pkl')
trainer.load_model('model.pkl')
```

### Configuration Management
```python
from src.config import ConfigManager

config = ConfigManager()
config.load_from_yaml('config.yaml')
config.update(data_test_size=0.3)
config.save_to_yaml('config_output.yaml')
```

## Available Models

- `logistic_regression` - Fast, interpretable
- `random_forest` - Robust, handles non-linearity
- `svm` - Powerful, good for high-dimensional data
- `gradient_boosting` - High performance, slower training

## Feature Scaling Methods

- `standard` - Zero mean, unit variance (default)
- `minmax` - Scale to [0, 1]
- `robust` - Robust to outliers

## Categorical Encoding Methods

- `label` - Integer encoding (default)
- `onehot` - One-hot encoding

## Cross-Validation Strategies

- `stratified` - Maintains class distribution (default)
- `kfold` - Standard k-fold

## Logging Levels

- `DEBUG` - Detailed information
- `INFO` - General information (default)
- `WARNING` - Warning messages
- `ERROR` - Error messages

## File Formats Supported

- CSV (`.csv`)
- Parquet (`.parquet`)
- HDF5 (`.h5`, `.hdf5`)

## Output Structure

```
results/
├── results_YYYYMMDD_HHMMSS/
│   ├── metrics.json          # Evaluation metrics
│   ├── predictions.json      # Predictions and ground truth
│   └── model_name.pkl        # Saved model
```

## Performance Tips

1. **Batch Size**: Adjust based on available memory
2. **Feature Scaling**: Always scale before training
3. **Parallel Processing**: Use `n_jobs=-1` for scikit-learn
4. **Cross-Validation**: Use stratified for imbalanced data
5. **Feature Selection**: Use PCA for high-dimensional data

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce batch size or use streaming |
| Slow Training | Use fewer CV folds or simpler model |
| Poor Performance | Check data quality, try feature engineering |
| Model Not Found | Verify model name in `ModelRegistry.create()` |

## Example Workflows

### Workflow 1: Quick Baseline
```python
from src.data import DataLoader
from src.model import ModelRegistry
from src.train import Trainer

loader = DataLoader()
loader.load('data.csv')
X_train, X_test, y_train, y_test = loader.split()

model = ModelRegistry.create('random_forest')
trainer = Trainer(model)
trainer.train(X_train, y_train)
print(trainer.evaluate(X_test, y_test))
```

### Workflow 2: Full Pipeline
```python
from src.data import DataLoader
from src.features import create_default_transformer
from src.model import ModelRegistry
from src.train import Trainer
from src.config import ConfigManager

# Configuration
config = ConfigManager()
config.load_from_yaml('config.yaml')

# Data
loader = DataLoader(batch_size=config.data.batch_size)
loader.load('data.csv')
loader.handle_missing_values(strategy=config.data.handle_missing)
X_train, X_val, X_test, y_train, y_val, y_test = loader.split(
    test_size=config.data.test_size,
    validation_size=0.1
)

# Features
transformer = create_default_transformer()
X_train = transformer.fit_transform(X_train)
X_val = transformer.transform(X_val)
X_test = transformer.transform(X_test)

# Model
model = ModelRegistry.create(config.model.model_type)
trainer = Trainer(model, output_dir=config.experiment.output_dir)

# Training
trainer.train(X_train, y_train, X_val, y_val)
metrics = trainer.evaluate(X_test, y_test)

# Save
trainer.save_model(f"{config.experiment.output_dir}/model.pkl")
```

### Workflow 3: Model Comparison
```python
models = ['logistic_regression', 'random_forest', 'svm']
results = {}

for model_name in models:
    model = ModelRegistry.create(model_name)
    trainer = Trainer(model)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    results[model_name] = metrics['accuracy']

# Print best model
best_model = max(results, key=results.get)
print(f"Best model: {best_model} ({results[best_model]:.4f})")
```

## Resources

- **Documentation**: See `README.md`
- **Examples**: See `notebooks/example_usage.ipynb`
- **Tests**: See `tests/test_unit.py`
- **Configuration**: See `config.yaml` template

## Getting Help

1. Check the README for detailed documentation
2. Review example notebooks
3. Run tests to verify installation
4. Check logging output for debugging

## Next Steps

1. Prepare your data (CSV, Parquet, or HDF5)
2. Create a configuration file (optional)
3. Run the framework with your data
4. Analyze results in the output directory
5. Iterate and improve your model
