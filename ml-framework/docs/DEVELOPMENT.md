# ML Framework - Development Guide

## Project Overview

The ML Framework is a production-ready machine learning framework designed for analyzing large datasets with mixed data types. It provides a unified interface for both scikit-learn and PyTorch models.

## Architecture

### Core Components

1. **Data Layer** (`src/data.py`)
   - DataLoader: Handles data loading, preprocessing, and batching
   - Supports multiple formats: CSV, Parquet, HDF5
   - Features: missing value handling, outlier detection, data profiling

2. **Feature Layer** (`src/features.py`)
   - FeatureTransformer: Composable transformation pipeline
   - Transformers: Scaling, Encoding, Dimensionality Reduction
   - Automatic feature type detection

3. **Model Layer** (`src/model.py`)
   - BaseModel: Abstract base class for all models
   - ScikitLearnModel: Wrapper for scikit-learn models
   - PyTorchModel: Wrapper for PyTorch models
   - ModelRegistry: Factory for creating models

4. **Training Layer** (`src/train.py`)
   - Trainer: Handles training, validation, and evaluation
   - Features: cross-validation, metrics tracking, result persistence

5. **Configuration Layer** (`src/config.py`)
   - ConfigManager: Centralized configuration management
   - Support for YAML/JSON files
   - Environment-aware settings

6. **Utilities** (`src/utils.py`)
   - Logging setup and management
   - Performance profiling
   - Execution time tracking

## Development Setup

### Prerequisites

- Python 3.9+
- Poetry (install from https://python-poetry.org/)
- Git

### Installation for Development

```bash
# Clone repository
git clone <repository-url>
cd izabels_project

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies with Poetry (includes dev dependencies)
poetry install

# Activate virtual environment
poetry shell

# Verify installation
python -c "from src.config import ConfigManager; print('Setup successful')"
```

### Using Poetry Commands

```bash
# Run Python scripts
poetry run python main.py --help

# Run tests
poetry run pytest tests/ -v

# Format code
poetry run black src/ tests/ main.py

# Type checking
poetry run mypy src/

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Export requirements.txt (optional)
poetry export -f requirements.txt --output requirements.txt
```

## Code Style & Standards

### Python Style Guide

- Follow PEP 8 guidelines
- Use type hints for all function parameters and returns
- Maximum line length: 100 characters
- Use meaningful variable names

### Formatting

```bash
# Format code with black
black src/ tests/ main.py

# Check style with flake8
flake8 src/ tests/ main.py

# Type checking with mypy
mypy src/
```

### Documentation

- All classes and functions must have docstrings
- Use Google-style docstrings
- Include type hints in docstrings
- Provide usage examples for complex functions

Example:
```python
def load_data(path: str, format: str = 'csv') -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        path: Path to data file
        format: File format (csv, parquet, hdf5)
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported
    
    Example:
        >>> df = load_data('data.csv')
        >>> df.shape
        (1000, 10)
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_unit.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_unit.py::TestDataLoader::test_load_from_dataframe -v
```

### Writing Tests

- Use pytest framework
- Create fixtures for reusable test data
- Test both success and failure cases
- Use descriptive test names
- Aim for >80% code coverage

Example:
```python
import pytest
from src.data import DataLoader

class TestDataLoader:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    def test_load_from_dataframe(self, sample_data):
        """Test loading data from DataFrame."""
        loader = DataLoader()
        loader.load_from_dataframe(sample_data)
        assert loader.data is not None
```

## Adding New Features

### Adding a New Model

1. Create model class inheriting from BaseModel
2. Implement required methods: fit, predict, evaluate, save, load
3. Add to ModelRegistry.create()
4. Write unit tests
5. Update documentation

Example:
```python
from src.model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, name: str = "custom_model"):
        super().__init__(name, "custom")
        self.model = None
    
    def fit(self, X, y, **kwargs):
        # Implementation
        self.is_trained = True
        return self
    
    def predict(self, X):
        # Implementation
        return predictions
    
    def evaluate(self, X, y):
        # Implementation
        return metrics
    
    def save(self, path):
        # Implementation
        pass
    
    def load(self, path):
        # Implementation
        pass
```

### Adding a New Transformer

1. Create transformer class inheriting from Transformer
2. Implement fit() and transform() methods
3. Add to FeatureTransformer
4. Write unit tests
5. Update documentation

Example:
```python
from src.features import Transformer

class CustomTransformer(Transformer):
    def fit(self, X):
        # Learn from data
        return self
    
    def transform(self, X):
        # Apply transformation
        return X_transformed
```

### Adding a New Configuration Option

1. Add field to appropriate dataclass in config.py
2. Update ConfigManager if needed
3. Update example config file
4. Update documentation
5. Add tests

## Common Development Tasks

### Running the CLI

```bash
# Basic usage
python main.py --data data.csv --model random_forest

# With all options
python main.py --data data.csv \
    --model random_forest \
    --output results/ \
    --test-size 0.2 \
    --cv 5 \
    --scale standard \
    --encode label \
    --log-level DEBUG
```

### Debugging

```python
# Enable debug logging
from src.utils import setup_logging
logger = setup_logging("ml_framework", level=logging.DEBUG)

# Use breakpoints
import pdb; pdb.set_trace()

# Print intermediate values
print(f"Shape: {X.shape}, Type: {type(X)}")
```

### Performance Profiling

```python
from src.utils import PerformanceProfiler

with PerformanceProfiler("My Operation"):
    # Code to profile
    result = expensive_operation()
```

## Git Workflow

### Branch Naming

- Feature: `feature/description`
- Bug fix: `bugfix/description`
- Documentation: `docs/description`

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Be descriptive: "Add cross-validation support" not "Update code"
- Reference issues: "Fix #123: Add cross-validation support"

### Pull Request Process

1. Create feature branch
2. Make changes and commit
3. Write/update tests
4. Update documentation
5. Submit PR with description
6. Address review comments
7. Merge when approved

## Documentation

### Updating README

- Keep API reference up to date
- Add examples for new features
- Update troubleshooting section
- Maintain table of contents

### Adding Examples

- Create Jupyter notebooks in `notebooks/`
- Include step-by-step explanations
- Show both success and error cases
- Provide expected output

### API Documentation

- Update docstrings when changing signatures
- Keep type hints current
- Document exceptions
- Provide usage examples

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = expensive_operation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Optimization

- Use generators for large datasets
- Avoid unnecessary copies
- Use appropriate data types
- Profile memory usage

### Speed Optimization

- Use vectorized operations (numpy, pandas)
- Parallelize where possible
- Cache expensive computations
- Profile before optimizing

## Troubleshooting Development Issues

### Import Errors

```bash
# Reinstall package in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -vv

# Run with print statements
pytest tests/ -s

# Run specific test with debugging
pytest tests/test_unit.py::TestClass::test_method -vv --pdb
```

### Dependency Issues

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check for conflicts
pip check

# Create fresh environment
python -m venv venv_fresh
source venv_fresh/bin/activate
pip install -r requirements.txt
```

## Release Process

1. Update version in `src/__init__.py`
2. Update CHANGELOG
3. Update README with new features
4. Run full test suite
5. Create git tag
6. Build distribution: `python -m build`
7. Upload to PyPI: `twine upload dist/*`

## Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Getting Help

- Check existing issues on GitHub
- Review documentation and examples
- Ask in project discussions
- Create detailed issue reports

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Type hints are present
- [ ] Tests are written and passing
- [ ] Coverage is >80%
- [ ] Documentation is updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages are clear
- [ ] No debug code left in
- [ ] Performance is acceptable

## Future Development Areas

- GPU support for PyTorch models
- Distributed training with Dask
- AutoML capabilities
- Model interpretability tools
- Real-time prediction serving
- Advanced hyperparameter tuning
- Time series support
- NLP preprocessing utilities
- Web API for model serving
- Docker containerization

## Contact & Support

For questions or issues:
1. Check documentation
2. Review examples
3. Search existing issues
4. Create new issue with details
5. Contact maintainers
