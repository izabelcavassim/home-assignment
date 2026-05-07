# Poetry Setup Guide

**⚠️ CRITICAL: You must run `poetry install` before using the framework. This is the first step after cloning the repository.**

This project uses **Poetry** for dependency management and packaging. Poetry provides a modern, reliable way to manage Python project dependencies.

## Why Poetry?

- **Deterministic builds**: Lock file ensures reproducible environments
- **Dependency resolution**: Automatically resolves complex dependency trees
- **Virtual environment management**: Automatic virtual environment creation
- **Easy publishing**: Simplified package publishing to PyPI
- **Modern standards**: Uses PEP 517/518 build standards

## Installation

### Install Poetry

```bash
# macOS / Linux / WSL
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Or using pip
pip install poetry
```

### Verify Installation

```bash
poetry --version
```

## Quick Start

### 1. Install Project Dependencies

```bash
# Install all dependencies (including dev)
poetry install

# Install only production dependencies
poetry install --no-dev
```

### 2. Activate Virtual Environment

```bash
# Enter the virtual environment
poetry shell

# Or run commands directly without entering shell
poetry run python main.py --help
```

### 3. Run Commands

```bash
# Run Python scripts
poetry run python main.py --data data.csv --model random_forest

# Run tests
poetry run pytest tests/ -v

# Format code
poetry run black src/ tests/

# Type checking
poetry run mypy src/
```

## Common Poetry Commands

### Dependency Management

```bash
# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update all dependencies
poetry update

# Update specific dependency
poetry update package-name

# Remove a dependency
poetry remove package-name

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree
```

### Virtual Environment

```bash
# Activate virtual environment
poetry shell

# Exit virtual environment
exit

# Show virtual environment info
poetry env info

# List all virtual environments
poetry env list

# Remove virtual environment
poetry env remove python3.9
```

### Lock File

```bash
# Generate/update lock file
poetry lock

# Lock without updating
poetry lock --no-update

# Export lock file to requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

### Building & Publishing

```bash
# Build distribution packages
poetry build

# Publish to PyPI
poetry publish

# Publish to private repository
poetry publish -r my-repo
```

## Project Structure

```
pyproject.toml          # Poetry configuration file
poetry.lock             # Lock file (auto-generated)
src/                    # Source code
tests/                  # Tests
notebooks/              # Jupyter notebooks
```

## pyproject.toml Structure

The `pyproject.toml` file contains:

```toml
[tool.poetry]
name = "ml-framework"
version = "0.1.0"
description = "..."
authors = ["..."]

[tool.poetry.dependencies]
python = "^3.9"
pandas = "^2.0.0"
# ... more dependencies

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
# ... dev dependencies

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## Version Specifiers

Poetry uses semantic versioning:

```
^1.2.3   # Compatible with 1.2.3 (>=1.2.3, <2.0.0)
~1.2.3   # Approximately 1.2.3 (>=1.2.3, <1.3.0)
==1.2.3  # Exact version
>=1.2.3  # Greater than or equal
<=1.2.3  # Less than or equal
```

## Troubleshooting

### ModuleNotFoundError: No module named 'src'

This error occurs when the project hasn't been installed with Poetry. **Solution:**

```bash
# Make sure you're in the project root directory
cd /path/to/ml-framework

# Install the project and all dependencies
poetry install

# Verify the installation
python scripts/verify_setup.py
```

### Poetry Command Not Found

```bash
# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or use full path
~/.local/bin/poetry --version
```

### Virtual Environment Issues

```bash
# Remove and recreate virtual environment
poetry env remove python3.9
poetry install

# Use specific Python version
poetry env use python3.10
```

### Dependency Conflicts

```bash
# Clear cache and reinstall
poetry cache clear . --all
poetry install
```

### Lock File Issues

```bash
# Regenerate lock file
poetry lock --no-update
poetry install
```

## Development Workflow

### 1. Clone Repository

```bash
git clone <repository-url>
cd izabels_project
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Activate Environment

```bash
poetry shell
```

### 4. Make Changes

```bash
# Edit code
# Run tests
poetry run pytest tests/ -v

# Format code
poetry run black src/

# Type check
poetry run mypy src/
```

### 5. Add New Dependencies

```bash
# For production
poetry add new-package

# For development
poetry add --group dev new-package
```

### 6. Update Lock File

```bash
poetry lock
```

### 7. Commit Changes

```bash
git add pyproject.toml poetry.lock
git commit -m "Update dependencies"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - uses: snok/install-poetry@v1
      - run: poetry install
      - run: poetry run pytest
```

## Docker Integration

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-dev

# Copy source code
COPY src/ src/

# Run application
CMD ["poetry", "run", "python", "main.py"]
```

## Migration from pip

### Export to requirements.txt

```bash
poetry export -f requirements.txt --output requirements.txt
```

### Import from requirements.txt

```bash
poetry add $(cat requirements.txt | grep -v '^#' | grep -v '^$')
```

## Resources

- **Official Documentation**: https://python-poetry.org/docs/
- **GitHub**: https://github.com/python-poetry/poetry
- **PyPI**: https://pypi.org/project/poetry/

## Tips & Best Practices

1. **Always commit poetry.lock**: Ensures reproducible builds
2. **Use version constraints**: Use `^` for compatible versions
3. **Separate dev dependencies**: Keep dev tools in separate group
4. **Update regularly**: Run `poetry update` periodically
5. **Use poetry shell**: Easier than `poetry run` for multiple commands
6. **Check for conflicts**: Use `poetry check` to validate pyproject.toml

## Getting Help

```bash
# Show help for any command
poetry help <command>

# Examples
poetry help add
poetry help install
poetry help run
```

---

For more information, visit the [Poetry documentation](https://python-poetry.org/docs/).
