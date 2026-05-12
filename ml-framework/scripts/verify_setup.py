#!/usr/bin/env python3
"""
Setup Verification Script

This script verifies that the ML Framework project is properly installed and configured.
Run this if you encounter import errors or other setup issues.

Usage:
    python scripts/verify_setup.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (FAILED)")
        print(f"    Required: Python 3.9+")
        return False


def check_src_module():
    """Check if src module can be imported."""
    print("\nChecking src module import...")
    try:
        import src
        print("  ✓ src module found (OK)")
        return True
    except ModuleNotFoundError:
        print("  ✗ src module not found (FAILED)")
        print("    This means 'poetry install' has not been run or the virtual environment is not active.")
        return False


def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")
    project_root = Path(__file__).parent.parent
    required_dirs = ["src", "scripts", "tests", "docs", "data"]
    required_files = ["pyproject.toml", "README.md"]
    
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {dir_name}/ found")
        else:
            print(f"  ✗ {dir_name}/ not found")
            all_ok = False
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ {file_name} found")
        else:
            print(f"  ✗ {file_name} not found")
            all_ok = False
    
    return all_ok


def check_key_modules():
    """Check if key modules can be imported."""
    print("\nChecking key modules...")
    modules = [
        "src.utils",
        #"src.config",
        #"src.data",
        "src.model",
        "src.generate_html_report",
        "src.html_report",
        "src.metrics" ,
        "src.plotting",
        "src.streaming_data"
        #"src.train",
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name} imported successfully")
        except ImportError as e:
            print(f"  ✗ {module_name} import failed: {e}")
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Check if key dependencies are installed."""
    print("\nChecking dependencies...")
    dependencies = [
        "pandas",
        "numpy",
        "sklearn",
        "torch",
        "pydantic",
        "yaml",
    ]
    
    all_ok = True
    for dep_name in dependencies:
        try:
            __import__(dep_name)
            print(f"  ✓ {dep_name} installed")
        except ImportError:
            print(f"  ✗ {dep_name} not installed")
            all_ok = False
    
    return all_ok


def print_solution():
    """Print solution for common issues."""
    print("\n" + "=" * 70)
    print("SOLUTION")
    print("=" * 70)
    print("\nIf any checks failed, follow these steps:\n")
    print("1. Make sure you're in the project root directory:")
    print("   cd /path/to/ml-framework\n")
    print("2. Install Poetry (if not already installed):")
    print("   curl -sSL https://install.python-poetry.org | python3 -\n")
    print("3. Install the project and all dependencies:")
    print("   poetry install\n")
    print("4. Activate the virtual environment:")
    print("   poetry shell\n")
    print("5. Run this verification script again:")
    print("   python scripts/verify_setup.py\n")
    print("For more help, see:")
    print("  - docs/QUICKSTART.md")
    print("  - docs/POETRY_SETUP.md")
    print("  - README.md")


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("ML Framework Setup Verification")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("src Module", check_src_module),
        ("Key Modules", check_key_modules),
        ("Dependencies", check_dependencies),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"  ✗ Error during check: {e}")
            results[check_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    if all_passed:
        print("\n✓ All checks passed! Your setup is correct.")
        print("You can now run the framework scripts.")
        return 0
    else:
        print("\n✗ Some checks failed. See the solution below:")
        print_solution()
        return 1


if __name__ == "__main__":
    sys.exit(main())
