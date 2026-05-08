import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'n_samples': len(y_true)
    }


def calculate_cv_metrics(cv_scores) -> tuple:
    """Calculate cross-validation metrics."""
    return (
        np.mean(cv_scores['r2']),
        np.std(cv_scores['r2']),
        np.mean(cv_scores['rmse']),
        np.std(cv_scores['rmse']),
        np.mean(cv_scores['mae']),
        np.std(cv_scores['mae'])
    )