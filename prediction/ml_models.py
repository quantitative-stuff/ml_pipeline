
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def get_ml_model(model_name: str):
    """
    Returns a machine learning model based on the given name.

    Args:
        model_name: The name of the model to return.

    Returns:
        A scikit-learn model.
    """
    if model_name == 'linear_regression':
        return LinearRegression()
    elif model_name == 'gradient_boosting':
        return GradientBoostingRegressor()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
