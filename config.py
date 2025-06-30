"""
Configuration file for the ML pipeline.
"""

# Database configuration
DB_CONFIG = {
    'host': '192.168.1.27',
    'port': '3306',
    'db_name': 'quantdb_maria',
    'username': 'quantdb',
    'password': 'QuantDb2023!'
}

# Pipeline configuration
PIPELINE_CONFIG = {
    'default_start_date': '2023-01-01',
    'default_end_date': '2024-12-31',
    'default_models': ['linear_regression', 'gradient_boosting', 'dnn'],
    'available_models': {
        'ml': ['linear_regression', 'gradient_boosting', 'random_forest', 'svr'],
        'dl': ['dnn', 'residual', 'transformer']
    }
}

# Data processing configuration
DATA_CONFIG = {
    'data_dir': 'data',
    'factors_dir': 'factors',
    'models_dir': 'models',
    'output_dir': 'output'
}

# Model training configuration
TRAINING_CONFIG = {
    'cv_splits': 5,
    'random_state': 42,
    'test_size': 0.2,
    'dl_batch_size': 32,
    'dl_epochs': 10,
    'dl_learning_rate': 0.001
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'momentum_windows': [5, 20],
    'volatility_windows': [5, 20],
    'fill_method': 'forward'
} 