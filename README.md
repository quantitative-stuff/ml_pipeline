# Stock Return Prediction ML Pipeline

A modular and efficient ML pipeline for predicting 1-month stock returns using multiple machine learning and deep learning models.

## 🏗️ Architecture

The pipeline is now organized into clean, modular components:

```
ml_pipeline/
├── pipeline/                    # Modular pipeline components
│   ├── __init__.py
│   ├── data_processor.py       # Data extraction and preprocessing
│   ├── factor_processor.py     # Factor computation and target creation
│   └── model_trainer.py        # Model training and evaluation
├── factor_creation/            # Factor computation logic
├── prediction/                 # ML/DL models and feature engineering
├── data/                       # Extracted data storage
├── factors/                    # Computed factors storage
├── models/                     # Model results and plots
├── config.py                   # Centralized configuration
├── main_pipeline.py           # Main pipeline orchestrator
└── run_pipeline.py            # Simple runner interface
```

## 🚀 Quick Start

### Option 1: Simple Runner (Recommended)
```bash
# Quick run with default settings
python run_pipeline.py --quick

# Full run with all models
python run_pipeline.py --full

# Custom run
python run_pipeline.py --start_date 2020-01-01 --end_date 2024-12-31 --models linear_regression,gradient_boosting,dnn
```

### Option 2: Direct Pipeline
```bash
python main_pipeline.py --start_date 2020-01-01 --end_date 2024-12-31 --models linear_regression,gradient_boosting,dnn
```

## 📋 Pipeline Steps

1. **Data Processing** (`pipeline/data_processor.py`)
   - Extract data from database
   - Convert to Polars for efficiency
   - Save/load data for reuse

2. **Factor Processing** (`pipeline/factor_processor.py`)
   - Compute financial factors
   - Create 1-month forward return target
   - Standardize factors

3. **Model Training** (`pipeline/model_trainer.py`)
   - Feature engineering
   - Train multiple ML/DL models
   - Cross-validation and evaluation

## 🎯 Available Models

### Machine Learning Models
- `linear_regression`: Linear regression
- `gradient_boosting`: Gradient boosting regressor
- `random_forest`: Random forest regressor
- `svr`: Support vector regression

### Deep Learning Models
- `dnn`: Deep neural network
- `residual`: Residual network
- `transformer`: Transformer model

## ⚙️ Configuration

All settings are centralized in `config.py`:

```python
# Database settings
DB_CONFIG = {...}

# Pipeline defaults
PIPELINE_CONFIG = {
    'default_start_date': '2020-01-01',
    'default_end_date': '2024-12-31',
    'default_models': ['linear_regression', 'gradient_boosting', 'dnn']
}

# Training settings
TRAINING_CONFIG = {
    'cv_splits': 5,
    'dl_epochs': 10,
    'dl_batch_size': 32
}
```

## 📊 Output

The pipeline generates:

- **Data**: `data/` - Extracted database data
- **Factors**: `factors/` - Computed financial factors
- **Results**: `models/results/` - Model performance metrics
- **Plots**: `models/plots/` - Performance comparison visualizations

## 🔧 Key Improvements

### Before (Complex)
- Single massive class with 400+ lines
- All logic mixed together
- Hard to maintain and debug
- No reusability

### After (Modular)
- Clean separation of concerns
- Reusable components
- Easy to test and modify
- Simple configuration
- Multiple usage interfaces

## 📈 Usage Examples

```python
# Quick test run
python run_pipeline.py --quick

# Full production run
python run_pipeline.py --full

# Custom configuration
python run_pipeline.py --start_date 2023-01-01 --end_date 2024-12-31 --models dnn,transformer

# Using the main pipeline directly
python main_pipeline.py --models linear_regression,gradient_boosting
```

## 🛠️ Development

### Adding New Models
1. Add model to `prediction/ml_models.py` or `prediction/dl_models.py`
2. Update `config.py` with new model name
3. Use in pipeline

### Adding New Factors
1. Add factor computation to `factor_creation/factor_pipeline.py`
2. Update `factor_library.py` with factor description
3. Factor will be automatically included

### Modifying Pipeline Steps
Each step is isolated in its own module:
- `pipeline/data_processor.py` - Data handling
- `pipeline/factor_processor.py` - Factor computation
- `pipeline/model_trainer.py` - Model training

## 📝 Requirements

```
pandas
polars
numpy
scikit-learn
torch
matplotlib
seaborn
sqlalchemy
pymysql
```

## 🎉 Benefits

- **Modular**: Each component has a single responsibility
- **Reusable**: Components can be used independently
- **Configurable**: All settings in one place
- **Efficient**: Uses Polars for fast data processing
- **Scalable**: Easy to add new models and factors
- **Maintainable**: Clean, readable code structure 