"""
Model Training Module
====================

Handles model training, evaluation, and comparison.
"""

import polars as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import logging

from prediction.feature_engineering import create_features
from prediction.ml_models import get_ml_model
from prediction.dl_models import get_dl_model

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, output_dir: str = 'models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        self.results = {}
        self.trained_models = {}
    
    def prepare_data(self, factors: pl.DataFrame) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training."""
        logging.info("Preparing data for training...")
        
        # Engineer features
        features = create_features(factors)
        features = features.drop_nulls()
        
        # Split features and target
        feature_cols = [col for col in features.columns 
                       if col not in ['Symbol', 'Dates', 'target_return_1m'] and features[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        X = features.select(feature_cols).to_numpy()
        y = features.select('target_return_1m').to_numpy().flatten()
        
        logging.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logging.info(f"Number of features: {len(feature_cols)}")
        logging.info(f"Features: {feature_cols}")
        return features, X, y, feature_cols

    def train_single_model(self, model_name: str, features: pl.DataFrame, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train a single model with cross-validation."""
        logging.info(f"  Training {model_name}...")
        
        model_type = 'dl' if model_name in ['dnn', 'residual', 'transformer'] else 'ml'
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_dates = features.item(train_idx[0], 'Dates'), features.item(train_idx[-1], 'Dates')
            val_dates = features.item(val_idx[0], 'Dates'), features.item(val_idx[-1], 'Dates')
            logging.info(f"    Fold {fold+1}: Train period: {train_dates[0]} to {train_dates[1]}, Test period: {val_dates[0]} to {val_dates[1]}")
            
            if model_type == 'ml':
                model = get_ml_model(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
            else:  # Deep learning
                import torch
                import torch.nn as nn
                import torch.optim as optim
                from torch.utils.data import TensorDataset, DataLoader
                
                model = get_dl_model(model_name, X_train.shape[1])
                
                # Prepare data
                train_dataset = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                # Training
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                model.train()
                for epoch in range(10):
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y.view(-1, 1))
                        loss.backward()
                        optimizer.step()
                
                # Prediction
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.tensor(X_val, dtype=torch.float32)).numpy().flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            cv_scores.append({
                'fold': fold,
                'mse': mse,
                'mae': mae,
                'r2': r2
            })
        
        # Store results
        results = {
            'cv_scores': cv_scores,
            'mean_mse': np.mean([s['mse'] for s in cv_scores]),
            'mean_mae': np.mean([s['mae'] for s in cv_scores]),
            'mean_r2': np.mean([s['r2'] for s in cv_scores])
        }
        
        logging.info(f"    {model_name} - MSE: {results['mean_mse']:.6f}, "
              f"MAE: {results['mean_mae']:.6f}, R²: {results['mean_r2']:.4f}")
        
        return results
    
    def train_models(self, factors: pl.DataFrame, models: List[str]):
        """Train multiple models."""
        logging.info("Training models...")
        
        # Prepare data
        features, X, y, feature_cols = self.prepare_data(factors)
        # print(X)
        # Train each model
        for model_name in models:
            self.results[model_name] = self.train_single_model(model_name, features, X, y)
        
        logging.info("All models trained successfully")
    
    def evaluate_models(self):
        """Evaluate and compare model performance."""
        logging.info("Evaluating models...")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'MSE': results['mean_mse'],
                'MAE': results['mean_mae'],
                'R²': results['mean_r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save results
        comparison_df.to_csv(os.path.join(self.output_dir, 'results', 'model_comparison.csv'), index=False)
        
        # Create visualization
        self._create_comparison_plot(comparison_df)
        
        logging.info("Model evaluation completed")
        return comparison_df
    
    def _create_comparison_plot(self, comparison_df: pd.DataFrame):
        """Create comparison plots."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MSE comparison
        axes[0].bar(comparison_df['Model'], comparison_df['MSE'])
        axes[0].set_title('Mean Squared Error')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[2].bar(comparison_df['Model'], comparison_df['R²'])
        axes[2].set_title('R² Score')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close() 