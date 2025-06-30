"""
Factor Processing Module
=======================

Handles factor computation and target variable creation.
"""

import polars as pl
import numpy as np
from typing import Optional
import os
from datetime import datetime, timedelta

from factor_creation.factor_pipeline import FactorPipeline

class FactorProcessor:
    """Handles factor computation and target creation."""
    
    def __init__(self, output_dir: str = 'factors'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.factor_pipeline = FactorPipeline()
    
    def compute_factors(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute factors from raw data."""
        print("🔄 Computing factors...")
        
        # For simplicity, use same data for historical calculations
        # In practice, you'd need actual historical data
        past1m_data = data
        past3m_data = data
        past12m_data = data
        
        # Compute all factors
        factors = self.factor_pipeline.compute_all_factors(
            data, past1m_data, past3m_data, past12m_data
        )
        
        # Standardize factors
        exclude_cols = ['Symbol', 'Dates']
        factors = self.factor_pipeline.standardize_factors(factors, exclude_cols)
        
        print(f"✅ Factors computed: {factors.shape[1]} features")
        return factors
    
    def create_target(self, factors: pl.DataFrame, raw_data: pl.DataFrame) -> pl.DataFrame:
        """
        Create 1-month forward return target variable using actual price data.
        
        Args:
            factors: DataFrame with computed factors
            raw_data: Original raw data containing price information
        """
        print("🔄 Creating target variable (1-month forward return)...")
        
        # Sort by symbol and date
        factors = factors.sort(['Symbol', 'Dates'])
        raw_data = raw_data.sort(['Symbol', 'Dates'])
        
        # Get the adjusted price column name from factor dictionary
        adj_price_col = self.factor_pipeline.factor_dict['adjPrice']
        
        if adj_price_col not in raw_data.columns:
            raise ValueError(f"❌ Error: {adj_price_col} (adjusted price) not found in data. Cannot calculate forward returns without price data.")
        
        # Calculate 1-month forward returns
        print(f"  Using price column: {adj_price_col}")
        
        # Create a copy of raw_data with price information
        price_data = raw_data.select(['Symbol', 'Dates', adj_price_col])
        
        # Calculate forward returns using Polars
        # Formula: (adj_price[date+1month] / adj_price[date]) - 1
        forward_returns = price_data.with_columns([
            # Shift price forward by 1 month for each symbol
            pl.col(adj_price_col).shift(-1).over('Symbol').alias('future_price'),
            # Current price
            pl.col(adj_price_col).alias('current_price')
        ])
        
        # Calculate return: (future_price / current_price) - 1
        forward_returns = forward_returns.with_columns([
            ((pl.col('future_price') / pl.col('current_price')) - 1).alias('target_return_1m')
        ])
        
        # Select only the columns we need
        forward_returns = forward_returns.select(['Symbol', 'Dates', 'target_return_1m'])
        
        # Join with factors
        factors = factors.join(forward_returns, on=['Symbol', 'Dates'], how='left')
        
        # Remove rows where we don't have future price data (last month of data)
        factors = factors.drop_nulls(subset=['target_return_1m'])
        
        print(f"✅ Target variable created: {len(factors)} samples with valid forward returns")
        print(f"  Return statistics: mean={factors['target_return_1m'].mean():.4f}, "
              f"std={factors['target_return_1m'].std():.4f}")
        print(f"  Return range: {factors['target_return_1m'].min():.4f} to {factors['target_return_1m'].max():.4f}")
        
        return factors
    
    def save_factors(self, factors: pl.DataFrame):
        """Save computed factors."""
        factors.write_parquet(os.path.join(self.output_dir, 'factors.parquet'))
        print(f"✅ Factors saved to {self.output_dir}")
    
    def load_factors(self) -> pl.DataFrame:
        """Load computed factors."""
        return pl.read_parquet(os.path.join(self.output_dir, 'factors.parquet')) 