"""
Main ML Pipeline for Stock Return Prediction
============================================

Simple and clean pipeline orchestrator that uses modular components.

Usage:
    python main_pipeline.py --start_date 2020-01-01 --end_date 2024-12-31 --models linear_regression,gradient_boosting,dnn
"""

import argparse
import os
import sys
from typing import List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import DataProcessor, FactorProcessor, ModelTrainer

def run_pipeline(start_date: str, end_date: str, models: List[str]):
    """
    Run the complete ML pipeline.
    
    Args:
        start_date: Start date for data extraction
        end_date: End date for data extraction  
        models: List of model names to train
    """
    print("🚀 Starting Stock Return Prediction Pipeline")
    print("=" * 50)
    
    # Step 1: Data Processing
    print("\n📋 Step 1: Data Processing")
    print("-" * 30)
    data_processor = DataProcessor()
    
    # Check if data already exists
    if os.path.exists('data/com.parquet'):
        print("📁 Loading existing data...")
        data = data_processor.load_data()
    else:
        print("🔄 Extracting data from database...")
        data = data_processor.extract_data(start_date, end_date)
        data_processor.save_data(data)
    
    # Step 2: Factor Processing
    print("\n📋 Step 2: Factor Processing")
    print("-" * 30)
    factor_processor = FactorProcessor()
    
    # Check if factors already exist
    if os.path.exists('factors/factors.parquet'):
        print("📁 Loading existing factors...")
        factors = factor_processor.load_factors()
    else:
        print("🔄 Computing factors...")
        factors = factor_processor.compute_factors(data['com'])
        print("🔄 Creating target variable...")
        factors = factor_processor.create_target(factors, data['com'])
        factor_processor.save_factors(factors)
    
    # Step 3: Model Training
    print("\n📋 Step 3: Model Training")
    print("-" * 30)
    model_trainer = ModelTrainer()
    model_trainer.train_models(factors, models)
    
    # Step 4: Evaluation
    print("\n📋 Step 4: Model Evaluation")
    print("-" * 30)
    comparison_df = model_trainer.evaluate_models()
    
    # Print final results
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 50)
    print("\n📊 Final Results:")
    print("-" * 20)
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:20} | MSE: {row['MSE']:.6f} | "
              f"MAE: {row['MAE']:.6f} | R²: {row['R²']:.4f}")
    
    print(f"\n📁 Results saved in:")
    print("   - models/results/: Performance metrics")
    print("   - models/plots/: Visualization plots")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Stock Return Prediction Pipeline')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                       help='Start date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-12-31',
                       help='End date for data extraction (YYYY-MM-DD)')
    parser.add_argument('--models', type=str, 
                       default='linear_regression,gradient_boosting,dnn',
                       help='Comma-separated list of models to train')
    
    args = parser.parse_args()
    
    # Parse models
    models = [m.strip() for m in args.models.split(',')]
    
    # Run pipeline
    run_pipeline(args.start_date, args.end_date, models)

if __name__ == "__main__":
    main() 