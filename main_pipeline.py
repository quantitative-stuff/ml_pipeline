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
import logging
from typing import List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("pipeline.log"),
                        logging.StreamHandler()
                    ])

from pipeline import DataProcessor, FactorProcessor, ModelTrainer

def run_pipeline(start_date: str, end_date: str, models: List[str]):
    """
    Run the complete ML pipeline.
    
    Args:
        start_date: Start date for data extraction
        end_date: End date for data extraction  
        models: List of model names to train
    """
    logging.info("Starting Stock Return Prediction Pipeline")
    logging.info(f"Data period: {start_date} to {end_date}")
    logging.info("=" * 50)
    
    # Step 1: Data Processing
    logging.info("\nStep 1: Data Processing")
    logging.info("-" * 30)
    data_processor = DataProcessor()
    
    # Check if data already exists
    if os.path.exists('data/com.parquet'):
        logging.info("Loading existing data...")
        data = data_processor.load_data()
    else:
        logging.info("Extracting data from database...")
        data = data_processor.extract_data(start_date, end_date)
        data_processor.save_data(data)
    
    # Step 2: Factor Processing
    logging.info("\nStep 2: Factor Processing")
    logging.info("-" * 30)
    factor_processor = FactorProcessor()
    
    # Check if factors already exist
    if os.path.exists('factors/factors.parquet'):
        logging.info("Loading existing factors...")
        factors = factor_processor.load_factors()
    else:
        logging.info("Computing factors...")
        factors = factor_processor.compute_factors(data['com'])
        logging.info("Creating target variable...")
        factors = factor_processor.create_target(factors, data['com'])
        factor_processor.save_factors(factors)
    
    # Step 3: Model Training
    logging.info("\nStep 3: Model Training")
    logging.info("-" * 30)
    model_trainer = ModelTrainer()
    model_trainer.train_models(factors, models)
    
    # Step 4: Evaluation
    logging.info("\nStep 4: Model Evaluation")
    logging.info("-" * 30)
    comparison_df = model_trainer.evaluate_models()
    
    # Print final results
    logging.info("Pipeline completed successfully!")
    logging.info("=" * 50)
    logging.info("\nFinal Results:")
    logging.info("-" * 20)
    for _, row in comparison_df.iterrows():
        logging.info(f"{row['Model']:20} | MSE: {row['MSE']:.6f} | "
              f"MAE: {row['MAE']:.6f} | R²: {row['R²']:.4f}")
    
    logging.info(f"\nResults saved in:")
    logging.info("   - models/results/: Performance metrics")
    logging.info("   - models/plots/: Visualization plots")

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