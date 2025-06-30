"""
Simple Pipeline Runner
=====================

Easy-to-use interface for running the ML pipeline.
"""

import argparse
from main_pipeline import run_pipeline
from config import PIPELINE_CONFIG, DATA_CONFIG

def main():
    """Simple interface to run the pipeline."""
    parser = argparse.ArgumentParser(description='Stock Return Prediction Pipeline Runner')
    
    # Quick run options
    parser.add_argument('--quick', action='store_true', 
                       help='Quick run with default settings')
    parser.add_argument('--full', action='store_true',
                       help='Full run with all models')
    
    # Custom options
    parser.add_argument('--start_date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of models')
    
    args = parser.parse_args()
    
    # Set defaults based on config
    start_date = args.start_date or PIPELINE_CONFIG['default_start_date']
    end_date = args.end_date or PIPELINE_CONFIG['default_end_date']
    
    if args.quick:
        models = ['linear_regression', 'gradient_boosting']
    elif args.full:
        models = PIPELINE_CONFIG['available_models']['ml'] + PIPELINE_CONFIG['available_models']['dl']
    else:
        models = args.models.split(',') if args.models else PIPELINE_CONFIG['default_models']
    
    print(f"🚀 Running pipeline with:")
    print(f"   Start date: {start_date}")
    print(f"   End date: {end_date}")
    print(f"   Models: {', '.join(models)}")
    print()
    
    # Run the pipeline
    run_pipeline(start_date, end_date, models)

if __name__ == "__main__":
    main() 