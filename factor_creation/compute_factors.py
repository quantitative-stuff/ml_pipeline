

import pandas as pd
import polars as pl
from .factor_pipeline import FactorPipeline
import os

def load_data(data_path):
    """Load and combine data from pickle files."""
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]
    
    # Load all dataframes
    dfs = []
    for f in all_files:
        df = pd.read_pickle(f)
        if isinstance(df.index, pd.MultiIndex):
            dfs.append(df)
        else:
            # Assuming the index is the date
            df.index.name = 'date'
            dfs.append(df)

    # For simplicity, we'll merge them on their index. 
    # This assumes the indices are compatible (e.g., same tickers and dates).
    # A more robust solution would involve a more sophisticated merging strategy.
    merged_df = pd.concat(dfs, axis=1)
    # print(merged_df.columns)
    # Handle duplicate columns by keeping the first one
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]

    # Reset index to make 'Symbol' and 'Dates' columns
    merged_df = merged_df.reset_index()
    if 'Dates' not in merged_df.columns and 'level_0' in merged_df.columns:
        merged_df = merged_df.rename(columns={'level_0': 'Dates'})
    if 'Symbol' not in merged_df.columns and 'level_1' in merged_df.columns:
        merged_df = merged_df.rename(columns={'level_1': 'Symbol'})


    # Convert to Polars DataFrame
    return pl.from_pandas(merged_df)

if __name__ == "__main__":
    data_path = 'data'
    
    print("Loading and combining data...")
    raw_data = load_data(data_path)
    
    print("Computing factors...")
    pipeline = FactorPipeline()
    all_factors = pipeline.compute_all_factors(raw_data)
    
    print("Factors computed successfully.")
    print(all_factors.head())
    
    # Save the factors to a Parquet file
    all_factors.write_parquet('all_factors.parquet')
    print("Factors saved to all_factors.parquet")

