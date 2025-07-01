"""
Data Processing Module
=====================

Handles data extraction from database and preprocessing.
"""

import polars as pl
import pandas as pd
from typing import Dict, Tuple
import os
import logging

from data_loader import get_data

class DataProcessor:
    """Handles data extraction and preprocessing."""
    
    def __init__(self, output_dir: str = 'data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_data(self, start_date: str, end_date: str) -> Dict[str, pl.DataFrame]:
        """Extract data from database and convert to Polars."""
        logging.info("Extracting data from database...")
        
        data_com, data_cia, data_ssc, data_nfr_ifrs, data_nfs_ifrs, data_con = get_data(
            start_date, end_date
        )
        
        # Convert to Polars for efficiency
        data = {
            'com': pl.from_pandas(data_com),
            'cia': pl.from_pandas(data_cia),
            'ssc': pl.from_pandas(data_ssc),
            'nfr_ifrs': pl.from_pandas(data_nfr_ifrs),
            'nfs_ifrs': pl.from_pandas(data_nfs_ifrs),
            'con': pl.from_pandas(data_con)
        }
        
        logging.info(f"Data extracted: {len(data_com)} records")
        return data
    
    def save_data(self, data: Dict[str, pl.DataFrame]):
        """Save extracted data to files."""
        for name, df in data.items():
            df.write_parquet(os.path.join(self.output_dir, f'{name}.parquet'))
        logging.info(f"Data saved to {self.output_dir}")
    
    def load_data(self) -> Dict[str, pl.DataFrame]:
        """Load data from saved files."""
        data = {}
        for file in os.listdir(self.output_dir):
            if file.endswith('.parquet'):
                name = file.replace('.parquet', '')
                data[name] = pl.read_parquet(os.path.join(self.output_dir, file))
        return data 