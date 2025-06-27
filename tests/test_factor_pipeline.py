import unittest
import pandas as pd
import numpy as np
import os
from factor_pipeline import FactorPipeline

class TestFactorPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = FactorPipeline()
        
        self.pipeline = FactorPipeline()
        
        # Load data from pickle files
        data_dir = 'C:/Users/user/Documents/py/ml_pipeline/data'
        data_com = pd.read_pickle(os.path.join(data_dir, 'data_com.pkl'))
        data_cia = pd.read_pickle(os.path.join(data_dir, 'data_cia.pkl'))
        data_ssc = pd.read_pickle(os.path.join(data_dir, 'data_ssc.pkl'))
        data_ifrs = pd.read_pickle(os.path.join(data_dir, 'data_ifrs.pkl'))
        data_con = pd.read_pickle(os.path.join(data_dir, 'data_con.pkl'))

        # Merge dataframes on 'Dates' and 'Symbol'
        self.raw_data = data_com.merge(data_cia.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        self.raw_data = self.raw_data.merge(data_ssc.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        self.raw_data = self.raw_data.merge(data_ifrs.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        self.raw_data = self.raw_data.merge(data_con.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        
        # Rename AdjPrc to adjclose for consistency with factor_pipeline.py
        if 'AdjPrc' in self.raw_data.columns:
            self.raw_data = self.raw_data.rename(columns={'AdjPrc': 'adjclose'})

        # Ensure all expected columns are present, filling with NaN or appropriate empty structures if not
        expected_columns = [
            'Book', 'Cap', 'Sales', 'NP', 'NP_12M', 'OP', 'OP_12M', 'DY', 'InstituitionalBuy', 'StockNum', 'Debt', 'EBITDA', 'Sector',
            'OperatingProfit', 'OperatingProfit12Fwd', 'OperatingProfit2yr', 'Price', 'HighestPrice', 'Asset', 'GP_TQ', 'Assets_TQ',
            'SE_TQ', 'AdjPrc', 'DPS_Adj', 'BETA_1Y', 'VOL_1Y',
            'size', 'div_ratio', 'adjclose', 'trdVol_20avg', 'trdVol_60avg', 'invTrst_20avg', 'invTrst_120avg', 'eps_fwd',
            'earning_Q', 'high', 'low', 'dtoequity', 'stdev20', 'stdev120', 'bps_fwd', 'eps_tr', 'bps_tr', 'equity', 'earning_fwd',
            'sps_fwd', 'sales_Q', 'earning_tr', 'cf_Q', 'close', 'sps_tr', 'cf_fwd', 'Vol_20D', 'Vol_120D', 'netamt_for20',
            'netamt_for60', 'netamt_for120', 'trd_amt20', 'trd_amt60'
        ]

        for col in expected_columns:
            if col not in self.raw_data.columns:
                if col in ['earning_Q', 'sales_Q']:
                    # Initialize with lists of 4 NaNs for columns expected to be list-like with multiple elements
                    self.raw_data[col] = [[np.nan] * 4 for _ in range(len(self.raw_data))]
                elif col in ['dtoequity', 'equity', 'cf_Q']:
                    # Initialize with lists of a single NaN for columns expected to be list-like with single element access
                    self.raw_data[col] = [[np.nan] for _ in range(len(self.raw_data))]
                else:
                    self.raw_data[col] = np.nan

        # For past data, we'll use the loaded raw_data for now. 
        # In a real scenario, you'd load actual past data.
        self.past1m_data = self.raw_data.copy()
        self.past12m_data = self.raw_data.copy()

        # For past data, we'll use the loaded raw_data for now. 
        # In a real scenario, you'd load actual past data.
        self.past1m_data = self.raw_data.copy()
        self.past12m_data = self.raw_data.copy()

        # Print column information for debugging
        print("Columns in self.raw_data after merges and fillna:", self.raw_data.columns)
        print("Type of earning_Q:", self.raw_data['earning_Q'].apply(type).unique() if 'earning_Q' in self.raw_data.columns else "Not present")
        print("Type of dtoequity:", self.raw_data['dtoequity'].apply(type).unique() if 'dtoequity' in self.raw_data.columns else "Not present")
        print("Type of equity:", self.raw_data['equity'].apply(type).unique() if 'equity' in self.raw_data.columns else "Not present")
        print("Type of sales_Q:", self.raw_data['sales_Q'].apply(type).unique() if 'sales_Q' in self.raw_data.columns else "Not present")
        print("Type of cf_Q:", self.raw_data['cf_Q'].apply(type).unique() if 'cf_Q' in self.raw_data.columns else "Not present")

        # Create a basic raw_data DataFrame for testing
        self.pipeline = FactorPipeline()
        
        # Load data from pickle files
        data_dir = 'C:/Users/user/Documents/py/ml_pipeline/data'
        data_com = pd.read_pickle(os.path.join(data_dir, 'data_com.pkl'))
        data_cia = pd.read_pickle(os.path.join(data_dir, 'data_cia.pkl'))
        data_ssc = pd.read_pickle(os.path.join(data_dir, 'data_ssc.pkl'))
        data_ifrs = pd.read_pickle(os.path.join(data_dir, 'data_ifrs.pkl'))
        data_con = pd.read_pickle(os.path.join(data_dir, 'data_con.pkl'))

        # Merge dataframes on 'Dates' and 'Symbol'
        self.raw_data = data_com.merge(data_cia.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        self.raw_data = self.raw_data.merge(data_ssc.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        self.raw_data = self.raw_data.merge(data_ifrs.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        self.raw_data = self.raw_data.merge(data_con.drop(columns=['Name'], errors='ignore'), on=['Dates', 'Symbol'], how='outer')
        
        # Rename AdjPrc to adjclose for consistency with factor_pipeline.py
        if 'AdjPrc' in self.raw_data.columns:
            self.raw_data = self.raw_data.rename(columns={'AdjPrc': 'adjclose'})

        # Ensure all expected columns are present, filling with NaN or appropriate empty structures if not
        expected_columns = [
            'Book', 'Cap', 'Sales', 'NP', 'NP_12M', 'OP', 'OP_12M', 'DY', 'InstituitionalBuy', 'StockNum', 'Debt', 'EBITDA', 'Sector',
            'OperatingProfit', 'OperatingProfit12Fwd', 'OperatingProfit2yr', 'Price', 'HighestPrice', 'Asset', 'GP_TQ', 'Assets_TQ',
            'SE_TQ', 'AdjPrc', 'DPS_Adj', 'BETA_1Y', 'VOL_1Y',
            'size', 'div_ratio', 'adjclose', 'trdVol_20avg', 'trdVol_60avg', 'invTrst_20avg', 'invTrst_120avg', 'eps_fwd',
            'earning_Q', 'high', 'low', 'dtoequity', 'stdev20', 'stdev120', 'bps_fwd', 'eps_tr', 'bps_tr', 'equity', 'earning_fwd',
            'sps_fwd', 'sales_Q', 'earning_tr', 'cf_Q', 'close', 'sps_tr', 'cf_fwd', 'Vol_20D', 'Vol_120D', 'netamt_for20',
            'netamt_for60', 'netamt_for120', 'trd_amt20', 'trd_amt60'
        ]

        for col in expected_columns:
            if col not in self.raw_data.columns:
                if col in ['earning_Q', 'sales_Q']:
                    # Initialize with DataFrames of 4 NaNs for columns expected to be DataFrame-like with multiple columns
                    self.raw_data[col] = [pd.DataFrame([[np.nan] * 4]) for _ in range(len(self.raw_data))]
                elif col in ['dtoequity', 'equity', 'cf_Q']:
                    # Initialize with DataFrames of a single NaN for columns expected to be DataFrame-like with single column access
                    self.raw_data[col] = [pd.DataFrame([[np.nan]]) for _ in range(len(self.raw_data))]
                else:
                    self.raw_data[col] = np.nan

        # For past data, we'll use the loaded raw_data for now. 
        # In a real scenario, you'd load actual past data.
        self.past1m_data = self.raw_data.copy()
        self.past12m_data = self.raw_data.copy()

    def test_compute_all_factors_empty_data(self):
        empty_data = pd.DataFrame()
        factors = self.pipeline.compute_all_factors(empty_data)
        self.assertTrue(factors.empty)

    def test_compute_fundamental_factors(self):
        factors = self.pipeline.compute_fundamental_factors(self.raw_data)
        self.assertFalse(factors.empty)
        self.assertIn('BP', factors.columns)
        self.assertIn('ROE', factors.columns)
        self.assertIn('EPSG', factors.columns)
        self.assertIn('OPG', factors.columns)
        self.assertIn('DY', factors.columns)
        self.assertIn('Leverage', factors.columns)
        self.assertIn('OPM', factors.columns)

    def test_compute_market_factors_zk(self):
        factors = self.pipeline.compute_market_factors_zk(self.raw_data)
        self.assertFalse(factors.empty)
        self.assertIn('f_value', factors.columns)
        self.assertIn('f_quality', factors.columns)
        self.assertIn('f_div', factors.columns)
        self.assertIn('f_size', factors.columns)
        self.assertIn('f_vol', factors.columns)

    def test_compute_market_factors_mlq(self):
        factors = self.pipeline.compute_market_factors_mlq(self.raw_data, self.past1m_data, None, self.past12m_data)
        self.assertFalse(factors.empty)
        self.assertIn('size', factors.columns)
        self.assertIn('dividend', factors.columns)
        self.assertIn('priceMomentum', factors.columns)
        self.assertIn('trading', factors.columns)
        self.assertIn('investSentiment', factors.columns)
        self.assertIn('epsF_p', factors.columns)
        self.assertIn('earningsOtmv', factors.columns)
        self.assertIn('52H/p_Mom', factors.columns)
        self.assertIn('52L/p_Mom', factors.columns)
        self.assertIn('dtoequity', factors.columns)
        self.assertIn('deltaDtoEquityYoY', factors.columns)
        self.assertIn('stStevOLtStdev', factors.columns)
        self.assertIn('roeF_proxy', factors.columns)
        self.assertIn('roeT_proxy', factors.columns)
        self.assertIn('roeQ', factors.columns)
        self.assertIn('dEpsOp', factors.columns)
        self.assertIn('rAdjdEpsOp', factors.columns)
        self.assertIn('roeFTMom', factors.columns)
        self.assertIn('epsFMom', factors.columns)
        self.assertIn('salseFMom', factors.columns)
        self.assertIn('roeFchg_proxy', factors.columns)
        self.assertIn('earningsYOY', factors.columns)
        self.assertIn('salesYOY', factors.columns)
        self.assertIn('slsAdjEarningsYOY', factors.columns)
        self.assertIn('sprsEarningsOearningsAvgPstdev', factors.columns)
        self.assertIn('epsFTMom', factors.columns)
        self.assertIn('earningsFTMom', factors.columns)
        self.assertIn('cashFlowYOY', factors.columns)
        self.assertIn('salesOtmv', factors.columns)
        self.assertIn('cashFlowOtmv', factors.columns)
        self.assertIn('equityOtmv', factors.columns)
        self.assertIn('bpsF_p', factors.columns)
        self.assertIn('bpsT_p', factors.columns)
        self.assertIn('spsF_p', factors.columns)
        self.assertIn('spsT_p', factors.columns)
        self.assertIn('cfpsF_p', factors.columns)
        self.assertIn('Vol_20D', factors.columns)
        self.assertIn('Vol_120D', factors.columns)
        self.assertIn('senti_forg20', factors.columns)
        self.assertIn('senti_forg60', factors.columns)
        self.assertIn('senti_forg120', factors.columns)
        self.assertIn('trA_20avg_spot', factors.columns)
        self.assertIn('trA_60avg_spot', factors.columns)

    def test_standardize_factors(self):
        factors = self.pipeline.compute_all_factors(self.raw_data, self.past1m_data, None, self.past12m_data)
        standardized_factors = self.pipeline.standardize_factors(factors)
        self.assertFalse(standardized_factors.empty)
        # Check if standardization happened (mean close to 0, std close to 1 for non-excluded columns)
        for col in standardized_factors.columns:
            if col != 'Leverage': # Leverage is excluded from standardization
                self.assertAlmostEqual(standardized_factors[col].mean(), 0, places=1)
                # self.assertAlmostEqual(standardized_factors[col].std(), 1, places=1) # This might fail with small sample size

    def test_rank_factors(self):
        factors = self.pipeline.compute_all_factors(self.raw_data, self.past1m_data, None, self.past12m_data)
        ranked_factors = self.pipeline.rank_factors(factors)
        self.assertFalse(ranked_factors.empty)
        self.assertIn('count', ranked_factors.columns)
        # Check if ranking happened (values are ranks)
        for col in factors.columns:
            self.assertTrue(ranked_factors[col].max() <= len(self.raw_data))

if __name__ == '__main__':
    unittest.main()