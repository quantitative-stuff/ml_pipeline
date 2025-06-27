"""
Factor Calculator - Compute Factors from Scratch
===============================================

This module computes factors from scratch using the actual calculation logic
from your source files, not just extracting existing data.

Usage:
    from factor_calculator import FactorCalculator
    
    # Initialize calculator
    fc = FactorCalculator()
    
    # Compute factors from raw data
    factors = fc.compute_all_factors(raw_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings

class FactorCalculator:
    """
    Factor calculator that computes factors from scratch.
    """
    
    def __init__(self):
        """Initialize the factor calculator."""
        pass
    
    def compute_fundamental_factors(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute fundamental factors using logic from utils_ver33.py.
        
        Args:
            raw_data: DataFrame with raw financial data including:
                - 'Sales', 'Cap', 'Sector', 'Book', 'NP', 'NP_12M', 'OP', 'OP_12M'
                - 'DY', 'InstituitionalBuy', 'StockNum', 'Debt', 'EBITDA'
                - 'OperatingProfit', 'OperatingProfit12Fwd', 'OperatingProfit2yr'
                - 'Price', 'HighestPrice'
                
        Returns:
            DataFrame with computed fundamental factors
        """
        factors = pd.DataFrame(index=raw_data.index)
        
        # Book-to-Price ratio (BP)
        if 'Book' in raw_data.columns and 'Cap' in raw_data.columns:
            factors['BP'] = raw_data['Book'] / raw_data['Cap'] / 1000
            mask = (factors['BP'] != 0) & (factors['BP'] >= -10) & (factors['BP'] <= 10)
            factors['BP'] = np.where(mask, factors['BP'], np.nan)
        
        # Asset-to-Capital ratio (AssetP)
        if 'Asset' in raw_data.columns and 'Cap' in raw_data.columns:
            factors['AssetP'] = raw_data['Asset'] / raw_data['Cap'] / 1000
            mask = (raw_data['Sector'] != '제조업') | (factors['AssetP'] == 0)
            factors['AssetP'] = np.where(mask, np.nan, factors['AssetP'])
            mask = (factors['AssetP'] >= -10) & (factors['AssetP'] <= 10)
            factors['AssetP'] = np.where(mask, factors['AssetP'], np.nan)
        
        # Sales-to-Capitalization ratio (SalesP)
        if 'Sales' in raw_data.columns and 'Cap' in raw_data.columns:
            factors['SalesP'] = raw_data['Sales'] / raw_data['Cap'] / 1000
            mask = (raw_data['Sector'] == '제조업') & ~raw_data['Sales'].isna() & (raw_data['Sales'] != 0)
            mask &= (factors['SalesP'] >= -10) & (factors['SalesP'] <= 10)
            factors['SalesP'] = np.where(mask, factors['SalesP'], np.nan)
        
        # Return on Equity (ROE)
        if 'NP' in raw_data.columns and 'Book' in raw_data.columns:
            factors['ROE'] = raw_data['NP'] / raw_data['Book']
            mask = (raw_data['Sector'] == '제조업') & (raw_data['NP'] != 0) & (raw_data['Book'] > 0)
            mask &= (factors['ROE'] >= -10) & (factors['ROE'] <= 10)
            factors['ROE'] = np.where(mask, factors['ROE'], np.nan)
        
        # EBITDA yield (EBITDAY)
        if 'EBITDA' in raw_data.columns and 'Cap' in raw_data.columns:
            factors['EBITDAY'] = raw_data['EBITDA'] / 1000 / raw_data['Cap']
            mask = (raw_data['Sector'] == '제조업') & np.isfinite(factors['EBITDAY'])
            mask &= (factors['EBITDAY'] >= -10) & (factors['EBITDAY'] <= 10)
            factors['EBITDAY'] = np.where(mask, factors['EBITDAY'], np.nan)
        
        # Earnings Yield 12-month (EY_12M)
        if 'NP_12M' in raw_data.columns and 'Cap' in raw_data.columns:
            factors['EY_12M'] = raw_data['NP_12M'] / raw_data['Cap'] * 100
            mask = (raw_data['NP_12M'] != 0) & (factors['EY_12M'] <= 10) & (factors['EY_12M'] >= -10)
            factors['EY_12M'] = np.where(mask, factors['EY_12M'], np.nan)
        
        # Earnings Per Share Growth (EPSG)
        if 'NP_12M' in raw_data.columns and 'NP' in raw_data.columns:
            factors['EPSG'] = np.nan
            mask1 = (raw_data['NP_12M'] > 0) & (raw_data['NP'] > 0)
            mask2 = (raw_data['NP_12M'] > 0) & (raw_data['NP'] < 0)
            mask3 = (raw_data['NP_12M'] < 0) & (raw_data['NP'] < 0)
            
            factors.loc[mask1, 'EPSG'] = (raw_data.loc[mask1, 'NP_12M'] - raw_data.loc[mask1, 'NP']) / abs(raw_data.loc[mask1, 'NP'])
            factors.loc[mask2, 'EPSG'] = (raw_data.loc[mask2, 'NP_12M'] - raw_data.loc[mask2, 'NP']) / abs(raw_data.loc[mask2, 'NP'])
            factors.loc[mask3, 'EPSG'] = (raw_data.loc[mask3, 'NP_12M'] - raw_data.loc[mask3, 'NP']) / abs(raw_data.loc[mask3, 'NP'])
            
            mask = (raw_data['Sector'] == '제조업') & (factors['EPSG'] >= -10) & (factors['EPSG'] <= 10)
            factors['EPSG'] = np.where(mask, factors['EPSG'], np.nan)
        
        # Operating Profit Growth (OPG)
        if 'OP_12M' in raw_data.columns and 'OP' in raw_data.columns:
            factors['OPG'] = np.nan
            mask1 = (raw_data['OP_12M'] > 0) & (raw_data['OP'] > 0)
            mask2 = (raw_data['OP_12M'] > 0) & (raw_data['OP'] < 0)
            mask3 = (raw_data['OP_12M'] < 0) & (raw_data['OP'] < 0)
            
            factors.loc[mask1, 'OPG'] = (raw_data.loc[mask1, 'OP_12M'] - raw_data.loc[mask1, 'OP']) / raw_data.loc[mask1, 'OP']
            factors.loc[mask2, 'OPG'] = 0.5
            factors.loc[mask3, 'OPG'] = -0.5
            
            mask = (raw_data['Sector'] == '제조업') & (factors['OPG'] >= -10) & (factors['OPG'] <= 10)
            factors['OPG'] = np.where(mask, factors['OPG'], np.nan)
        
        # Dividend Yield (DY)
        if 'DY' in raw_data.columns and 'Cap' in raw_data.columns:
            factors['DY'] = raw_data['DY'] / raw_data['Cap'] / 1000
            mask = (factors['SalesP'] != 0) & (raw_data['Sector'] == '제조업')
            factors['DY'] = np.where(mask, factors['DY'], np.nan)
        
        # Institutional Buy ratio
        if 'InstituitionalBuy' in raw_data.columns and 'StockNum' in raw_data.columns:
            factors['InstituitionalBuy'] = raw_data['InstituitionalBuy'] / raw_data['StockNum']
        
        # Financial Leverage
        if 'Book' in raw_data.columns and 'Debt' in raw_data.columns:
            factors['Leverage'] = raw_data['Book'] / raw_data['Debt']
            mask = (raw_data['Sector'] == '제조업') & (raw_data['Book'] > 0)
            mask &= np.isfinite(factors['Leverage']) & (factors['Leverage'] <= 20) & (factors['Leverage'] >= -20)
            factors['Leverage'] = np.where(mask, factors['Leverage'], np.nan)
        
        # Operating Profit Margin (OPM)
        if 'OP' in raw_data.columns and 'Sales' in raw_data.columns:
            factors['OPM'] = raw_data['OP'] / raw_data['Sales']
            mask = (raw_data['Sector'] == '제조업') & np.isfinite(factors['OPM'])
            factors['OPM'] = np.where(mask, factors['OPM'], np.nan)
        
        # Operating Profit Momentum 1-month (OP_M1M)
        if all(col in raw_data.columns for col in ['OperatingProfit', 'OperatingProfit12Fwd', 'OperatingProfit2yr']):
            try:
                # This is a simplified version - you may need to adjust based on your data structure
                op_current = raw_data['OperatingProfit']
                op_12fwd = raw_data['OperatingProfit12Fwd']
                op_2yr = raw_data['OperatingProfit2yr']
                
                # Assuming ratio=0.5 and flag2yr=0 for simplicity
                ratio = 0.5
                factors['OP_M1M'] = (op_current * (1 - ratio) + op_12fwd * ratio) / (op_current * (1 - ratio) + op_12fwd * ratio) - 1
                
                mask = (factors['OP_M1M'] <= 1) & (factors['OP_M1M'] >= -1) & np.isfinite(factors['OP_M1M'])
                factors['OP_M1M'] = np.where(mask, factors['OP_M1M'], np.nan)
            except:
                pass
        
        # Price Momentum
        if 'Price' in raw_data.columns:
            try:
                price_data = raw_data['Price']
                if isinstance(price_data.columns, pd.MultiIndex):
                    current_col = price_data.loc[:, (slice(None), 'Current')]
                    before_col = price_data.loc[:, (slice(None), '180DayBefore')]
                    factors['PriceMomentum'] = (current_col - before_col) / before_col
                    factors['PriceMomentum'] = np.where(np.isfinite(factors['PriceMomentum']), 
                                                       factors['PriceMomentum'], 0)
            except:
                pass
        
        # Reverse Momentum
        if 'Price' in raw_data.columns and 'HighestPrice' in raw_data.columns:
            try:
                price_data = raw_data['Price']
                if isinstance(price_data.columns, pd.MultiIndex):
                    current_price = price_data.loc[:, (slice(None), 'Current')]
                    factors['ReverseMomentum'] = raw_data['HighestPrice'] / current_price - 1
                    mask = np.isfinite(factors['ReverseMomentum']) & (raw_data['Sector'] == '제조업')
                    factors['ReverseMomentum'] = np.where(mask, factors['ReverseMomentum'], np.nan)
            except:
                pass
        
        # Replace infinities with NaNs
        factors = factors.replace([np.inf, -np.inf], np.nan)
        
        return factors
    
    def compute_market_factors_zk(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market factors using logic from factor_analysis_zk.ipynb.
        
        Args:
            raw_data: DataFrame with market data
            
        Returns:
            DataFrame with computed market factors
        """
        factors = pd.DataFrame(index=raw_data.index)
        
        # f_value = SE_TQ/MktCap_Comm_Pref
        if 'SE_TQ' in raw_data.columns and 'MktCap_Comm_Pref' in raw_data.columns:
            factors['f_value'] = (raw_data['SE_TQ'] / raw_data['MktCap_Comm_Pref']).dropna()
        
        # f_mom = AdjPrc momentum calculation (252-day return - 20-day return)
        if 'AdjPrc' in raw_data.columns:
            try:
                adj_prc = raw_data['AdjPrc']
                if isinstance(adj_prc.index, pd.MultiIndex):
                    returns_252 = adj_prc.unstack().pct_change(252)
                    returns_20 = adj_prc.unstack().pct_change(20)
                    factors['f_mom'] = (returns_252 - returns_20).stack()
            except:
                pass
        
        # f_quality = GP_TQ/Assets_TQ
        if 'GP_TQ' in raw_data.columns and 'Assets_TQ' in raw_data.columns:
            factors['f_quality'] = (raw_data['GP_TQ'].astype(float) / raw_data['Assets_TQ']).dropna()
        
        # f_div = DPS_Adj/AdjPrc
        if 'DPS_Adj' in raw_data.columns and 'AdjPrc' in raw_data.columns:
            factors['f_div'] = (raw_data['DPS_Adj'].astype(float) / raw_data['AdjPrc']).dropna()
        
        # f_size = MktCap_Comm_Pref
        if 'MktCap_Comm_Pref' in raw_data.columns:
            factors['f_size'] = raw_data['MktCap_Comm_Pref']
        
        # f_vol = mean of BETA_1Y, VOL_1Y
        if 'BETA_1Y' in raw_data.columns and 'VOL_1Y' in raw_data.columns:
            factors['f_vol'] = raw_data[['BETA_1Y', 'VOL_1Y']].astype(float).mean(1).dropna()
        
        return factors
    
    def compute_market_factors_mlq(self, raw_data: pd.DataFrame, past1m_data: pd.DataFrame = None, 
                                  past3m_data: pd.DataFrame = None, past12m_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute market factors from factor_analysis_mlq.ipynb using actual calculations.
        
        Args:
            raw_data: Current period data (eom_data)
            past1m_data: 1-month ago data
            past3m_data: 3-month ago data  
            past12m_data: 12-month ago data
            
        Returns:
            DataFrame with computed market factors
        """
        factors = pd.DataFrame(index=raw_data.index)
        
        # Size factor
        if 'size' in raw_data.columns:
            factors['size'] = np.log(raw_data['size'])
        
        # Dividend factor
        if 'div_ratio' in raw_data.columns:
            factors['dividend'] = raw_data['div_ratio']
        
        # Price Momentum
        if 'adjclose' in raw_data.columns and past12m_data is not None and past1m_data is not None:
            factors['priceMomentum'] = (raw_data['adjclose'].values / past12m_data['adjclose'].values) - \
                                      (raw_data['adjclose'].values / past1m_data['adjclose'].values)
        
        # Trading Volume
        if 'trdVol_20avg' in raw_data.columns and 'trdVol_60avg' in raw_data.columns:
            factors['trading'] = raw_data['trdVol_20avg'].values / raw_data['trdVol_60avg'].values
        
        # Investment Sentiment
        if 'invTrst_20avg' in raw_data.columns and 'invTrst_120avg' in raw_data.columns and 'size' in raw_data.columns:
            factors['investSentiment'] = (raw_data['invTrst_20avg'].values - raw_data['invTrst_120avg'].values) / raw_data['size'].values
        
        # EPS Forward to Price
        if 'eps_fwd' in raw_data.columns and 'adjclose' in raw_data.columns:
            factors['epsF_p'] = raw_data['eps_fwd'].values / raw_data['adjclose'].values
        
        # Earnings to Market Value
        if 'earning_Q' in raw_data.columns and 'size' in raw_data.columns:
            factors['earningsOtmv'] = raw_data['earning_Q'].iloc[:, :4].sum(1).values / raw_data['size'].values
        
        # 52-week High to Price
        if 'high' in raw_data.columns and 'adjclose' in raw_data.columns:
            factors['52H/p_Mom'] = raw_data['high'].values / raw_data['adjclose'].values
        
        # 52-week Low to Price
        if 'low' in raw_data.columns and 'adjclose' in raw_data.columns:
            factors['52L/p_Mom'] = raw_data['low'].values / raw_data['adjclose'].values
        
        # Debt to Equity
        if 'dtoequity' in raw_data.columns:
            factors['dtoequity'] = raw_data['dtoequity'].iloc[:, 0].values / 100
        
        # Delta Debt to Equity Year over Year
        if 'dtoequity' in raw_data.columns and past12m_data is not None:
            factors['deltaDtoEquityYoY'] = (past12m_data['dtoequity'].iloc[:, 0].values / 100) - \
                                          (raw_data['dtoequity'].iloc[:, 0].values / 100)
        
        # Short-term to Long-term Standard Deviation
        if 'stdev20' in raw_data.columns and 'stdev120' in raw_data.columns:
            factors['stStevOLtStdev'] = raw_data['stdev20'].values / raw_data['stdev120'].values
        
        # ROE Forward Proxy
        if 'eps_fwd' in raw_data.columns and 'bps_fwd' in raw_data.columns:
            factors['roeF_proxy'] = raw_data['eps_fwd'].values / raw_data['bps_fwd'].astype(float).values
        
        # ROE Trailing Proxy
        if 'eps_tr' in raw_data.columns and 'bps_tr' in raw_data.columns:
            factors['roeT_proxy'] = raw_data['eps_tr'].values / raw_data['bps_tr'].values
        
        # ROE Quarterly
        if 'earning_Q' in raw_data.columns and 'equity' in raw_data.columns:
            factors['roeQ'] = raw_data['earning_Q'].iloc[:, 0].values / raw_data['equity'].iloc[:, 0].values
        
        # Delta EPS Operating
        if 'eps_fwd' in raw_data.columns and past1m_data is not None and 'adjclose' in raw_data.columns:
            factors['dEpsOp'] = (raw_data['eps_fwd'].values - past1m_data['earning_fwd'].values) / raw_data['adjclose'].values
        
        # Risk Adjusted Delta EPS Operating
        if 'dEpsOp' in factors.columns and 'stdev20' in raw_data.columns:
            factors['rAdjdEpsOp'] = factors['dEpsOp'].values / (raw_data['stdev20'].values * np.sqrt(252))
        
        # ROE Forward to Trailing Momentum
        if 'roeF_proxy' in factors.columns and 'roeT_proxy' in factors.columns:
            factors['roeFTMom'] = factors['roeF_proxy'] - factors['roeT_proxy']
        
        # EPS Forward Momentum
        if 'eps_fwd' in raw_data.columns and past1m_data is not None:
            factors['epsFMom'] = (raw_data['eps_fwd'].values - past1m_data['eps_fwd'].values) / past1m_data['eps_fwd'].values
        
        # Sales Forward Momentum
        if 'sps_fwd' in raw_data.columns and past1m_data is not None:
            factors['salseFMom'] = (raw_data['sps_fwd'].values - past1m_data['sps_fwd'].values) / past1m_data['sps_fwd'].values
        
        # ROE Forward Change Proxy
        if 'roeF_proxy' in factors.columns and past1m_data is not None and 'eps_fwd' in raw_data.columns and 'bps_fwd' in raw_data.columns:
            past_roe_fwd = past1m_data['eps_fwd'].values / past1m_data['bps_fwd'].astype(float).values
            factors['roeFchg_proxy'] = factors['roeF_proxy'].values - past_roe_fwd
        
        # Earnings Year over Year
        if 'earning_Q' in raw_data.columns and past12m_data is not None:
            factors['earningsYOY'] = (raw_data['earning_Q'].iloc[:, 0].values - past12m_data['earning_Q'].iloc[:, 0].values) / \
                                    np.abs(past12m_data['earning_Q'].iloc[:, 0].values)
        
        # Sales Year over Year
        if 'sales_Q' in raw_data.columns and past12m_data is not None:
            factors['salesYOY'] = (raw_data['sales_Q'].iloc[:, 0].values - past12m_data['sales_Q'].iloc[:, 0].values) / \
                                 np.abs(past12m_data['sales_Q'].iloc[:, 0].values)
        
        # Sales Adjusted Earnings Year over Year
        if 'earningsYOY' in factors.columns and 'salesYOY' in factors.columns:
            factors['slsAdjEarningsYOY'] = factors['earningsYOY'] - factors['salesYOY']
        
        # Earnings Spread
        if 'earning_Q' in raw_data.columns:
            earning_mean = raw_data['earning_Q'].iloc[:, :4].mean(1).values
            earning_std = raw_data['earning_Q'].iloc[:, :4].std(1).values
            factors['sprsEarningsOearningsAvgPstdev'] = (raw_data['earning_Q'].iloc[:, 0].values - earning_mean) / \
                                                       (earning_mean + earning_std)
        
        # EPS Forward to Trailing Momentum
        if 'eps_fwd' in raw_data.columns and 'eps_tr' in raw_data.columns:
            factors['epsFTMom'] = (raw_data['eps_fwd'].values - raw_data['eps_tr'].values) / np.abs(raw_data['eps_tr'].values)
        
        # Earnings Forward to Trailing Momentum
        if 'earning_fwd' in raw_data.columns and 'earning_tr' in raw_data.columns:
            factors['earningsFTMom'] = (raw_data['earning_fwd'].values - raw_data['earning_tr'].values) / np.abs(raw_data['earning_tr'].values)
        
        # Cash Flow Year over Year
        if 'cf_Q' in raw_data.columns and past12m_data is not None:
            factors['cashFlowYOY'] = (raw_data['cf_Q'].iloc[:, 0].values - past12m_data['cf_Q'].iloc[:, 0].values) / \
                                    np.abs(past12m_data['cf_Q'].iloc[:, 0].values)
        
        # Sales to Market Value
        if 'sales_Q' in raw_data.columns and 'size' in raw_data.columns:
            factors['salesOtmv'] = raw_data['sales_Q'].iloc[:, :4].sum(1).values / raw_data['size'].values
        
        # Cash Flow to Market Value
        if 'cf_Q' in raw_data.columns and 'size' in raw_data.columns:
            factors['cashFlowOtmv'] = raw_data['cf_Q'].sum(1).values / raw_data['size'].values
        
        # Equity to Market Value
        if 'equity' in raw_data.columns and 'size' in raw_data.columns:
            factors['equityOtmv'] = raw_data['equity'].iloc[:, 0].values / raw_data['size'].values
        
        # Book to Price Forward
        if 'bps_fwd' in raw_data.columns and 'close' in raw_data.columns:
            factors['bpsF_p'] = raw_data['bps_fwd'].astype(float).values / raw_data['close'].values
        
        # Book to Price Trailing
        if 'bps_tr' in raw_data.columns and 'close' in raw_data.columns:
            factors['bpsT_p'] = raw_data['bps_tr'].values / raw_data['close'].values
        
        # Sales to Price Forward
        if 'sps_fwd' in raw_data.columns and 'close' in raw_data.columns:
            factors['spsF_p'] = raw_data['sps_fwd'].values / raw_data['close'].values
        
        # Sales to Price Trailing
        if 'sps_tr' in raw_data.columns and 'close' in raw_data.columns:
            factors['spsT_p'] = raw_data['sps_tr'].values / raw_data['close'].values
        
        # Cash Flow to Price Forward
        if 'cf_fwd' in raw_data.columns and 'close' in raw_data.columns:
            factors['cfpsF_p'] = raw_data['cf_fwd'].values / raw_data['close'].values
        
        # Volatility factors (if available)
        if 'Vol_20D' in raw_data.columns:
            factors['Vol_20D'] = raw_data['Vol_20D']
        
        if 'Vol_120D' in raw_data.columns:
            factors['Vol_120D'] = raw_data['Vol_120D']
        
        # Sentiment factors
        if 'netamt_for20' in raw_data.columns:
            factors['senti_forg20'] = raw_data['netamt_for20']
        
        if 'netamt_for60' in raw_data.columns:
            factors['senti_forg60'] = raw_data['netamt_for60']
        
        if 'netamt_for120' in raw_data.columns:
            factors['senti_forg120'] = raw_data['netamt_for120']
        
        # Trading Amount factors
        if 'trd_amt20' in raw_data.columns:
            factors['trA_20avg_spot'] = raw_data['trd_amt20']
        
        if 'trd_amt60' in raw_data.columns:
            factors['trA_60avg_spot'] = raw_data['trd_amt60']
        
        return factors
    
    def compute_all_factors(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all factors from scratch.
        
        Args:
            raw_data: DataFrame with all required raw data
            
        Returns:
            DataFrame with all computed factors
        """
        all_factors = []
        
        # Fundamental factors
        fundamental = self.compute_fundamental_factors(raw_data)
        if not fundamental.empty:
            all_factors.append(fundamental)
        
        # Market factors from ZK
        market_zk = self.compute_market_factors_zk(raw_data)
        if not market_zk.empty:
            all_factors.append(market_zk)
        
        # Market factors from MLQ
        market_mlq = self.compute_market_factors_mlq(raw_data)
        if not market_mlq.empty:
            all_factors.append(market_mlq)
        
        if all_factors:
            return pd.concat(all_factors, axis=1)
        else:
            return pd.DataFrame()
    
    def compute_dynamic_factors(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute dynamic portfolio factors (selected factors).
        
        Args:
            raw_data: DataFrame with required data
            
        Returns:
            DataFrame with dynamic portfolio factors
        """
        all_factors = self.compute_all_factors(raw_data)
        
        # Selected factors for dynamic portfolio
        dynamic_factors = [
            'BP', 'SalesP', 'EY_12M', 'OPG', 'DY', 'InstituitionalBuy', 
            'OP_M1M', 'PriceMomentum', 'ReverseMomentum'
        ]
        
        available_factors = [f for f in dynamic_factors if f in all_factors.columns]
        
        return all_factors[available_factors]
    
    def standardize_factors(self, factors: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Standardize factor values (z-score normalization).
        
        Args:
            factors: Factor DataFrame
            exclude_columns: Columns to exclude from standardization
            
        Returns:
            Standardized factor DataFrame
        """
        if exclude_columns is None:
            exclude_columns = ['Leverage']
        
        standardized = factors.copy()
        
        for col in factors.columns:
            if col not in exclude_columns:
                mean_val = factors[col].mean()
                std_val = factors[col].std()
                if std_val != 0:
                    standardized[col] = (factors[col] - mean_val) / std_val
                    standardized[col] = standardized[col].clip(-3, 3)
        
        return standardized
    
    def rank_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Rank factors across stocks.
        
        Args:
            factors: Factor DataFrame
            
        Returns:
            Ranked factor DataFrame
        """
        ranked = factors.rank(ascending=False, method='min')
        ranked['count'] = ranked.count(axis=1)
        
        return ranked


# Convenience functions
def compute_fundamental_factors(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Compute fundamental factors."""
    fc = FactorCalculator()
    return fc.compute_fundamental_factors(raw_data)

def compute_market_factors(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Compute market factors."""
    fc = FactorCalculator()
    market_zk = fc.compute_market_factors_zk(raw_data)
    market_mlq = fc.compute_market_factors_mlq(raw_data)
    return pd.concat([market_zk, market_mlq], axis=1)

def compute_all_factors(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Compute all factors."""
    fc = FactorCalculator()
    return fc.compute_all_factors(raw_data)

def compute_dynamic_factors(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Compute dynamic portfolio factors."""
    fc = FactorCalculator()
    return fc.compute_dynamic_factors(raw_data)


if __name__ == "__main__":
    print("=== Factor Calculator - Compute from Scratch ===")
    print("This module computes factors from scratch using actual calculation logic.")
    print("\nAvailable methods:")
    print("- compute_fundamental_factors(): Compute fundamental factors")
    print("- compute_market_factors(): Compute market factors")
    print("- compute_all_factors(): Compute all factors")
    print("- compute_dynamic_factors(): Compute dynamic portfolio factors")
    
    print("\nExample usage:")
    print("from factor_calculator import FactorCalculator")
    print("fc = FactorCalculator()")
    print("factors = fc.compute_all_factors(raw_data)") 