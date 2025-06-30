"""
Factor Calculator using Polars for efficient factor computation.
This module provides vectorized factor calculations using Polars DataFrame operations.
No random number generation is used - all calculations are deterministic.
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, List
import warnings

class FactorCalculatorPL:
    """
    Efficient factor computation calculator using Polars.
    Implements vectorized operations for fast factor calculations.
    No random number generation - all calculations are deterministic.
    """
    
    def __init__(self):
        """Initialize the factor calculator."""
        # Factor dictionary mapping from DB_mlq_get_dbdata.ipynb
        self.factor_dict = {
            'sector': 'CP10000600',
            'industry': 'CP10000800',
            'mkt': 'CP10000300',
            'k2YN': 'CP10000310',
            'clsMnth': 'CP10002200',
            'adjPrice': 'S410000700',
            'trdYN': 'S410002600',
            'mgtYN': 'S410002700',
            'tmv': 'S420002100',
            'trdVol_20avg': 'S41000630F',
            'trdVol_60avg': 'S41000640F',
            'trdVol_120avg': 'S41000650F',
            'trA_20avg_spot': 'S410007000',
            'trA_60avg_spot': 'S410007100',
            'p_price_spot': 'S41000060F',
            'pHigh_spot': 'S41000040F',
            'pLow_spot': 'S41000040F',
            'retStdev_20_spot': 'S410008600',
            'retStdev_120_spot': 'S410008800',
            'senti_invTrst20': 'CI20033022',
            'senti_invTrst60': 'CI20033023',
            'senti_invTrst120': 'CI20033024',
            'senti_inst20': 'CI20003022',
            'senti_inst60': 'CI20003023',
            'senti_inst120': 'CI20003024',
            'senti_forg20': 'CI20113022',
            'senti_forg60': 'CI20113023',
            'senti_forg120': 'CI20113024',
            'epsConNmb': 'FM10012110',
            'epsF_spot': 'FM30041100',
            'epsT_spot': 'FM30041200',
            'bpsF_spot': 'FM30041250',
            'bpsT_spot': 'FM30041260',
            'salesF_spot': 'FM30041500',
            'salesT_spot': 'FM30041530',
            'earningsF_spot': 'FM30041515',
            'earningsT_spot': 'FM30041545',
            'cashFlowF_spot': 'FM30041585',
            'dbtRtQ_': 'M000102001',
            'totalEQ_0': 'M000903001',
            'salseQ_0': 'M000904001',
            'earningsQ_0': 'M000906001',
            'cashFlowQ_0': 'M000909011',
            'div_0': 'M000706032',
        }
    
    def compute_fundamental_factors(self, raw_data: pl.DataFrame) -> pl.DataFrame:
        """
        Compute fundamental factors using Polars.
        
        Args:
            raw_data: Raw financial data
            
        Returns:
            Polars DataFrame with computed fundamental factors
        """
        factors = raw_data.select(pl.all())
        
        # Value factors - Book to Price
        if (self.factor_dict['bpsF_spot'] in raw_data.columns and 
            self.factor_dict['p_price_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['bpsF_spot']).cast(pl.Float64) / 
                 pl.col(self.factor_dict['p_price_spot'])).alias('f_value')
            )
        
        # Quality factors - ROE
        if (self.factor_dict['earningsQ_0'] in raw_data.columns and 
            self.factor_dict['totalEQ_0'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['earningsQ_0']).cast(pl.Float64) / 
                 pl.col(self.factor_dict['totalEQ_0'])).alias('f_quality')
            )
        
        # Dividend factors
        if (self.factor_dict['div_0'] in raw_data.columns and 
            self.factor_dict['adjPrice'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['div_0']).cast(pl.Float64) / 
                 pl.col(self.factor_dict['adjPrice'])).alias('f_div')
            )
        
        # Size factors
        if self.factor_dict['tmv'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['tmv']).alias('f_size')
            )
        
        # Volatility factors - using available volatility measures
        if (self.factor_dict['retStdev_20_spot'] in raw_data.columns and 
            self.factor_dict['retStdev_120_spot'] in raw_data.columns):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['retStdev_20_spot']) + 
                  pl.col(self.factor_dict['retStdev_120_spot'])) / 2).alias('f_vol')
            )
        
        return factors
    
    def compute_market_factors_zk(self, raw_data: pl.DataFrame) -> pl.DataFrame:
        """
        Compute market factors from zk analysis using Polars.
        
        Args:
            raw_data: Raw financial data
            
        Returns:
            Polars DataFrame with computed market factors
        """
        factors = raw_data.select(pl.all())
        
        # Value factor - SE_TQ / MktCap_Comm_Pref
        if 'SE_TQ' in raw_data.columns and 'MktCap_Comm_Pref' in raw_data.columns:
            factors = factors.with_columns(
                (pl.col('SE_TQ') / pl.col('MktCap_Comm_Pref')).alias('f_value')
            )
        
        # Momentum factor - AdjPrc momentum calculation
        if 'AdjPrc' in raw_data.columns:
            try:
                # For Polars, we need to handle the multi-index differently
                # This is a simplified version - in practice you might need to restructure
                factors = factors.with_columns(
                    pl.col('AdjPrc').alias('f_mom')  # Placeholder - actual calculation depends on data structure
                )
            except:
                pass
        
        # Quality factor - GP_TQ / Assets_TQ
        if 'GP_TQ' in raw_data.columns and 'Assets_TQ' in raw_data.columns:
            factors = factors.with_columns(
                (pl.col('GP_TQ').cast(pl.Float64) / pl.col('Assets_TQ')).alias('f_quality')
            )
        
        # Dividend factor - DPS_Adj / AdjPrc
        if 'DPS_Adj' in raw_data.columns and 'AdjPrc' in raw_data.columns:
            factors = factors.with_columns(
                (pl.col('DPS_Adj').cast(pl.Float64) / pl.col('AdjPrc')).alias('f_div')
            )
        
        # Size factor - MktCap_Comm_Pref
        if 'MktCap_Comm_Pref' in raw_data.columns:
            factors = factors.with_columns(
                pl.col('MktCap_Comm_Pref').alias('f_size')
            )
        
        # Volatility factor - mean of BETA_1Y, VOL_1Y
        if 'BETA_1Y' in raw_data.columns and 'VOL_1Y' in raw_data.columns:
            factors = factors.with_columns(
                ((pl.col('BETA_1Y').cast(pl.Float64) + pl.col('VOL_1Y').cast(pl.Float64)) / 2).alias('f_vol')
            )
        
        return factors
    
    def compute_market_factors_mlq(self, raw_data: pl.DataFrame, 
                                  past1m_data: Optional[pl.DataFrame] = None,
                                  past3m_data: Optional[pl.DataFrame] = None, 
                                  past12m_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Compute market factors using Polars with proper column mapping.
        
        Args:
            raw_data: Current period data
            past1m_data: 1-month ago data
            past3m_data: 3-month ago data
            past12m_data: 12-month ago data
            
        Returns:
            Polars DataFrame with computed factors
        """
        # Start with the original data
        factors = raw_data.select(pl.all())
        
        # Size factor - using tmw (market value)
        if self.factor_dict['tmv'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['tmv']).log().alias('size')
            )
        
        # Dividend factor
        if self.factor_dict['div_0'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['div_0']).alias('dividend')
            )
        
        # Price Momentum - using adjPrice
        if (self.factor_dict['adjPrice'] in raw_data.columns and 
            past12m_data is not None and past1m_data is not None):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['adjPrice']) / 
                  past12m_data.select(self.factor_dict['adjPrice']).to_series()) -
                 (pl.col(self.factor_dict['adjPrice']) / 
                  past1m_data.select(self.factor_dict['adjPrice']).to_series())).alias('priceMomentum')
            )
        
        # Trading Volume
        if (self.factor_dict['trdVol_20avg'] in raw_data.columns and 
            self.factor_dict['trdVol_60avg'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['trdVol_20avg']) / 
                 pl.col(self.factor_dict['trdVol_60avg'])).alias('trading')
            )
        
        # Investment Sentiment
        if (self.factor_dict['senti_invTrst20'] in raw_data.columns and 
            self.factor_dict['senti_invTrst120'] in raw_data.columns and 
            self.factor_dict['tmv'] in raw_data.columns):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['senti_invTrst20']) - 
                  pl.col(self.factor_dict['senti_invTrst120'])) / 
                 pl.col(self.factor_dict['tmv'])).alias('investSentiment')
            )
        
        # EPS Forward to Price
        if (self.factor_dict['epsF_spot'] in raw_data.columns and 
            self.factor_dict['adjPrice'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['epsF_spot']) / 
                 pl.col(self.factor_dict['adjPrice'])).alias('epsF_p')
            )
        
        # Earnings to Market Value
        if (self.factor_dict['earningsQ_0'] in raw_data.columns and 
            self.factor_dict['tmv'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['earningsQ_0']) / 
                 pl.col(self.factor_dict['tmv'])).alias('earningsOtmv')
            )
        
        # 52-week High to Price
        if (self.factor_dict['pHigh_spot'] in raw_data.columns and 
            self.factor_dict['adjPrice'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['pHigh_spot']) / 
                 pl.col(self.factor_dict['adjPrice'])).alias('52H/p_Mom')
            )
        
        # 52-week Low to Price
        if (self.factor_dict['pLow_spot'] in raw_data.columns and 
            self.factor_dict['adjPrice'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['pLow_spot']) / 
                 pl.col(self.factor_dict['adjPrice'])).alias('52L/p_Mom')
            )
        
        # Debt to Equity
        if self.factor_dict['dbtRtQ_'] in raw_data.columns:
            factors = factors.with_columns(
                (pl.col(self.factor_dict['dbtRtQ_']) / 100).alias('dtoequity')
            )
        
        # Delta Debt to Equity Year over Year
        if (self.factor_dict['dbtRtQ_'] in raw_data.columns and 
            past12m_data is not None):
            factors = factors.with_columns(
                ((past12m_data.select(self.factor_dict['dbtRtQ_']).to_series() / 100) -
                 (pl.col(self.factor_dict['dbtRtQ_']) / 100)).alias('deltaDtoEquityYoY')
            )
        
        # Short-term to Long-term Standard Deviation
        if (self.factor_dict['retStdev_20_spot'] in raw_data.columns and 
            self.factor_dict['retStdev_120_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['retStdev_20_spot']) / 
                 pl.col(self.factor_dict['retStdev_120_spot'])).alias('stStevOLtStdev')
            )
        
        # ROE Forward Proxy
        if (self.factor_dict['epsF_spot'] in raw_data.columns and 
            self.factor_dict['bpsF_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['epsF_spot']) / 
                 pl.col(self.factor_dict['bpsF_spot']).cast(pl.Float64)).alias('roeF_proxy')
            )
        
        # ROE Trailing Proxy
        if (self.factor_dict['epsT_spot'] in raw_data.columns and 
            self.factor_dict['bpsT_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['epsT_spot']) / 
                 pl.col(self.factor_dict['bpsT_spot'])).alias('roeT_proxy')
            )
        
        # ROE Quarterly
        if (self.factor_dict['earningsQ_0'] in raw_data.columns and 
            self.factor_dict['totalEQ_0'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['earningsQ_0']) / 
                 pl.col(self.factor_dict['totalEQ_0'])).alias('roeQ')
            )
        
        # Delta EPS Operating
        if (self.factor_dict['epsF_spot'] in raw_data.columns and 
            past1m_data is not None and 
            self.factor_dict['adjPrice'] in raw_data.columns):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['epsF_spot']) - 
                  past1m_data.select(self.factor_dict['earningsF_spot']).to_series()) / 
                 pl.col(self.factor_dict['adjPrice'])).alias('dEpsOp')
            )
        
        # Risk Adjusted Delta EPS Operating
        if ('dEpsOp' in factors.columns and 
            self.factor_dict['retStdev_20_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col('dEpsOp') / 
                 (pl.col(self.factor_dict['retStdev_20_spot']) * np.sqrt(252))).alias('rAdjdEpsOp')
            )
        
        # ROE Forward to Trailing Momentum
        if 'roeF_proxy' in factors.columns and 'roeT_proxy' in factors.columns:
            factors = factors.with_columns(
                (pl.col('roeF_proxy') - pl.col('roeT_proxy')).alias('roeFTMom')
            )
        
        # EPS Forward Momentum
        if (self.factor_dict['epsF_spot'] in raw_data.columns and 
            past1m_data is not None):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['epsF_spot']) - 
                  past1m_data.select(self.factor_dict['epsF_spot']).to_series()) / 
                 past1m_data.select(self.factor_dict['epsF_spot']).to_series()).alias('epsFMom')
            )
        
        # Sales Forward Momentum
        if (self.factor_dict['salesF_spot'] in raw_data.columns and 
            past1m_data is not None):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['salesF_spot']) - 
                  past1m_data.select(self.factor_dict['salesF_spot']).to_series()) / 
                 past1m_data.select(self.factor_dict['salesF_spot']).to_series()).alias('salseFMom')
            )
        
        # ROE Forward Change Proxy
        if ('roeF_proxy' in factors.columns and 
            past1m_data is not None and 
            self.factor_dict['epsF_spot'] in raw_data.columns and 
            self.factor_dict['bpsF_spot'] in raw_data.columns):
            past_roe_fwd = (past1m_data.select(self.factor_dict['epsF_spot']).to_series() / 
                           past1m_data.select(self.factor_dict['bpsF_spot']).to_series().cast(pl.Float64))
            factors = factors.with_columns(
                (pl.col('roeF_proxy') - past_roe_fwd).alias('roeFchg_proxy')
            )
        
        # Earnings Year over Year
        if (self.factor_dict['earningsQ_0'] in raw_data.columns and 
            past12m_data is not None):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['earningsQ_0']) - 
                  past12m_data.select(self.factor_dict['earningsQ_0']).to_series()) / 
                 past12m_data.select(self.factor_dict['earningsQ_0']).to_series().abs()).alias('earningsYOY')
            )
        
        # Sales Year over Year
        if (self.factor_dict['salseQ_0'] in raw_data.columns and 
            past12m_data is not None):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['salseQ_0']) - 
                  past12m_data.select(self.factor_dict['salseQ_0']).to_series()) / 
                 past12m_data.select(self.factor_dict['salseQ_0']).to_series().abs()).alias('salesYOY')
            )
        
        # Sales Adjusted Earnings Year over Year
        if 'earningsYOY' in factors.columns and 'salesYOY' in factors.columns:
            factors = factors.with_columns(
                (pl.col('earningsYOY') - pl.col('salesYOY')).alias('slsAdjEarningsYOY')
            )
        
        # Earnings Spread (simplified)
        if self.factor_dict['earningsQ_0'] in raw_data.columns:
            factors = factors.with_columns(
                (pl.col(self.factor_dict['earningsQ_0']) / 
                 pl.col(self.factor_dict['earningsQ_0'])).alias('sprsEarningsOearningsAvgPstdev')
            )
        
        # EPS Forward to Trailing Momentum
        if (self.factor_dict['epsF_spot'] in raw_data.columns and 
            self.factor_dict['epsT_spot'] in raw_data.columns):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['epsF_spot']) - 
                  pl.col(self.factor_dict['epsT_spot'])) / 
                 pl.col(self.factor_dict['epsT_spot']).abs()).alias('epsFTMom')
            )
        
        # Earnings Forward to Trailing Momentum
        if (self.factor_dict['earningsF_spot'] in raw_data.columns and 
            self.factor_dict['earningsT_spot'] in raw_data.columns):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['earningsF_spot']) - 
                  pl.col(self.factor_dict['earningsT_spot'])) / 
                 pl.col(self.factor_dict['earningsT_spot']).abs()).alias('earningsFTMom')
            )
        
        # Cash Flow Year over Year
        if (self.factor_dict['cashFlowQ_0'] in raw_data.columns and 
            past12m_data is not None):
            factors = factors.with_columns(
                ((pl.col(self.factor_dict['cashFlowQ_0']) - 
                  past12m_data.select(self.factor_dict['cashFlowQ_0']).to_series()) / 
                 past12m_data.select(self.factor_dict['cashFlowQ_0']).to_series().abs()).alias('cashFlowYOY')
            )
        
        # Sales to Market Value
        if (self.factor_dict['salseQ_0'] in raw_data.columns and 
            self.factor_dict['tmv'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['salseQ_0']) / 
                 pl.col(self.factor_dict['tmv'])).alias('salesOtmv')
            )
        
        # Cash Flow to Market Value
        if (self.factor_dict['cashFlowQ_0'] in raw_data.columns and 
            self.factor_dict['tmv'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['cashFlowQ_0']) / 
                 pl.col(self.factor_dict['tmv'])).alias('cashFlowOtmv')
            )
        
        # Equity to Market Value
        if (self.factor_dict['totalEQ_0'] in raw_data.columns and 
            self.factor_dict['tmv'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['totalEQ_0']) / 
                 pl.col(self.factor_dict['tmv'])).alias('equityOtmv')
            )
        
        # Book to Price Forward
        if (self.factor_dict['bpsF_spot'] in raw_data.columns and 
            self.factor_dict['p_price_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['bpsF_spot']).cast(pl.Float64) / 
                 pl.col(self.factor_dict['p_price_spot'])).alias('bpsF_p')
            )
        
        # Book to Price Trailing
        if (self.factor_dict['bpsT_spot'] in raw_data.columns and 
            self.factor_dict['p_price_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['bpsT_spot']) / 
                 pl.col(self.factor_dict['p_price_spot'])).alias('bpsT_p')
            )
        
        # Sales to Price Forward
        if (self.factor_dict['salesF_spot'] in raw_data.columns and 
            self.factor_dict['p_price_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['salesF_spot']) / 
                 pl.col(self.factor_dict['p_price_spot'])).alias('spsF_p')
            )
        
        # Sales to Price Trailing
        if (self.factor_dict['salesT_spot'] in raw_data.columns and 
            self.factor_dict['p_price_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['salesT_spot']) / 
                 pl.col(self.factor_dict['p_price_spot'])).alias('spsT_p')
            )
        
        # Cash Flow to Price Forward
        if (self.factor_dict['cashFlowF_spot'] in raw_data.columns and 
            self.factor_dict['p_price_spot'] in raw_data.columns):
            factors = factors.with_columns(
                (pl.col(self.factor_dict['cashFlowF_spot']) / 
                 pl.col(self.factor_dict['p_price_spot'])).alias('cfpsF_p')
            )
        
        # Sentiment factors
        if self.factor_dict['senti_forg20'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['senti_forg20']).alias('senti_forg20')
            )
        
        if self.factor_dict['senti_forg60'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['senti_forg60']).alias('senti_forg60')
            )
        
        if self.factor_dict['senti_forg120'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['senti_forg120']).alias('senti_forg120')
            )
        
        # Trading Amount factors
        if self.factor_dict['trA_20avg_spot'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['trA_20avg_spot']).alias('trA_20avg_spot')
            )
        
        if self.factor_dict['trA_60avg_spot'] in raw_data.columns:
            factors = factors.with_columns(
                pl.col(self.factor_dict['trA_60avg_spot']).alias('trA_60avg_spot')
            )
        
        return factors
    
    def compute_all_factors(self, raw_data: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all available factors using Polars.
        
        Args:
            raw_data: Raw financial data
            
        Returns:
            Polars DataFrame with all computed factors
        """
        # Compute fundamental factors
        fundamental_factors = self.compute_fundamental_factors(raw_data)
        
        # Compute market factors (without past data for now)
        market_factors = self.compute_market_factors_mlq(raw_data)
        
        # Combine all factors
        all_factors = fundamental_factors.join(market_factors, how='outer')
        
        return all_factors
    
    def compute_dynamic_factors(self, raw_data: pl.DataFrame) -> pl.DataFrame:
        """
        Compute dynamic factors using Polars.
        
        Args:
            raw_data: Raw financial data
            
        Returns:
            Polars DataFrame with computed dynamic factors
        """
        factors = raw_data.select(pl.all())
        
        # Add any dynamic factor calculations here
        # These would be factors that depend on time series or rolling calculations
        
        return factors
    
    def standardize_factors(self, factors: pl.DataFrame, 
                           exclude_columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Standardize factors using z-score normalization.
        
        Args:
            factors: DataFrame with factors
            exclude_columns: Columns to exclude from standardization
            
        Returns:
            Standardized factors DataFrame
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # Get numeric columns excluding specified columns
        numeric_cols = [col for col in factors.columns 
                       if col not in exclude_columns and 
                       factors.select(pl.col(col)).dtypes[0] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        if not numeric_cols:
            return factors
        
        # Standardize each numeric column
        standardized = factors.select(pl.all())
        for col in numeric_cols:
            mean_val = factors.select(pl.col(col).mean()).item()
            std_val = factors.select(pl.col(col).std()).item()
            if std_val != 0:
                standardized = standardized.with_columns(
                    ((pl.col(col) - mean_val) / std_val).alias(col)
                )
        
        return standardized
    
    def rank_factors(self, factors: pl.DataFrame, 
                    exclude_columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Rank factors using percentile ranking.
        
        Args:
            factors: DataFrame with factors
            exclude_columns: Columns to exclude from ranking
            
        Returns:
            Ranked factors DataFrame
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # Get numeric columns excluding specified columns
        numeric_cols = [col for col in factors.columns 
                       if col not in exclude_columns and 
                       factors.select(pl.col(col)).dtypes[0] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        if not numeric_cols:
            return factors
        
        # Rank each numeric column
        ranked = factors.select(pl.all())
        for col in numeric_cols:
            ranked = ranked.with_columns(
                pl.col(col).rank(method='average', descending=False).alias(col)
            )
        
        return ranked


# Convenience functions
def compute_fundamental_factors(raw_data: pl.DataFrame) -> pl.DataFrame:
    """Convenience function to compute fundamental factors."""
    calculator = FactorCalculatorPL()
    return calculator.compute_fundamental_factors(raw_data)


def compute_market_factors_zk(raw_data: pl.DataFrame) -> pl.DataFrame:
    """Convenience function to compute market factors from zk analysis."""
    calculator = FactorCalculatorPL()
    return calculator.compute_market_factors_zk(raw_data)


def compute_market_factors_mlq(raw_data: pl.DataFrame, 
                              past1m_data: Optional[pl.DataFrame] = None,
                              past3m_data: Optional[pl.DataFrame] = None, 
                              past12m_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Convenience function to compute market factors."""
    calculator = FactorCalculatorPL()
    return calculator.compute_market_factors_mlq(raw_data, past1m_data, past3m_data, past12m_data)


def compute_all_factors(raw_data: pl.DataFrame) -> pl.DataFrame:
    """Convenience function to compute all factors."""
    calculator = FactorCalculatorPL()
    return calculator.compute_all_factors(raw_data)


def compute_dynamic_factors(raw_data: pl.DataFrame) -> pl.DataFrame:
    """Convenience function to compute dynamic factors."""
    calculator = FactorCalculatorPL()
    return calculator.compute_dynamic_factors(raw_data)


def standardize_factors(factors: pl.DataFrame, 
                       exclude_columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Convenience function to standardize factors."""
    calculator = FactorCalculatorPL()
    return calculator.standardize_factors(factors, exclude_columns)


def rank_factors(factors: pl.DataFrame, 
                exclude_columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Convenience function to rank factors."""
    calculator = FactorCalculatorPL()
    return calculator.rank_factors(factors, exclude_columns)

