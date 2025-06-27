"""
Comprehensive Factor Library
============================

This module combines all factors from:
- utils_ver33.py (fundamental factors)
- guru_get_dbdata.ipynb (market factors)
- DB_mlq_get_dbdata.ipynb (MLQ factors)
- 다이나믹포트 팩터 데이터 계산.ipynb (dynamic portfolio factors)
- factor_analysis_zk.ipynb and factor_analysis_mlq.ipynb (analysis factors)

Usage:
    from factor_library import FactorLibrary
    
    # Initialize the library
    fl = FactorLibrary()
    
    # Get all available factors
    all_factors = fl.get_all_factors()
    
    # Get specific factor groups
    fundamental_factors = fl.get_fundamental_factors()
    market_factors = fl.get_market_factors()
    dynamic_factors = fl.get_dynamic_factors()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings

class FactorLibrary:
    """
    A comprehensive library for accessing and managing all factors from the ML pipeline.
    """
    
    def __init__(self):
        """Initialize the factor library."""
        self._initialize_factor_groups()
    
    def _initialize_factor_groups(self):
        """Initialize all factor groups and their definitions."""
        
        # Fundamental factors from utils_ver33.py
        self.fundamental_factors = {
            'BP': 'Book-to-Price ratio',
            'AssetP': 'Asset-to-Capital ratio', 
            'SalesP': 'Sales-to-Capitalization ratio',
            'ROE': 'Return on Equity',
            'EBITDAY': 'EBITDA yield',
            'EY_12M': 'Earnings Yield (12-month)',
            'EPSG': 'Earnings Per Share Growth',
            'OPG': 'Operating Profit Growth',
            'DY': 'Dividend Yield',
            'InstituitionalBuy': 'Institutional Buy ratio',
            'Leverage': 'Financial Leverage',
            'OPM': 'Operating Profit Margin',
            'OP_M1M': 'Operating Profit Momentum (1-month)',
            'ReverseMomentum': 'Reverse Price Momentum',
            'PriceMomentum': 'Price Momentum'
        }
        
        # Market factors from notebooks
        self.market_factors = {
            'size': 'Market Size/Capitalization',
            'dividend': 'Dividend Yield',
            'priceMomentum': 'Price Momentum',
            'trading': 'Trading Volume',
            'investSentiment': 'Investment Sentiment',
            'epsF_p': 'Forward EPS to Price',
            'earningsOtmv': 'Earnings to Market Value',
            '52H/p_Mom': '52-week High to Price Momentum',
            '52L/p_Mom': '52-week Low to Price Momentum',
            'dtoequity': 'Debt to Equity',
            'deltaDtoEquityYoY': 'Year-over-Year Change in Debt to Equity',
            'stStevOLtStdev': 'Short-term Standard Deviation',
            'Vol_20D': '20-day Volatility',
            'Vol_120D': '120-day Volatility'
        }
        
        # Quality factors
        self.quality_factors = {
            'roeF_proxy': 'Forward ROE Proxy',
            'roeT_proxy': 'Trailing ROE Proxy', 
            'roeQ': 'Quarterly ROE',
            'GP_TQ': 'Gross Profit to Total Assets',
            'Assets_TQ': 'Asset Turnover'
        }
        
        # Momentum factors
        self.momentum_factors = {
            'roeFTMom': 'ROE Forward to Trailing Momentum',
            'epsFMom': 'Forward EPS Momentum',
            'salseFMom': 'Forward Sales Momentum',
            'roeFchg_proxy': 'Forward ROE Change Proxy',
            'slsAdjEarningsYOY': 'Sales Adjusted Earnings Year-over-Year',
            'sprsEarningsOearningsAvgPstdev': 'Earnings Spread',
            'AdjPrc': 'Adjusted Price Momentum'
        }
        
        # Value factors
        self.value_factors = {
            'salesOtmv': 'Sales to Market Value',
            'cashFlowOtmv': 'Cash Flow to Market Value',
            'equityOtmv': 'Equity to Market Value',
            'bpsF_p': 'Forward Book to Price',
            'bpsT_p': 'Trailing Book to Price',
            'spsF_p': 'Forward Sales to Price',
            'spsT_p': 'Trailing Sales to Price',
            'cfpsF_p': 'Forward Cash Flow to Price',
            'SE_TQ': 'Shareholders Equity to Total Assets',
            'MktCap_Comm_Pref': 'Market Capitalization'
        }
        
        # Growth factors
        self.growth_factors = {
            'epsFTMom': 'EPS Forward to Trailing Momentum',
            'earningsFTMom': 'Earnings Forward to Trailing Momentum',
            'earningsYOY': 'Earnings Year-over-Year Growth',
            'salesYOY': 'Sales Year-over-Year Growth',
            'cashFlowYOY': 'Cash Flow Year-over-Year Growth'
        }
        
        # Volatility factors
        self.volatility_factors = {
            'BETA_1Y': '1-Year Beta',
            'VOL_1Y': '1-Year Volatility'
        }
        
        # Dividend factors
        self.dividend_factors = {
            'DPS_Adj': 'Adjusted Dividend Per Share'
        }
        
        # Dynamic portfolio factors (selected from 다이나믹포트 팩터 데이터 계산.ipynb)
        self.dynamic_factors = {
            'BP': 'Book-to-Price ratio',
            'SalesP': 'Sales-to-Capitalization ratio',
            'EY_12M': 'Earnings Yield (12-month)',
            'OPG': 'Operating Profit Growth',
            'DY': 'Dividend Yield',
            'InstituitionalBuy': 'Institutional Buy ratio',
            'OP_M1M': 'Operating Profit Momentum (1-month)',
            'PriceMomentum': 'Price Momentum',
            'ReverseMomentum': 'Reverse Price Momentum'
        }
        
        # Factor groups for easy access
        self.factor_groups = {
            'fundamental': self.fundamental_factors,
            'market': self.market_factors,
            'quality': self.quality_factors,
            'momentum': self.momentum_factors,
            'value': self.value_factors,
            'growth': self.growth_factors,
            'volatility': self.volatility_factors,
            'dividend': self.dividend_factors,
            'dynamic': self.dynamic_factors
        }
    
    def get_all_factors(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available factors organized by category.
        
        Returns:
            Dictionary containing all factor groups
        """
        return self.factor_groups
    
    def get_fundamental_factors(self) -> Dict[str, str]:
        """Get fundamental factors from utils_ver33.py"""
        return self.fundamental_factors
    
    def get_market_factors(self) -> Dict[str, str]:
        """Get market-based factors from notebooks"""
        return self.market_factors
    
    def get_quality_factors(self) -> Dict[str, str]:
        """Get quality factors"""
        return self.quality_factors
    
    def get_momentum_factors(self) -> Dict[str, str]:
        """Get momentum factors"""
        return self.momentum_factors
    
    def get_value_factors(self) -> Dict[str, str]:
        """Get value factors"""
        return self.value_factors
    
    def get_growth_factors(self) -> Dict[str, str]:
        """Get growth factors"""
        return self.growth_factors
    
    def get_volatility_factors(self) -> Dict[str, str]:
        """Get volatility factors"""
        return self.volatility_factors
    
    def get_dividend_factors(self) -> Dict[str, str]:
        """Get dividend factors"""
        return self.dividend_factors
    
    def get_dynamic_factors(self) -> Dict[str, str]:
        """Get dynamic portfolio factors"""
        return self.dynamic_factors
    
    def get_factor_list(self, group: str = None) -> List[str]:
        """
        Get a list of factor names.
        
        Args:
            group: Specific factor group to get. If None, returns all factors.
            
        Returns:
            List of factor names
        """
        if group is None:
            # Return all factors
            all_factors = []
            for group_factors in self.factor_groups.values():
                all_factors.extend(list(group_factors.keys()))
            return all_factors
        elif group in self.factor_groups:
            return list(self.factor_groups[group].keys())
        else:
            raise ValueError(f"Unknown factor group: {group}. Available groups: {list(self.factor_groups.keys())}")
    
    def get_factor_description(self, factor_name: str) -> Optional[str]:
        """
        Get the description of a specific factor.
        
        Args:
            factor_name: Name of the factor
            
        Returns:
            Description of the factor, or None if not found
        """
        for group_factors in self.factor_groups.values():
            if factor_name in group_factors:
                return group_factors[factor_name]
        return None
    
    def get_factor_group(self, factor_name: str) -> Optional[str]:
        """
        Get the group that a factor belongs to.
        
        Args:
            factor_name: Name of the factor
            
        Returns:
            Group name, or None if not found
        """
        for group_name, group_factors in self.factor_groups.items():
            if factor_name in group_factors:
                return group_name
        return None
    
    def search_factors(self, keyword: str) -> Dict[str, List[str]]:
        """
        Search for factors by keyword in their names or descriptions.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            Dictionary mapping group names to lists of matching factors
        """
        results = {}
        keyword_lower = keyword.lower()
        
        for group_name, group_factors in self.factor_groups.items():
            matching_factors = []
            for factor_name, description in group_factors.items():
                if (keyword_lower in factor_name.lower() or 
                    keyword_lower in description.lower()):
                    matching_factors.append(factor_name)
            
            if matching_factors:
                results[group_name] = matching_factors
        
        return results
    
    def get_factor_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all factors.
        
        Returns:
            DataFrame with columns: factor_name, group, description
        """
        summary_data = []
        
        for group_name, group_factors in self.factor_groups.items():
            for factor_name, description in group_factors.items():
                summary_data.append({
                    'factor_name': factor_name,
                    'group': group_name,
                    'description': description
                })
        
        return pd.DataFrame(summary_data)
    
    def get_factor_count_by_group(self) -> pd.Series:
        """
        Get the count of factors in each group.
        
        Returns:
            Series with group names as index and counts as values
        """
        counts = {}
        for group_name, group_factors in self.factor_groups.items():
            counts[group_name] = len(group_factors)
        
        return pd.Series(counts)
    
    def get_common_factors(self, groups: List[str]) -> List[str]:
        """
        Get factors that appear in multiple specified groups.
        
        Args:
            groups: List of group names to check
            
        Returns:
            List of common factor names
        """
        if not groups:
            return []
        
        # Get factors from the first group
        common_factors = set(self.factor_groups[groups[0]].keys())
        
        # Find intersection with other groups
        for group in groups[1:]:
            if group in self.factor_groups:
                common_factors &= set(self.factor_groups[group].keys())
        
        return list(common_factors)
    
    def get_unique_factors(self, group: str) -> List[str]:
        """
        Get factors that are unique to a specific group (not found in other groups).
        
        Args:
            group: Group name to check
            
        Returns:
            List of unique factor names
        """
        if group not in self.factor_groups:
            return []
        
        group_factors = set(self.factor_groups[group].keys())
        other_factors = set()
        
        for other_group, other_group_factors in self.factor_groups.items():
            if other_group != group:
                other_factors.update(other_group_factors.keys())
        
        return list(group_factors - other_factors)


# Convenience functions for easy access
def get_all_factors() -> Dict[str, Dict[str, str]]:
    """Get all available factors."""
    fl = FactorLibrary()
    return fl.get_all_factors()

def get_fundamental_factors() -> Dict[str, str]:
    """Get fundamental factors."""
    fl = FactorLibrary()
    return fl.get_fundamental_factors()

def get_market_factors() -> Dict[str, str]:
    """Get market factors."""
    fl = FactorLibrary()
    return fl.get_market_factors()

def get_dynamic_factors() -> Dict[str, str]:
    """Get dynamic portfolio factors."""
    fl = FactorLibrary()
    return fl.get_dynamic_factors()

def get_factor_summary() -> pd.DataFrame:
    """Get factor summary DataFrame."""
    fl = FactorLibrary()
    return fl.get_factor_summary()

def search_factors(keyword: str) -> Dict[str, List[str]]:
    """Search for factors by keyword."""
    fl = FactorLibrary()
    return fl.search_factors(keyword)


if __name__ == "__main__":
    # Example usage
    fl = FactorLibrary()
    
    print("=== Factor Library Summary ===")
    print(f"Total factor groups: {len(fl.factor_groups)}")
    print(f"Total factors: {len(fl.get_factor_list())}")
    
    print("\n=== Factor Count by Group ===")
    print(fl.get_factor_count_by_group())
    
    print("\n=== Dynamic Portfolio Factors ===")
    for factor, desc in fl.get_dynamic_factors().items():
        print(f"{factor}: {desc}")
    
    print("\n=== Factor Summary ===")
    summary = fl.get_factor_summary()
    print(summary.head(10))
    
    print("\n=== Search for 'momentum' factors ===")
    momentum_results = fl.search_factors('momentum')
    for group, factors in momentum_results.items():
        print(f"{group}: {factors}") 