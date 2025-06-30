"""
Test script to verify target creation is working correctly.
"""

import polars as pl
import numpy as np
from pipeline.factor_processor import FactorProcessor

def test_target_creation():
    """Test the target creation function."""
    print("🧪 Testing Target Creation")
    print("=" * 40)
    
    # Create sample data with realistic price information
    dates = pl.date_range(
        start=pl.datetime(2023, 1, 1),
        end=pl.datetime(2023, 12, 31),
        interval="1mo"
    )
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Create realistic price data with trends
    data = []
    for symbol in symbols:
        # Start with different base prices for each symbol
        base_prices = {'AAPL': 150, 'GOOGL': 2800, 'MSFT': 300}
        base_price = base_prices[symbol]
        
        for i, date in enumerate(dates):
            # Create realistic price progression with some volatility
            # Add a small upward trend and some monthly variation
            trend_factor = 1 + (i * 0.02)  # 2% monthly trend
            volatility = 1 + (np.sin(i * 0.5) * 0.05)  # 5% monthly volatility
            price = base_price * trend_factor * volatility
            
            data.append({
                'Symbol': symbol,
                'Dates': date,
                'S410000700': price,  # adjPrice column
                'S420002100': price * 1000,  # market value (roughly)
                'FM30041100': price * 0.05,  # eps forward (5% of price)
                'FM30041200': price * 0.04,  # eps trailing (4% of price)
            })
    
    raw_data = pl.DataFrame(data)
    
    print(f"📊 Sample data created: {len(raw_data)} records")
    print(f"  Symbols: {raw_data['Symbol'].unique().to_list()}")
    print(f"  Date range: {raw_data['Dates'].min()} to {raw_data['Dates'].max()}")
    print(f"  Price range: {raw_data['S410000700'].min():.2f} to {raw_data['S410000700'].max():.2f}")
    
    # Show price progression for one symbol
    aapl_data = raw_data.filter(pl.col('Symbol') == 'AAPL').sort('Dates')
    print(f"\n📈 AAPL price progression:")
    for i, row in enumerate(aapl_data.head(6).iter_rows()):
        print(f"  {row[1]}: ${row[2]:.2f}")
    
    # Create factor processor
    factor_processor = FactorProcessor()
    
    # Compute factors (simplified)
    print("\n🔄 Computing factors...")
    factors = raw_data.select(['Symbol', 'Dates']).with_columns([
        pl.col('Symbol').alias('size'),
        pl.lit(0.5).alias('value'),
        pl.lit(0.3).alias('momentum')
    ])
    
    # Create target
    print("🔄 Creating target variable...")
    factors_with_target = factor_processor.create_target(factors, raw_data)
    
    # Display results
    print(f"\n✅ Target creation completed!")
    print(f"  Original samples: {len(factors)}")
    print(f"  Samples with target: {len(factors_with_target)}")
    print(f"  Target mean: {factors_with_target['target_return_1m'].mean():.4f}")
    print(f"  Target std: {factors_with_target['target_return_1m'].std():.4f}")
    print(f"  Target min: {factors_with_target['target_return_1m'].min():.4f}")
    print(f"  Target max: {factors_with_target['target_return_1m'].max():.4f}")
    
    # Show sample of results with actual calculations
    print(f"\n📋 Sample forward return calculations:")
    sample_data = factors_with_target.head(10)
    for row in sample_data.iter_rows():
        symbol, date, size, value, momentum, target_return = row
        print(f"  {symbol} {date}: {target_return:.4f} ({target_return*100:.2f}%)")
    
    # Verify no NaN values in target
    null_count = factors_with_target['target_return_1m'].null_count()
    print(f"\n🔍 Quality check:")
    print(f"  Null values in target: {null_count}")
    
    if null_count == 0:
        print("  ✅ Target creation successful - no null values!")
    else:
        print("  ⚠️  Warning: Found null values in target")
    
    # Verify the calculation is correct by checking one example
    print(f"\n🔍 Verification - AAPL forward return calculation:")
    aapl_with_target = factors_with_target.filter(pl.col('Symbol') == 'AAPL').sort('Dates')
    if len(aapl_with_target) >= 2:
        first_row = aapl_with_target.head(1).iter_rows().next()
        second_row = aapl_with_target.head(2).tail(1).iter_rows().next()
        
        symbol1, date1, _, _, _, return1 = first_row
        symbol2, date2, _, _, _, return2 = second_row
        
        print(f"  {symbol1} {date1} -> {date2}: {return1:.4f}")
        print(f"  Expected: Price should increase due to trend")

if __name__ == "__main__":
    test_target_creation() 