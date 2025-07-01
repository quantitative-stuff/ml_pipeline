
import polars as pl

def create_features(data: pl.DataFrame) -> pl.DataFrame:
    """
    Creates new features from the factor data.

    Args:
        data: A Polars DataFrame containing the factor data.

    Returns:
        A Polars DataFrame with the new features.
    """
    
    data = data.sort(['Symbol', 'Dates'])

    # Momentum features
    for col in data.columns:
        if col not in ['Symbol', 'Dates'] and data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            data = data.with_columns(
                pl.col(col).rolling_mean(window_size=5).over(['Symbol']).alias(f'{col}_mom5')
            )
            data = data.with_columns(
                pl.col(col).rolling_mean(window_size=20).over(['Symbol']).alias(f'{col}_mom20')
            )

    # Volatility features
    for col in data.columns:
        if col not in ['Symbol', 'Dates'] and data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            data = data.with_columns(
                pl.col(col).rolling_std(window_size=5).over(['Symbol']).alias(f'{col}_vol5')
            )
            data = data.with_columns(
                pl.col(col).rolling_std(window_size=20).over(['Symbol']).alias(f'{col}_vol20')
            )

    # Interaction features
    if 'size' in data.columns and 'f_value' in data.columns:
        data = data.with_columns(
            (pl.col('size') * pl.col('f_value')).alias('size_x_value')
        )

    return data.fill_null(0)

