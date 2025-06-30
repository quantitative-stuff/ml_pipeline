
import polars as pl

def load_factor_data(path: str) -> pl.DataFrame:
    """
    Loads factor data from a Parquet file.

    Args:
        path: The path to the Parquet file.

    Returns:
        A Polars DataFrame containing the factor data.
    """
    return pl.read_parquet(path)
