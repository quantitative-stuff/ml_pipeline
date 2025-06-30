
from .model_pipeline import ModelPipeline
from .data_loader import load_factor_data
import polars as pl

def predict_returns(pipeline: ModelPipeline, data_path: str) -> pl.DataFrame:
    """
    Predicts future returns using a trained model pipeline.

    Args:
        pipeline: The trained model pipeline.
        data_path: The path to the data to predict on.

    Returns:
        A Polars DataFrame with the predictions.
    """
    data = load_factor_data(data_path)
    predictions = pipeline.predict(data)
    
    data = data.with_columns(pl.Series("predicted_return", predictions.flatten()))
    return data

if __name__ == '__main__':
    # This is an example of how to use the predict_returns function.
    # You would typically run this after training a model.
    print("To run predictions, you need a trained pipeline.")
    print("Example: predictions = predict_returns(ml_pipeline, 'path/to/your/data.parquet')")
