
from .model_pipeline import ModelPipeline

def train_model(model_type: str, model_name: str, data_path: str):
    """
    Trains a model using the specified pipeline.

    Args:
        model_type: 'ml' or 'dl'
        model_name: The name of the model to train.
        data_path: The path to the training data.
    """
    pipeline = ModelPipeline(model_type, model_name, data_path)
    pipeline.run()
    return pipeline

if __name__ == '__main__':
    # Example usage
    data_path = '../../factor_creation/all_factors.parquet'
    
    import os
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please run the factor creation pipeline first.")
    else:
        # Train a machine learning model
        print("Training Linear Regression model...")
        ml_pipeline = train_model('ml', 'linear_regression', data_path)
        print("Linear Regression model trained.")

        # Train a deep learning model
        print("Training DNN model...")
        dl_pipeline = train_model('dl', 'dnn', data_path)
        print("DNN model trained.")
