from .data_loader import load_factor_data
from .feature_engineering import create_features
from .ml_models import get_ml_model
from .dl_models import get_dl_model
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ModelPipeline:
    def __init__(self, model_type: str, model_name: str, data_path: str):
        self.model_type = model_type
        self.model_name = model_name
        self.data_path = data_path
        self.model = None

    def run(self):
        # Load data
        data = load_factor_data(self.data_path)

        # Create features
        data = create_features(data)

        # For simplicity, we'll use a placeholder for the target variable
        # In a real scenario, this would be the future return
        data = data.with_columns(pl.lit(np.random.rand(len(data))).alias("future_return"))

        # Split data into features and target
        features = data.drop(['ticker', 'date', 'future_return'])
        target = data['future_return']

        # Train model
        if self.model_type == 'ml':
            self.model = get_ml_model(self.model_name)
            self.model.fit(features.to_numpy(), target.to_numpy())
        elif self.model_type == 'dl':
            input_shape = features.shape[1]
            self.model = get_dl_model(self.model_name, input_shape)
            
            # Create a DataLoader
            train_dataset = TensorDataset(torch.tensor(features.to_numpy(), dtype=torch.float32), torch.tensor(target.to_numpy(), dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Define loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            # Training loop
            self.model.train()
            for epoch in range(10):
                for batch_features, batch_target in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_target.view(-1, 1))
                    loss.backward()
                    optimizer.step()

    def predict(self, data: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise Exception("Model has not been trained yet. Please run the pipeline first.")

        features = create_features(data)
        features = features.drop(['Symbol', 'Dates'])
        
        if self.model_type == 'ml':
            return self.model.predict(features.to_numpy())
        elif self.model_type == 'dl':
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.tensor(features.to_numpy(), dtype=torch.float32)).numpy()