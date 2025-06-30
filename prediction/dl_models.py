import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_shape):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class ResidualNet(nn.Module):
    def __init__(self, input_shape):
        super(ResidualNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.res_block1 = ResidualBlock(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res_block1(x)
        x = self.fc2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_shape, nhead=4, num_encoder_layers=2, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_shape, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(input_shape, 1)

    def forward(self, src):
        # PyTorch TransformerEncoder expects (batch, seq_len, features)
        # We'll treat our features as a sequence of length 1
        if src.dim() == 2:
            src = src.unsqueeze(1)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.decoder(output)
        return output

def get_dl_model(model_name: str, input_shape):
    """
    Returns a deep learning model based on the given name.

    Args:
        model_name: The name of the model to return.
        input_shape: The shape of the input data.

    Returns:
        A PyTorch model.
    """
    if model_name == 'dnn':
        return DNN(input_shape)
    elif model_name == 'residual':
        return ResidualNet(input_shape)
    elif model_name == 'transformer':
        return TransformerModel(input_shape)
    else:
        raise ValueError(f"Unknown model name: {model_name}")