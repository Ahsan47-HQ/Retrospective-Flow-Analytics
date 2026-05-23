import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFlowCNN(nn.Module):
    def __init__(self, config):
        super(DeepFlowCNN, self).__init__()
        
        num_sensors = len(config['data']['sensor_cols'])
        num_metadata = len(config['data']['metadata_cols']) + 1 # +1 for Gender
        num_classes = config['model']['num_classes']
        n_filters = config['model']['num_filters']
        
        # 1. CNN Backbone for temporal sensor data
        self.conv_layers = nn.Sequential(
            nn.BatchNorm1d(num_sensors),
            
            nn.Conv1d(num_sensors, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )
        
        # Global pooling to handle variable window sizes if needed, 
        # but here we'll use a fixed size to calculate linear input
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. Fusion & Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(n_filters + num_metadata, config['model']['dense_units']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate_dense']),
            nn.Linear(config['model']['dense_units'], num_classes)
        )

    def forward(self, x, meta):
        # x shape: (batch, window_size, channels) -> (batch, channels, window_size) for Conv1D
        x = x.transpose(1, 2)
        
        features = self.conv_layers(x)
        features = self.adaptive_pool(features).squeeze(-1)
        
        # Concatenate metadata
        combined = torch.cat((features, meta), dim=1)
        
        logits = self.classifier(combined)
        return logits
