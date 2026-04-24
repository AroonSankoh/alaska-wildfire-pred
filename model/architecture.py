import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class CNNEncoder(nn.Module):
    """
    The core spatial encoder of the hybrid wildfire detection model.
    """
    
    def __init__(self, input_dim, embedding_dim):
        """
        Initialization function that initializes convultional and linear layers 
        according to the input dimension and output (embedding) dimension.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=input_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3)
        self.flatten = nn.Flatten()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, X):
        """
        Forward propogation call for a 1D CNN spatial encoder.
        """
        # core conv layers
        X = self.conv1(X)
        X = F.relu(X)
        X = self.conv2(X)
        X = F.relu(X)
        # flatten and pool for linear output call
        X = self.pool(X)
        X = self.flatten(X)
        X = self.linear(X) 
        return X
    
class TransformerEncoder(nn.Module):
    """
    The core temporal encoder of the hybrid wildfire detection model.
    """

    def __init__(self, input_dim, embedding_dim, n_head, num_layers):
        """
        Initialization function that initializes transformer layers according to the 
        input dimension, output (embedding) dimension, number of attention heads, and number of layers.
        """
        super().__init__()
        self.trans_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_head)
        self.trans_encoder = nn.TransformerEncoder(self.trans_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, X):
        """
        Forward propogation call for a 2D transformer encoder.
        """
        X = self.trans_encoder(X.T.unsqueeze(1))
        # collapse sequence dimension by taking the mean
        X = X.mean(0)
        X = self.linear(X)
        return X
    
class WildfireModel(nn.Module):
    """
    The wildfire detection model which fuses the results from both encoders into a 
    multi-head output that describes risk over multiple time horizons.
    """
    def __init__(self, spatial_input_dim, temporal_input_dim, embedding_dim, n_layers, n_head=8):
        """
        Initialization function that initializes a hybrid conv-transformer wildfire prediction model.
        """
        super().__init__();
        self.spatial_encoder = CNNEncoder(spatial_input_dim, embedding_dim)
        self.temporal_encoder = TransformerEncoder(temporal_input_dim, embedding_dim, n_head, n_layers)
        self.head_1month = nn.Linear(embedding_dim * 2, 1)
        self.head_3month = nn.Linear(embedding_dim * 2, 1) 
        self.head_6month = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, x_spatial, x_temporal):
        """
        Forward propogation call for the full conv-transformer model. 
        """
        x_spatial = x_spatial.unsqueeze(0).unsqueeze(0)
        spatial_embeddings = self.spatial_encoder(x_spatial)
        temporal_embeddings = self.temporal_encoder(x_temporal)

        fused_output = torch.cat([spatial_embeddings, temporal_embeddings], dim=-1)

        head1 = F.sigmoid(self.head_1month(fused_output))
        head2 = F.sigmoid(self.head_3month(fused_output))
        head3 = F.sigmoid(self.head_6month(fused_output))

        return head1, head2, head3
    
    def cross_attention_fusion(self, spatial_embeddings, temporal_embeddings):
        """
        Implement cross-attention fusion between spatial and temporal embeddings so spatial embeddings 
        attend to the time series to learn which time steps are most important per tile.
        TODO: Implement the full function and replace simple concatenation.
        """
        raise NotImplementedError("RTC not yet implemented.")
    







