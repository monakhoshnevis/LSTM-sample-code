# -*- coding: utf-8 -*-
"""

Neural-Kinematic Models

This module contains deep learning models designed for processing and predicting kinematic data
from neural signals.

"""

__all__ = ['KinematicLinRNN', 'KinematicLSTM', 'Conv1DKinematicModel']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class KinematicLinRNN(nn.Module):
    """
    A flexible RNN-based model designed for processing kinematic sequences from neural data.
    
    Components:
    - Input linear layer for feature transformation
    - Recurrent layer (RNN, LSTM, or GRU) for sequence modeling
    - Output linear layer for kinematic predictions (e.g., pos, vel)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_rnn_layers: int = 1,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.0,
        load_weight_file: str = None,
    ):
        """
        Initializes the kinematic RNN model.

        Args:
            input_size (int): Number of input features 
            output_size (int): Number of output features 
            hidden_size (int): Hidden size of the recurrent layer.
            num_rnn_layers (int): Number of recurrent layers.
            rnn_type (str): Type of recurrent unit ('rnn', 'lstm', 'gru').
            bidirectional (bool): If True, uses a bidirectional RNN.
            dropout (float): Dropout rate for regularization.
            load_weight_file (str, optional): Path to load pre-trained weights.
        """
        super().__init__()
        
        self.num_layers = num_rnn_layers
        self.hidden_dim = hidden_size
        
        rnn_classes = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if rnn_type.lower() not in rnn_classes:
            raise ValueError(f"Invalid rnn_type '{rnn_type}'. Choose from {list(rnn_classes.keys())}.")
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.rnn = rnn_classes[rnn_type.lower()] (
            hidden_size, hidden_size, num_rnn_layers, bidirectional=bidirectional,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = nn.Dropout(dropout)

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, init_state=None):
        """
        Forward pass for the kinematic RNN model.

        Args:
            x (Tensor): Input tensor of shape (sequence_length, input_size).
            init_state (Tensor, optional): Initial hidden state for the recurrent layer.

        Returns:
            Tensor: Output tensor with shape (sequence_length, output_size).
        """
        x = self.dropout(F.relu(self.input_layer(x)))
        x, _ = self.rnn(x, init_state)
        lstm_out = x[:, -1, :]
        x = self.output_layer(lstm_out)
        
        return x


class KinematicLSTM(nn.Module):
    """
    A simple LSTM-based model for predicting kinematic sequences from time-series data.
    
    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): LSTM hidden size.
        output_dim (int): Number of output features.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate for regularization.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super(KinematicLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM model."""
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(lstm_out)


class Conv1DKinematicModel(nn.Module):
    """
    A 1D CNN designed for processing time-series neural- kinematic data 
    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_dim (int): Number of convolutional filters.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=32, kernel_size=3, stride=1, padding=1):
        super(Conv1DKinematicModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.global_avg_pool(x).squeeze(-1)
        return self.fc(x)
