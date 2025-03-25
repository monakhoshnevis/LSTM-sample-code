"""
Training a model for neural-kinematic mapping.

"""

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import deque

from dataset_create import SplitNeuralKinematicDataset, NeuralKinematicDataset
from models import KinematicLSTM, KinematicLinRNN, Conv1DKinematicModel

def train_neural_kinematic_model(model: nn.Module, 
                                 train_loader: DataLoader, 
                                 val_loader: DataLoader, 
                                 num_epochs: int, 
                                 learning_rate: float,
                                 weight_decay: float,
                                 device: torch.device):
    """
    Trains the neural-kinematic model.

    Args:
        model (nn.Module): The deep learning model for mapping neural activity to kinematics.
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        weight_decay (float): Weight decay regularization.
        device (torch.device): Device to use (CPU or GPU).
    """
    # Move model to the specified device (GPU or CPU)
    model.to(device)

    # Define loss function for kinematic prediction (Mean Squared Error for now)
    criterion = nn.MSELoss()
    
    # Adam optimizer for stability in training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Track loss values
    train_losses = deque(maxlen=num_epochs)
    val_losses = deque(maxlen=num_epochs)

    epoch_durations = []
    start_time = time.time()

    # Live loss tracking setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Neural-Kinematic Model Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line_train, = ax.plot([], [], label="Train Loss")
    line_val, = ax.plot([], [], label="Validation Loss")
    ax.legend(loc='upper right')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        epoch_start_time = time.time()

        # Training step
        for neural_signals, kinematics, _, _ in train_loader:
            neural_signals, kinematics = neural_signals.to(device), kinematics.to(device)

            # Forward pass
            predictions = model(neural_signals)
            loss = criterion(predictions, kinematics)

            # Check for NaN loss
            if np.isnan(loss.item()):
                raise ValueError(f"Error in epoch {epoch}: loss is NaN!")

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for neural_signals, kinematics, _, _ in val_loader:
                neural_signals, kinematics = neural_signals.to(device), kinematics.to(device)
                predictions = model(neural_signals)
                loss = criterion(predictions, kinematics)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Epoch timing and estimated remaining time
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        remaining_time = (sum(epoch_durations) / len(epoch_durations)) * (num_epochs - (epoch + 1))
        remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

        # Update live plot
        line_train.set_data(range(len(train_losses)), train_losses)
        line_val.set_data(range(len(val_losses)), val_losses)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

        # Logging progress
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.2f} | "
              f"Val Loss: {avg_val_loss:.2f} | "
              f"Epoch duration: {epoch_duration:.1f}s | "
              f"Time Remaining: {remaining_time_str}")

    plt.ioff()
    plt.close(fig)

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {int(total_training_time // 3600)}:"
          f"{int((total_training_time % 3600) // 60)}:"
          f"{int((total_training_time % 3600) % 60)}.")

    return list(train_losses), list(val_losses), epoch_durations

if __name__ == '__main__':
    # Paths to neural and kinematic data
    train_neural_path = "../Data Processing/Processed_2/Train/Neural"
    train_kinematic_path = "../Data Processing/Processed_2/Train/Kinematic"
    val_neural_path = "../Data Processing/Processed_2/Val/Neural"
    val_kinematic_path = "../Data Processing/Processed_2/Val/Kinematic"

    # Data and model parameters
    sequence_length = 32
    num_epochs = 50
    learning_rate = 0.0001
    weight_decay = 0.00001
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = SplitNeuralKinematicDataset(neural_folder=train_neural_path, kinematic_folder=train_kinematic_path, sequence_length=sequence_length)
    val_dataset = SplitNeuralKinematicDataset(neural_folder=val_neural_path, kinematic_folder=val_kinematic_path, sequence_length=sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define input and output feature dimensions
    N_in = train_dataset.sequences[0][0].shape[1]
    N_out = train_dataset.sequences[0][1].shape[0]
    hidden_dim = 32
    num_layers = 2
    kernel_size = 4

    # Select model architecture
    model = KinematicLinRNN(input_size=N_in,
                   output_size=N_out,
                   hidden_size=hidden_dim,
                   num_rnn_layers=num_layers,
                   rnn_type="lstm",
                   bidirectional=False,
                   dropout=0.4,
                   load_weight_file=None)

    print(summary(model, input_size=(batch_size, sequence_length, N_in)))

    # Train the model
    train_loss, valid_loss, epoch_durations = train_neural_kinematic_model(
        model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, device)

    # Save training results
    df = pd.DataFrame({'epoch': range(1, num_epochs + 1), 'train_loss': train_loss,
                       'valid_loss': valid_loss, 'epoch_duration': epoch_durations})
    df.to_csv(f"TrainedModels/neural_kinematic_sequence{sequence_length}_hidden{hidden_dim}_layers{num_layers}.csv", index=False)

    # Save model as ONNX
    dummy_input = torch.randn(1, sequence_length, N_in).to(device)
    onnx_filename = f"TrainedModels/neural_kinematic_sequence{sequence_length}_hidden{hidden_dim}_layers{num_layers}.onnx"
    torch.onnx.export(model, dummy_input, onnx_filename, input_names=["input"], 
                      output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    print(f"Model saved as {onnx_filename}")

    # Plot training results
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_loss, label="Train Loss", marker=".")
    plt.plot(range(1, num_epochs + 1), valid_loss, label="Validation Loss", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Neural-Kinematic Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
