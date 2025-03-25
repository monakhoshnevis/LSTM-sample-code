"""
Creating a dataset module to handle neural and kinematic data for LSTM-based movement decoding. 
This dataset constructs sequences of neural activity and their corresponding kinematics.

*******    IMPORTANT **********
This dataset does not scale or normalize the data. For advanced preprocessing,
consider normalization techniques such as z-scoring or min-max scaling.

"""

__all__ = ['SplitNeuralKinematicDataset', 'NeuralKinematicDataset']

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SplitNeuralKinematicDataset(Dataset):
    def __init__(self, neural_folder, kinematic_folder, sequence_length=50):
        """
        Dataset for Neural to Kinematic mapping.
        This dataset class separates the kinematic targets into position, velocity, and joint angles.
        
        Args:
            neural_folder (str): Path to folder containing neural data CSV files.
            kinematic_folder (str): Path to folder containing corresponding kinematic CSV files.
            sequence_length (int): Number of timesteps in each sequence.
        """
        self.sequence_length = sequence_length
        self.neural_files = sorted([os.path.join(neural_folder, f) for f in os.listdir(neural_folder) if f.endswith('.csv')])
        self.kinematic_files = sorted([os.path.join(kinematic_folder, f) for f in os.listdir(kinematic_folder) if f.endswith('.csv')])

        assert len(self.neural_files) == len(self.kinematic_files), "Mismatch between Neural and Kinematic files!"

        self.sequences = self._load_data()
    
    def _load_data(self):
        """
        Loads all neural and kinematic data files and extracts sequences.

        Returns:
            list of tuples: [(neural_sequence, pos_target, vel_target, ang_target), ...]
        """
        all_sequences = []

        for neural_file, kinematic_file in zip(self.neural_files, self.kinematic_files):
            # Load Neural and Kinematic data
            neural_data = pd.read_csv(neural_file)
            kinematic_data = pd.read_csv(kinematic_file)

            # Extract position, velocity, and angle columns dynamically
            pos_cols = [col for col in kinematic_data.columns if col.startswith("pos_")]
            vel_cols = [col for col in kinematic_data.columns if col.startswith("vel_")]
            ang_cols = [col for col in kinematic_data.columns if col.startswith("ang_")]

            neural_array = neural_data.to_numpy(dtype=np.float32)
            pos_array = kinematic_data[pos_cols].to_numpy(dtype=np.float32)
            vel_array = kinematic_data[vel_cols].to_numpy(dtype=np.float32)
            ang_array = kinematic_data[ang_cols].to_numpy(dtype=np.float32)

            # Ensure data length consistency
            min_length = min(len(neural_array), len(pos_array), len(vel_array), len(ang_array))
            neural_array, pos_array, vel_array, ang_array = neural_array[:min_length], pos_array[:min_length], vel_array[:min_length], ang_array[:min_length]

            # Print session details for debugging
            print(f"{os.path.basename(neural_file)} | Neural: {neural_data.shape} | Kinematic: {kinematic_data.shape}")

            # Create sequences
            for i in range(len(neural_array) - self.sequence_length):
                neural_seq = neural_array[i:i + self.sequence_length]
                pos_target = pos_array[i + self.sequence_length - 1]  # Last timestep
                vel_target = vel_array[i + self.sequence_length - 1]  # Last timestep
                ang_target = ang_array[i + self.sequence_length - 1]  # Last timestep

                all_sequences.append((neural_seq, pos_target, vel_target, ang_target))

        return all_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        neural_seq, pos_target, vel_target, ang_target = self.sequences[idx]

        # Convert to torch tensors
        neural_seq = torch.tensor(neural_seq, dtype=torch.float32)
        pos_target = torch.tensor(pos_target, dtype=torch.float32)
        vel_target = torch.tensor(vel_target, dtype=torch.float32)
        ang_target = torch.tensor(ang_target, dtype=torch.float32)

        return neural_seq, pos_target, vel_target, ang_target


class NeuralKinematicDataset(Dataset):
    """
    A PyTorch Dataset to load neural and kinematic data for training an LSTM model.
    
    Each sample consists of:
    - An input sequence of length `sequence_length` from the neural data (shape: S × N_in)
    - A single target frame from the kinematic data corresponding to the next timestep (shape: N_out)
    """

    def __init__(self, neural_folder: str, kinematic_folder: str, sequence_length: int = 50):
        """
        Initializes the dataset by loading and processing neural and kinematic data.

        Args:
            neural_folder (str): Path to the folder containing neural CSV files.
            kinematic_folder (str): Path to the folder containing kinematic CSV files.
            sequence_length (int): Number of timesteps in each input sequence.
        """
        self.sequence_length = sequence_length
        self.sequences = []  # List to store (input_seq, target) pairs

        session_files = sorted(os.listdir(neural_folder))

        for session_name in session_files:
            neural_path = os.path.join(neural_folder, session_name)
            kinematic_path = os.path.join(kinematic_folder, session_name)

            neural_data = pd.read_csv(neural_path).values  # Shape: (T, N_in)
            kinematic_data = pd.read_csv(kinematic_path).values  # Shape: (T, N_out)

            min_len = min(len(neural_data), len(kinematic_data))
            neural_data, kinematic_data = neural_data[:min_len], kinematic_data[:min_len]

            neural_data = torch.tensor(neural_data, dtype=torch.float32)  # (T, N_in)
            kinematic_data = torch.tensor(kinematic_data, dtype=torch.float32)  # (T, N_out)

            print(f"{session_name} | Neural: {neural_data.shape} | Kinematic: {kinematic_data.shape}")

            for i in range(min_len - sequence_length):
                input_seq = neural_data[i:i + sequence_length]  # Shape: (S, N_in)
                target = kinematic_data[i + sequence_length]  # Shape: (N_out,)
                self.sequences.append((input_seq, target))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx]  # Returns tuple: (S, N_in), (N_out,)

# Example usage
if __name__ == "__main__":
    neural_folder = "../Data/Neural"
    kinematic_folder = "../Data/Kinematics"
    sequence_length = 50
    batch_size = 32

    dataset = SplitNeuralKinematicDataset(neural_folder, kinematic_folder, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        input_batch, pos_batch, vel_batch, ang_batch = batch
        print(f"Input batch: {input_batch.shape} Pos batch: {pos_batch.shape} Vel batch: {vel_batch.shape} Ang batch: {ang_batch.shape}")
        break



