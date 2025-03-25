# Neural-Kinematic Mapping

This repository contains a pipeline for training deep learning models to map neural signals to kinematic outputs. It includes dataset preparation, model architectures, and training scripts.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Models](#models)
- [Training](#training)
- [Results and Saving](#results-and-saving)

## Overview

This project implements a neural-kinematic mapping system using deep learning. The goal is to process neural activity sequences and predict corresponding kinematic values, including position, velocity, and joint angles.

## Installation

Ensure you have Python installed and install dependencies:

- PyTorch
- NumPy
- Pandas
- Matplotlib
- torchinfo

## Dataset Preparation

The dataset consists of neural recordings and their corresponding kinematic outputs stored in CSV format.

### Dataset Structure

Neural and kinematic data should be stored in separate folders:

```
Data/
  ├── Neural/
  │     ├── session1.csv
  │     ├── session2.csv
  │     └── ...
  ├── Kinematics/
  │     ├── session1.csv
  │     ├── session2.csv
  │     └── ...
```

Each CSV file should contain time-series data, with timestamps aligned between neural and kinematic recordings.

### Dataset Loader

The dataset is managed using `dataset_create.py`, which defines:

- `SplitNeuralKinematicDataset`: Separates kinematic outputs into position, velocity, and joint angles.
- `NeuralKinematicDataset`: Loads data in a standard format for training.

## Models

The `models.py` file defines multiple architectures:

- `KinematicLinRNN`: An RNN-based model (LSTM/GRU/RNN options available).
- `KinematicLSTM`: A dedicated LSTM-based sequence-to-sequence model.
- `Conv1DKinematicModel`: A 1D CNN for processing time-series data.

## Training

To train a model, run:

```sh
python train.py
```

This script performs:

1. **Data Loading:** Loads the neural and kinematic datasets.
2. **Model Selection:** Chooses between RNN, LSTM, or CNN architectures.
3. **Training:** Uses Adam optimizer and MSE loss.
4. **Validation:** Evaluates model performance at each epoch.
5. **Live Plotting:** Displays real-time loss updates.

Training hyperparameters can be adjusted in `train.py`:

```python
num_epochs = 50
learning_rate = 0.0001
weight_decay = 0.00001
batch_size = 32
sequence_length = 32
```

## Results and Saving

- Model checkpoints are saved in `TrainedModels/`.
- Training loss and validation loss are logged and exported to CSV.
- The trained model is also converted to ONNX format for inference compatibility.

##

