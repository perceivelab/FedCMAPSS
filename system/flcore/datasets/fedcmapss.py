import pandas as pd
import os
import json
import torch
from torch.utils.data import Dataset
from .rul_dataset import RULDataset


class FedCMAPSSDataset(RULDataset):
    @staticmethod
    def get_num_clients(data_root, task_id, split_id):
        # Load splits configuration
        with open(os.path.join(data_root, 'tasks.json'), 'r') as f:
            tasks = json.load(f)
        return len(tasks[str(task_id)][str(split_id)]['train'])
    
    def __init__(self, data_root, task_id, split_id, client_id, mode='train', transform=None, target_transform=None, return_full_rul=False):
        """
        mode:
        - 'train': Standard CMAPSS training data
        - 'test': Standard CMAPSS test data
        - 'test_full': Held-out training data used as test data for RUL trajectory evaluation ONLY
        return_full_rul:
        - If True, returns the full RUL trajectory (vector of size T) for each sequence.
        - If False, returns only the final RUL value (scalar) for each sequence
        """
        super().__init__()
        self.mode = mode
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.return_full_rul = return_full_rul
        # Load splits configuration
        with open(os.path.join(data_root, 'tasks.json'), 'r') as f:
            tasks = json.load(f)
        # Access the specific split configuration
        self.client_data = tasks[str(task_id)][str(split_id)][mode][str(client_id)]
        # Load the appropriate data file
        if mode == 'train' or mode == 'test_full':
            data_file = 'cmapss_processed_train_data.csv'
        elif mode == 'test':
            data_file = 'cmapss_processed_test_data.csv'
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.df = pd.read_csv(os.path.join(data_root, data_file))
        # Define feature columns (sensors and settings)
        self.feature_cols = ['cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                            [f'sensor_{i}' for i in range(1, 22)]
        # Initialize and fill sequences and labels
        self.sequences = []
        self.labels = []
        self._read_data()

    def _read_data(self):
        # Process each selected unit
        for entry in self.client_data:
            fd, unit, condition = entry
            # Filter data by FD and Unit
            mask = (self.df['fd'] == fd) & (self.df['unit'] == unit)
            # Filter by condition if specified (condition != -1)
            if condition != -1:
                mask &= (self.df['condition'] == int(condition))
            unit_df = self.df[mask]
            if len(unit_df) == 0:
                print(f"Warning: No data found for FD={fd}, Unit={unit}, Condition={condition} in mode={self.mode}")
                continue
            # Extract sequence features
            sequence = torch.tensor(unit_df[self.feature_cols].values, dtype=torch.float32)
            self.sequences.append(sequence)
            # Extract RUL labels (vector of size T)
            label = torch.tensor(unit_df['rul'].values, dtype=torch.float32)
            self.labels.append(label)

    def stats(self):
        """
        Compute mean and standard deviation of features.
        """
        if not self.sequences:
             return None, None
        all_data = torch.cat(self.sequences, dim=0)
        mean = torch.mean(all_data, dim=0)
        std = torch.std(all_data, dim=0)
        return mean, std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Get item data
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Apply transforms if any
        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            label = self.target_transform(label)
        # Remove NaN values if any
        sequence = torch.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        return sequence, label if self.return_full_rul else label[-1]

class FedCMAPSSWindowDataset(RULDataset):
    def __init__(self, dataset, window_size, last_window_only=False):
        """
        Wraps a FedCMAPSSDataset to extract windows.
        """
        super().__init__()
        self.dataset = dataset
        self.stride = 1
        self.window_size = window_size
        self.last_window_only = last_window_only
        self.windows = []
        self._build_index()

    def _build_index(self):
        if not self.last_window_only:
            # Process each sequence
            for seq_idx, sequence in enumerate(self.dataset.sequences):
                # Check length
                seq_len = sequence.shape[0]
                if seq_len < self.window_size:
                    continue
                # Calculate number of windows
                num_windows = (seq_len - self.window_size) // self.stride + 1
                # Store window start indices
                for i in range(num_windows):
                    start_idx = i * self.stride
                    self.windows.append((seq_idx, start_idx))
        else:
            # Only last window of each sequence
            for seq_idx, sequence in enumerate(self.dataset.sequences):
                seq_len = sequence.shape[0]
                if seq_len < self.window_size:
                    continue
                start_idx = seq_len - self.window_size
                self.windows.append((seq_idx, start_idx))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Get window info
        seq_idx, start_idx = self.windows[idx]
        end_idx = start_idx + self.window_size
        # Extract sequence window
        sequence_window = self.dataset.sequences[seq_idx][start_idx:end_idx]
        # Extract label window
        label_window = self.dataset.labels[seq_idx][start_idx:end_idx]
        # Apply transforms from the wrapped dataset
        if self.dataset.transform:
            sequence_window = self.dataset.transform(sequence_window)
        if self.dataset.target_transform:
            label_window = self.dataset.target_transform(label_window)
        # Remove NaN values if any
        sequence_window = torch.nan_to_num(sequence_window, nan=0.0, posinf=0.0, neginf=0.0)
        # Return
        return sequence_window, label_window if self.dataset.return_full_rul else label_window[-1]