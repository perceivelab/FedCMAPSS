from .fedcmapss import FedCMAPSSDataset, FedCMAPSSWindowDataset
from flcore.datasets.rul_utils import PiecewiseScaler
import copy

class RULDatasetFactory:
    @staticmethod
    def get_num_clients(dataset_name, task_id, split_id, **kwargs):
        if dataset_name == "FedCMAPSS" or dataset_name == "FedCMAPSSWindow":
            return FedCMAPSSDataset.get_num_clients(
                data_root=kwargs['data_root'],
                task_id=task_id,
                split_id=split_id
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")

    @staticmethod
    def get_input_size(dataset_name, **kwargs):
        if dataset_name == "FedCMAPSS" or dataset_name == "FedCMAPSSWindow":
            return 25
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")
        
    @staticmethod
    def create_dataset(dataset_name, mode, client_id, args):
        # Create scaler
        scaler = PiecewiseScaler(max_rul=args.max_rul, normalize=args.normalize_rul)
        if dataset_name == "FedCMAPSS" and args.centralized:
            # Get number of clients
            num_clients = FedCMAPSSDataset.get_num_clients(args.data_root, args.task, args.split)
            # Create training dataset using create_dataset for first client to get stats
            local_args = copy.deepcopy(args)
            local_args.centralized = False
            train_dataset = RULDatasetFactory.create_dataset(dataset_name, 'train', 0, local_args)
            # Create datasets for other clients and concatenate sequences and labels
            for cid in range(1, num_clients):
                other_dataset = RULDatasetFactory.create_dataset(dataset_name, 'train', cid, local_args)
                train_dataset.sequences.extend(other_dataset.sequences)
                train_dataset.labels.extend(other_dataset.labels)
            # Get stats
            mean, std = train_dataset.stats()
            # Create transforms
            train_dataset.transform = lambda x: (x - mean) / std
            train_dataset.target_transform = scaler
            # If mode is training, return the dataset
            if mode == 'train':
                return train_dataset
            # Create dataset for specified mode
            # Create dataset of first client
            dataset = RULDatasetFactory.create_dataset(dataset_name, mode, 0, local_args)
            # Concatenate datasets of other clients
            for cid in range(1, num_clients):
                other_dataset = RULDatasetFactory.create_dataset(dataset_name, mode, cid, local_args)
                dataset.sequences.extend(other_dataset.sequences)
                dataset.labels.extend(other_dataset.labels)
            # Set transforms
            dataset.transform = train_dataset.transform
            dataset.target_transform = train_dataset.target_transform
            # Return dataset
            return dataset
        elif dataset_name == "FedCMAPSS": # not centralized
            # Create training dataset first to get stats
            train_dataset = FedCMAPSSDataset(
                data_root=args.data_root,
                task_id=args.task,
                split_id=args.split,
                client_id=client_id,
                mode='train'
            )
            # Get stats
            mean, std = train_dataset.stats()
            # Create transforms
            train_dataset.transform = lambda x: (x - mean) / std
            train_dataset.target_transform = scaler
            # If mode is training, return the dataset
            if mode == 'train':
                return train_dataset
            # Create dataset for specified mode
            dataset = FedCMAPSSDataset(
                data_root=args.data_root,
                task_id=args.task,
                split_id=args.split,
                client_id=client_id,
                mode=mode,
                transform=train_dataset.transform,
                target_transform=train_dataset.target_transform
            )
            # Return dataset
            return dataset
        elif dataset_name == "FedCMAPSSWindow":
            # Create normal dataset first
            dataset = RULDatasetFactory.create_dataset(
                dataset_name="FedCMAPSS",
                mode=mode,
                client_id=client_id,
                args=args
            )
            # Return windowed dataset
            return FedCMAPSSWindowDataset(
                dataset=dataset,
                window_size=args.window_size,
                last_window_only=(mode == 'test') # only last window for test set
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")
