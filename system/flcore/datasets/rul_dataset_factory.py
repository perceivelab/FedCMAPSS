from .fedcmapss import FedCMAPSSDataset, FedCMAPSSWindowDataset
from flcore.datasets.rul_utils import PiecewiseScaler

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
        if dataset_name == "FedCMAPSS":
            # Create training dataset first to get stats
            dataset = FedCMAPSSDataset(
                data_root=args.data_root,
                task_id=args.task,
                split_id=args.split,
                client_id=client_id,
                mode='train'
            )
            # Get stats
            mean, std = dataset.stats()
            # Create transforms
            dataset.transform = lambda x: (x - mean) / std
            dataset.target_transform = PiecewiseScaler(max_rul=args.max_rul)
            # Check mode and recreate dataset if needed
            if mode != 'train':
                dataset = FedCMAPSSDataset(
                    data_root=args.data_root,
                    task_id=args.task,
                    split_id=args.split,
                    client_id=client_id,
                    mode=mode,
                    transform=dataset.transform
                    # No target_transform for test/test_full sets
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
