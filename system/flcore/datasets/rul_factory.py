from .fedcmapss import FedCMAPSSDataset, FedCMAPSSWindowDataset

class RULDatasetFactory:
    @staticmethod
    def create_dataset(dataset_name, client_id, mode, **kwargs):
        window_size = kwargs.get('window_size', None)
        stride = kwargs.get('stride', 1)

        if dataset_name == "FedCMAPSS":
            dataset = FedCMAPSSDataset(
                data_root=kwargs.get('data_root'),
                task_id=kwargs.get('task'),
                split_id=kwargs.get('split'),
                client_id=client_id,
                mode=mode
            )
            if window_size is not None:
                return FedCMAPSSWindowDataset(dataset, window_size, stride)
            return dataset
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")
