import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.dlg import DLG
from flcore.datasets.rul_factory import RULDatasetFactory
from flcore.servers.serverbase import Server


class Server_RUL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_dataset = RULDatasetFactory.create_dataset(
                dataset_name=self.dataset,
                client_id=i,
                mode='train',
                **self.args.__dict__,
            )
            test_dataset = RULDatasetFactory.create_dataset(
                dataset_name=self.dataset,
                client_id=i,
                mode='test',
                **self.args.__dict__,
            )
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_dataset), 
                            test_samples=len(test_dataset), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)