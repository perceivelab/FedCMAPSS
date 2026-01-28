import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client
from flcore.datasets.rul_factory import RULDatasetFactory


class Client_RUL(Client):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.data_root = args.data_root
        self.split = args.split
        self.device = args.device

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_dataset = RULDatasetFactory.create_dataset(
            dataset_name=self.dataset,
            data_root=self.data_root,
            client_id=self.id,
            split=self.split,
            mode='train'
        )
        return DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_dataset = RULDatasetFactory.create_dataset(
            dataset_name=self.dataset,
            data_root=self.data_root,
            client_id=self.id,
            split=self.split,
            mode='test',
        )
        return DataLoader(test_dataset, batch_size, drop_last=False, shuffle=True)
    
    def load_test_data_full(self):
        test_dataset = RULDatasetFactory.create_dataset(
            dataset_name=self.dataset,
            data_root=self.data_root,
            client_id=self.id,
            split=self.split,
            mode='test_full',
        )
        return DataLoader(test_dataset, batch_size=len(test_dataset), drop_last=False, shuffle=False)
        
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_num = 0
        mse = 0
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                mse += nn.MSELoss(reduction='sum')(output, y).item()
                test_num += y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        rmse = np.sqrt(mse / test_num)
        
        return rmse, test_num, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        mse = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                
                mse += nn.MSELoss(reduction='sum')(output, y).item()
                train_num += y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        rmse = np.sqrt(mse / train_num)

        return rmse, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
