import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client
from flcore.datasets.rul_dataset_factory import RULDatasetFactory
from flcore.datasets.rul_utils import PiecewiseScaler, compute_nasa_score


class Client_RUL(Client):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.data_root = args.data_root
        self.split = args.split
        self.device = args.device
        self.loss = nn.MSELoss()
        self.args = args

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_dataset = RULDatasetFactory.create_dataset(
            dataset_name=self.dataset,
            mode='train',
            client_id=self.id,
            args=self.args,
        )
        return DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_dataset = RULDatasetFactory.create_dataset(
            dataset_name=self.dataset,
            mode='test',
            client_id=self.id,
            args=self.args,
        )
        return DataLoader(test_dataset, batch_size, drop_last=False, shuffle=True)
    
    def load_test_data_full(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_dataset = RULDatasetFactory.create_dataset(
            dataset_name=self.dataset,
            data_root=self.data_root,
            client_id=self.id,
            split=self.split,
            mode='test_full',
        )
        return DataLoader(test_dataset, batch_size, drop_last=False, shuffle=False)
        
    def test_metrics(self):
        # Create piecewise scaler to invert predictions
        piecewise_scaler = PiecewiseScaler(max_rul=self.args.max_rul)
        # Process test set
        testloaderfull = self.load_test_data()
        self.model.eval()
        test_num = 0
        mse = 0
        nasa_score = 0
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                # Invert prediction (label should be in original scale)
                output = piecewise_scaler.inverse(output.detach())
                # Update MSE
                mse += nn.MSELoss(reduction='sum')(output, y).item()
                test_num += y.shape[0]
                # Update NASA score
                nasa_score += compute_nasa_score(y, output)
        # self.model.cpu()
        # self.save_model(self.model, 'model')
        # Compute average metrics
        rmse = np.sqrt(mse / test_num)
        return {
            'rmse': rmse,
            'nasa_score': nasa_score,
            'num_samples': test_num
        }

    def train_metrics(self):
        # Create piecewise scaler to invert predictions
        piecewise_scaler = PiecewiseScaler(max_rul=self.args.max_rul)
        # Process training set
        trainloader = self.load_train_data()
        self.model.eval()
        train_num = 0
        mse_loss = 0
        rmse = 0
        nasa_score = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                train_num += y.shape[0]
                # Update MSE loss
                mse_loss += nn.MSELoss(reduction='sum')(output, y).item()
                # Invert prediction and label
                output = piecewise_scaler.inverse(output.detach())
                y = piecewise_scaler.inverse(y)
                # Update RMSE
                rmse += nn.MSELoss(reduction='sum')(output, y).item()
                # Update NASA score
                nasa_score += compute_nasa_score(y, output)
        # self.model.cpu()
        # self.save_model(self.model, 'model')
        # Compute average metrics
        rmse = np.sqrt(mse_loss / train_num)
        mse_loss = mse_loss / train_num
        return {
            'mse_loss': mse_loss,
            'rmse': rmse,
            'nasa_score': nasa_score,
            'num_samples': train_num
        }
        # self.model = self.load_model('model')
        # self.model.to(self.device)

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
