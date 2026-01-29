import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.dlg import DLG
from flcore.datasets.rul_dataset_factory import RULDatasetFactory
from flcore.servers.serverbase import Server


class Server_RUL(Server):
    def __init__(self, args, times):
        # Get task info
        num_clients = RULDatasetFactory.get_num_clients(
            dataset_name=args.dataset,
            data_root=args.data_root,
            task_id=args.task,
            split_id=args.split,
        )
        args.num_clients = num_clients
        super().__init__(args, times)

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # Create dummy dataset to get split sizes
            dummy_train_dataset = RULDatasetFactory.create_dataset(
                dataset_name=self.dataset,
                mode='train',
                client_id=i,
                args=self.args,
            )
            dummy_test_dataset = RULDatasetFactory.create_dataset(
                dataset_name=self.dataset,
                mode='test',
                client_id=i,
                args=self.args,
            )
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(dummy_train_dataset), 
                            test_samples=len(dummy_test_dataset), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
    
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        # Gather metrics from all clients
        client_metrics = {'num_samples': [], 'rmse': [], 'nasa_score': [], 'client_ids': []}
        for c in self.clients:
            metrics = c.test_metrics()
            client_metrics['client_ids'].append(c.id)
            client_metrics['num_samples'].append(metrics['num_samples'])
            client_metrics['rmse'].append(metrics['rmse']*metrics['num_samples'])
            client_metrics['nasa_score'].append(metrics['nasa_score'])
        return client_metrics

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        # Gather metrics from all clients
        client_metrics = {'num_samples': [], 'mse_loss': [], 'rmse': [], 'nasa_score': [], 'client_ids': []}
        for c in self.clients:
            metrics = c.train_metrics()
            client_metrics['client_ids'].append(c.id)
            client_metrics['num_samples'].append(metrics['num_samples'])
            client_metrics['mse_loss'].append(metrics['mse_loss'])
            client_metrics['rmse'].append(metrics['rmse']*metrics['num_samples'])
            client_metrics['nasa_score'].append(metrics['nasa_score'])
        return client_metrics
        
    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        # Compute averaged metrics
        train_mse_loss = sum(stats_train['mse_loss']) * 1.0 / sum(stats_train['num_samples'])
        train_rmse = sum(stats_train['rmse']) * 1.0 / sum(stats_train['num_samples'])
        train_nasa_score = sum(stats_train['nasa_score'])
        test_rmse = sum(stats['rmse']) * 1.0 / sum(stats['num_samples'])
        test_nasa_score = sum(stats['nasa_score'])

        print("Averaged Train Loss: {:.4f}".format(train_mse_loss))
        print("Averaged Train RMSE: {:.4f}".format(train_rmse))
        print("Averaged Train NASA Score: {:.4f}".format(train_nasa_score))
        print("Averaged Test RMSE: {:.4f}".format(test_rmse))
        print("Averaged Test NASA Score: {:.4f}".format(test_nasa_score))
        # self.print_(test_acc, train_acc, train_loss)
        #print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        #print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    # TODO fix this
    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))