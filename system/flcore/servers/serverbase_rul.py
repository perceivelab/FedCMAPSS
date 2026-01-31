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
        self.results_history = []

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
        client_metrics = {'num_samples': [], 'mse_loss': [], 'weighted_mse_loss': [], 'sse': [], 'rmse': [], 'nasa_score': [], 'client_ids': []}
        for c in self.clients:
            metrics = c.test_metrics()
            client_metrics['client_ids'].append(c.id)
            client_metrics['num_samples'].append(metrics['num_samples'])
            client_metrics['mse_loss'].append(metrics['mse_loss'])  # Will not be averaged directly, to be used as a local metric
            client_metrics['weighted_mse_loss'].append(metrics['mse_loss'] * metrics['num_samples'])  # Used for weighted sum later
            client_metrics['rmse'].append(metrics['rmse'])
            client_metrics['sse'].append(metrics['sse'])
            client_metrics['nasa_score'].append(metrics['nasa_score'])
        return client_metrics

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        # Gather metrics from all clients
        client_metrics = {'num_samples': [], 'mse_loss': [], 'weighted_mse_loss': [], 'sse': [], 'rmse': [], 'nasa_score': [], 'client_ids': []}
        for c in self.clients:
            metrics = c.train_metrics()
            client_metrics['client_ids'].append(c.id)
            client_metrics['num_samples'].append(metrics['num_samples'])
            client_metrics['mse_loss'].append(metrics['mse_loss'])  # Will not be averaged directly, to be used as a local metric
            client_metrics['weighted_mse_loss'].append(metrics['mse_loss'] * metrics['num_samples'])  # Used for weighted sum later
            client_metrics['rmse'].append(metrics['rmse']) # Will not be averaged directly, to be used as a local metric
            client_metrics['sse'].append(metrics['sse']) # Cumulative SSE
            client_metrics['nasa_score'].append(metrics['nasa_score']) # Cumulative NASA score
        return client_metrics
        
    # evaluate selected clients
    def evaluate(self, round_idx=None):
        # Compute local stats
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        # Compute global metrics
        global_stats = {}
        global_stats['train_mse_loss'] = sum(stats_train['weighted_mse_loss']) / sum(stats_train['num_samples']) # Weighted average by number of samples
        global_stats['train_rmse'] = np.sqrt(sum(stats_train['sse']) / sum(stats_train['num_samples'])) # RMSE from cumulative SSE
        global_stats['train_nasa_score'] = sum(stats_train['nasa_score'])
        global_stats['test_mse_loss'] = sum(stats['weighted_mse_loss']) / sum(stats['num_samples']) # Weighted average by number of samples
        global_stats['test_rmse'] = np.sqrt(sum(stats['sse']) / sum(stats['num_samples'])) # RMSE from cumulative SSE
        global_stats['test_nasa_score'] = sum(stats['nasa_score'])
        # Save results
        self.results_history.append({
            'global_stats': global_stats,
            'local_stats_test': stats,
            'local_stats_train': stats_train
        })

        # Print global results
        print("Global Train Loss: {:.4f}".format(global_stats['train_mse_loss']))
        print("Global Train RMSE: {:.4f}".format(global_stats['train_rmse']))
        print("Global Train NASA Score: {:.4f}".format(global_stats['train_nasa_score']))
        print("Global Test Loss: {:.4f}".format(global_stats['test_mse_loss']))
        print("Global Test RMSE: {:.4f}".format(global_stats['test_rmse']))
        print("Global Test NASA Score: {:.4f}".format(global_stats['test_nasa_score']))
        # Print per-client results
        print("\nDetailed per-client results:\n")
        for i in range(self.num_clients):
            print(f"Client {stats['client_ids'][i]}")
            print("  Train Loss: {:.4f}".format(stats_train['mse_loss'][i]))
            print("  Train RMSE: {:.4f}".format(stats_train['rmse'][i]))
            print("  Train NASA Score: {:.4f}".format(stats_train['nasa_score'][i]))
            print("  Test Loss: {:.4f}".format(stats['mse_loss'][i]))
            print("  Test RMSE: {:.4f}".format(stats['rmse'][i]))
            print("  Test NASA Score: {:.4f}\n".format(stats['nasa_score'][i]))
            print("")

        # TODO add standard deviation
        self.log_wandb_metrics(global_stats, stats, stats_train, round_idx)

    def log_wandb_metrics(self, global_stats, stats_test, stats_train, round_idx):
        wandb_run = getattr(self.args, "wandb_run", None)
        if wandb_run is None:
            return
        try:
            import wandb
        except ImportError:
            return

        log_step = round_idx if round_idx is not None else len(self.results_history) - 1
        log_payload = {
            "global/train_loss": global_stats['train_mse_loss'],
            "global/train_rmse": global_stats['train_rmse'],
            "global/train_nasa_score": global_stats['train_nasa_score'],
            "global/test_loss": global_stats['test_mse_loss'],
            "global/test_rmse": global_stats['test_rmse'],
            "global/test_nasa_score": global_stats['test_nasa_score'],
            "round": round_idx,
        }

        client_table = wandb.Table(
            columns=[
                "client_id",
                "train_loss",
                "train_rmse",
                "train_nasa_score",
                "test_loss",
                "test_rmse",
                "test_nasa_score",
            ]
        )
        for client_id, train_loss, train_rmse, train_nasa, test_loss, test_rmse, test_nasa in zip(
            stats_train['client_ids'],
            stats_train['mse_loss'],
            stats_train['rmse'],
            stats_train['nasa_score'],
            stats_test['mse_loss'],
            stats_test['rmse'],
            stats_test['nasa_score'],
        ):
            client_table.add_data(
                int(client_id),
                float(train_loss),
                float(train_rmse),
                float(train_nasa),
                float(test_loss),
                float(test_rmse),
                float(test_nasa),
            )
            # Also log flattened per-client metrics for quick charting
            prefix = f"clients/{client_id}"
            log_payload[f"{prefix}/train_loss"] = train_loss
            log_payload[f"{prefix}/train_rmse"] = train_rmse
            log_payload[f"{prefix}/train_nasa_score"] = train_nasa
            log_payload[f"{prefix}/test_loss"] = test_loss
            log_payload[f"{prefix}/test_rmse"] = test_rmse
            log_payload[f"{prefix}/test_nasa_score"] = test_nasa

        log_payload["clients/metrics_table"] = client_table
        wandb_run.log(log_payload, step=log_step)
    
    def save_results(self, round):
        # Save results history to a file
        results_path = os.path.join(self.args.metrics_root, f'metrics_round_{round}.pt')
        torch.save(self.results_history, results_path)

    def save_models(self, round):
        # Save global model to a file
        model_path = os.path.join(self.args.model_root, f'global_model_round_{round}.pt')
        torch.save(self.global_model.state_dict(), model_path)
        # Save each client model to a file
        for client in self.clients:
            client_model_path = os.path.join(self.args.model_root, f'client_{client.id}_model_round_{round}.pt')
            torch.save(client.model.state_dict(), client_model_path)
