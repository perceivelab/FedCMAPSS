import time
import copy
from flcore.clients.clientavg_rul import clientAVG_RUL
from flcore.servers.serverbase_rul import Server_RUL


class Centralized_RUL(Server_RUL):
    def __init__(self, args, times):
        super().__init__(args, times)
        # Force one client
        self.num_clients = 1
        args.num_clients = 1 
        self.train_slow_clients = [False]
        self.send_slow_clients = [False]
        self.set_clients(clientAVG_RUL)
        # self.load_model()
        self.Budget = []
    
    def select_clients(self):
        self.current_num_join_clients = 1
        return [self.clients[0]]

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                self.evaluate_and_checkpoint(i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG_RUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(round_idx=self.global_rounds + 1)

    def aggregate_parameters(self):
        self.global_model = copy.deepcopy(self.uploaded_models[0])