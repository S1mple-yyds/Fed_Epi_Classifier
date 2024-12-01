import os
import time
import torch
import torch.optim as optim
from models.st_gat2 import ST_GAT2
from models.lstm import LSTM

class Server:
    def __init__(self, config):
        self.config = config
        self.global_model = ST_GAT2(in_channels=self.config['N_HIST'], n_classes=self.config['N_Classes'], out_channels=self.config['N_PRED'], n_nodes=self.config['NODE_PER_CLIENT'], dropout=self.config['DROPOUT'])
        #self.global_model = LSTM(in_channels=self.config['N_HIST'], n_classes=3, n_nodes=self.config['NODE_PER_CLIENT'], out_channels=self.config['N_PRED'], dropout=self.config['DROPOUT'])
        #self.global_model = GNN(dim_in=config['N_HIST'], periods=config['N_PRED'], n_classes=config['N_Classes'], batch_size=config['BATCH_SIZE'])
        self.global_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    def distribute(self):
        return self.global_model.state_dict()

    def aggregate(self, client_models):
        global_state_dict = self.global_model.state_dict()
        #client_state_dicts = [client_model.state_dict() for client_model in client_models]
        client_state_dicts = [client.model.state_dict() for client in client_models]
        for key in global_state_dict.keys():
            #global_state_dict[key] = torch.stack([client_model[key] for client_model in client_models], 0).mean(0)
            global_state_dict[key] = torch.stack([client_state_dict[key] for client_state_dict in client_state_dicts],0).mean(0)
        self.global_model.load_state_dict(global_state_dict)

    def aggregate_SGD2(self, client_models):
        global_state_dict = self.global_model.state_dict()

        # 初始化所有梯度为0
        aggregated_gradients = {key: torch.zeros_like(param) for key, param in global_state_dict.items()}
        total_samples = sum(client.num_samples for client in client_models)

        # 根据每个客户端的数据量加权累积梯度
        for client in client_models:
            client_gradients = client.model.state_dict()
            client_samples = client.num_samples
            for key in global_state_dict.keys():
                aggregated_gradients[key] += client_gradients[key] * (client_samples / total_samples)

        # 对累积梯度进行裁剪以避免梯度爆炸或消失
        for key in aggregated_gradients.keys():
            torch.nn.utils.clip_grad_norm_(aggregated_gradients[key], max_norm=1.0)

        # 使用动量项更新全局模型参数
        momentum = self.config.get('MOMENTUM', 0.9)
        # 更新全局模型参数
        for key in global_state_dict.keys():
            global_state_dict[key] -= self.config['INITIAL_LR'] * aggregated_gradients[key]

        self.global_model.load_state_dict(global_state_dict)
    def save_model(self):
        timestr = time.strftime("%m-%d-%H%M%S")
        torch.save(self.global_model.state_dict(), os.path.join(self.config["CHECKPOINT_DIR"], f"global_model_{timestr}.pt"))
