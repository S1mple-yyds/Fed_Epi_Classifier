import os
import time
import torch
import torch.optim as optim
from models.st_gat2 import ST_GAT2
from models.lstm import LSTM

class Server:
    def __init__(self, config):
        self.config = config
        if config['model'] == 'stgat':
            self.global_model = ST_GAT2(in_channels=config['N_HIST'], n_classes=config['N_Classes'], out_channels=config['N_PRED'], dropout=config['DROPOUT'])
        elif config['model'] == 'lstm':
            self.global_model = LSTM(in_channels=config['N_HIST'], n_classes=config['N_Classes'], out_channels=config['N_PRED'],  dropout=config['DROPOUT'])
        
        self.global_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    def distribute(self):
        return self.global_model.state_dict()

    def aggregate(self, client_models):
        global_state_dict = self.global_model.state_dict()
        client_state_dicts = [client.model.state_dict() for client in client_models]
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([client_state_dict[key] for client_state_dict in client_state_dicts],0).mean(0)
        self.global_model.load_state_dict(global_state_dict)
    def save_model(self):
        timestr = time.strftime("%m-%d-%H%M%S")
        torch.save(self.global_model.state_dict(), os.path.join(self.config["CHECKPOINT_DIR"], f"global_model_{timestr}.pt"))
