import torch
from torch_geometric.loader import DataLoader
from models.trainer import model_train, load_from_checkpoint, eval, model_whole, model_train_prox, model_test
from data_loader.dataloader import EpiDataset, get_splits, get_splits_by_ratio, get_splits_by_ratio2
from models.st_gat2 import ST_GAT2
from models.lstm import LSTM

class Client:
    def __init__(self, client_id, config, W, global_model_params=None):
        self.id = client_id
        self.config = config
        self.W = W
        self.dataset = EpiDataset(config, W, client_id)
        self.n_node = self.dataset.get_n_node()
        self.train_losses = []
        self.val_losses = []
        self.train, self.val, self.test, self.whole = get_splits_by_ratio2(self.dataset, (config['train'], config['val'], 0.3))
        self.train_dataloader = DataLoader(self.train, batch_size=config['BATCH_SIZE'], shuffle=True)
        self.val_dataloader = DataLoader(self.val, batch_size=config['BATCH_SIZE'], shuffle=True)
        self.test_dataloader = DataLoader(self.test, batch_size=config['BATCH_SIZE'], shuffle=False)
        self.whole_dataloader = DataLoader(self.whole, batch_size=config['BATCH_SIZE'], shuffle=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if config['model'] == 'stgat':
            self.model = ST_GAT2(in_channels=config['N_HIST'], n_classes=config['N_Classes'], out_channels=config['N_PRED'], dropout=config['DROPOUT']).to(self.device)
        elif config['model'] == 'lstm':
            self.model = LSTM(in_channels=config['N_HIST'], n_classes=config['N_Classes'], out_channels=config['N_PRED'],  dropout=config['DROPOUT']).to(self.device)
        
        self.global_model_params = global_model_params
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])

    def update_model(self, global_model_state_dict):
        self.model.load_state_dict(global_model_state_dict)
        self.global_model_params = {k: v.clone().detach() for k, v in global_model_state_dict.items()}

    def train_and_evaluate(self):
        if self.config['model_train'] == 'avg':
            self.model, self.train_losses, self.val_losses = model_train(self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.config, self.device)
        else:
            self.model, self.train_losses, self.val_losses = model_train_prox(self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.config, self.device, self.global_model_params)

        return  self.model, self.train_losses, self.val_losses

    def model_predict(self):
        rmse_test, mae_test, f1_macro_test, f1_micro_test, accuracy_test, test_cross_entropy_loss, test_y_pred, test_y_truth = model_test(self.model, self.test_dataloader, self.device, self.config)

        return rmse_test, mae_test, f1_macro_test, f1_micro_test, accuracy_test, test_cross_entropy_loss, test_y_pred, test_y_truth
