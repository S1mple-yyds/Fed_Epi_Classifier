import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

import torch.nn as nn
import numpy as np
import seaborn as sns
from models.st_gat2 import ST_GAT2
from models.lstm import LSTM
# from models.Arima import ARIMA
#from models.tgcn import GNN
from utils.math_utils import *
from torch.utils.tensorboard import SummaryWriter

import os
from pathlib import Path

# Make a tensorboard writer
writer = SummaryWriter()


@torch.no_grad()
def eval(model, device, dataloader, config, type=''):
    """
    Evaluation function to evaluate model on data
    :param model Model to evaluate
    :param device Device to evaluate on
    :param dataloader Data loader
    :param type Name of evaluation type, e.g. Train/Val/Test
    """
    n_nodes = config['N_NODE']
    model.eval()
    model.to(device)
    mae = 0
    rmse = 0
    accuracy = 0
    f1_macro = 0
    f1_micro = 0
    cross_entropy_loss = 0
    n = 0
    batch_size = config['BATCH_SIZE']

    # Evaluate model on all data
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch, device)

            
            truth = batch.y.long()  
            pred = pred.squeeze(dim=-1).float()  

            loss_true = truth.view(-1)  # Flatten to a 1D tensor
            loss_pred = pred.view(-1, pred.size(-1))  # Flatten the first two dimensions while keeping the last one


            log_probabilities = torch.log(loss_pred)
            nll_loss = nn.NLLLoss()
            cross_entropy_loss += nll_loss(log_probabilities, loss_true)  

            
            _, predicted_classes = torch.max(pred, dim=2)  
            truth_flat = truth.flatten().cpu()
            pred_flat = predicted_classes.flatten().cpu()

            
            accuracy += (predicted_classes == truth).float().mean()
            f1_macro += f1_score(truth_flat, pred_flat, average='macro')
            f1_micro += f1_score(truth_flat, pred_flat, average='micro')



            
            truth_infected = (truth == 1).sum(dim=-1).float()  
            pred_infected = (predicted_classes == 1).sum(dim=-1).float()  
            
            pred = predicted_classes
            if i == 0:
                y_pred_infected = torch.zeros(len(dataloader), pred_infected.shape[0])
                y_truth_infected = torch.zeros(len(dataloader), truth_infected.shape[0])
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])

            y_pred_infected[i, :pred_infected.shape[0]] = pred_infected
            y_truth_infected[i, :truth_infected.shape[0]] = truth_infected
            y_pred[i, :pred.shape[0], :] = pred
            y_truth[i, :pred.shape[0], :] = truth

            
            rmse += RMSE(truth_infected.flatten().cpu(), pred_infected.flatten().cpu())
            mae += MAE(truth_infected.flatten().cpu(), pred_infected.flatten().cpu())

            n += 1

    
    rmse = rmse / n
    mae = mae / n
    accuracy = accuracy / n
    f1_macro = f1_macro / n
    f1_micro = f1_micro / n
    cross_entropy_loss = cross_entropy_loss / n  

    
    print(f'{type}, MAE: {mae}, RMSE: {rmse}, Accuracy: {accuracy}, F1 (Macro): {f1_macro}, F1 (Micro): {f1_micro}, Cross-Entropy: {cross_entropy_loss}')
    
    return rmse, mae, accuracy, f1_macro, f1_micro, cross_entropy_loss, y_pred, y_truth

def model_train_prox(model, optimizer, train_dataloader, val_dataloader, config, device, global_model_params):

    loss_fn = nn.NLLLoss()

    
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, restore_best_weights=True)

    model.to(device)

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    all_train_losses = []  
    all_val_losses = []  

    for epoch in range(config['EPOCHS']):
        # Train the model
        #train_loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        model.train()
        for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            batch = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(batch, device)  
            y_true = torch.squeeze(batch.y).long()  

            y_true = y_true.view(-1)  # Flatten to a 1D tensor

            y_pred = y_pred.view(-1, y_pred.size(-1))  # Flatten the first two dimensions while keeping the last one

            log_probabilities = torch.log(y_pred)

            loss = loss_fn(log_probabilities, y_true)  

            
            prox_reg = 0.0
            mu = config['MU']
            if global_model_params is not None:
                for param, global_param in zip(model.parameters(), global_model_params.values()):
                    prox_reg += torch.sum(torch.pow(param - global_param, 2))
                prox_reg *= (mu / 2.0)

            total_loss = loss + prox_reg

            writer.add_scalar("Loss/train", total_loss.item(), epoch)
            total_loss.backward()
            optimizer.step()

        all_train_losses.append(total_loss.item())  

        print(f"Epoch {epoch}: Training Loss: {total_loss.item():.3f}")

        # Evaluate the model on the validation dataset
        #if epoch % 5 == 0:
        with torch.no_grad():
            model.eval()
            val_losses = []
            for batch in val_dataloader:
                batch = batch.to(device)
                y_pred = model(batch, device)
                y_true = torch.squeeze(batch.y).long().view(-1)
                y_pred = y_pred.view(-1, y_pred.size(-1))
                log_probabilities = torch.log(y_pred)
                nll_loss = nn.NLLLoss()
                val_loss = nll_loss(log_probabilities, y_true)
                val_losses.append(val_loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            all_val_losses.append(val_loss)
            print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}")

            if early_stopping(val_loss, model):
                print("Stopping training early")
                break
    
    writer.flush()

    return model, all_train_losses, all_val_losses

def model_train(model, optimizer, train_dataloader, val_dataloader, config, device):

    loss_fn = nn.NLLLoss()


    early_stopping = EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

    model.to(device)
    print(device)
    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    all_train_losses = []  
    all_val_losses = []  

    for epoch in range(config['EPOCHS']):
        # Train the model
        # train_loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        model.train()
        for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            batch = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(batch, device)  
            y_true = torch.squeeze(batch.y).long()  

            y_true = y_true.view(-1)  # Flatten to a 1D tensor

            y_pred = y_pred.view(-1, y_pred.size(-1))  # Flatten the first two dimensions while keeping the last one

            log_probabilities = torch.log(y_pred)

            loss = loss_fn(log_probabilities, y_true)  

            writer.add_scalar("Loss/train", loss.item(), epoch)
            loss.backward()
            optimizer.step()

        all_train_losses.append(loss.item())  

        print(f"Epoch {epoch}: Training Loss: {loss.item():.3f}")

        # Evaluate the model on the validation dataset
        # if epoch % 5 == 0:
        with torch.no_grad():
            model.eval()
            val_losses = []
            for batch in val_dataloader:
                batch = batch.to(device)
                y_pred = model(batch, device)
                y_true = torch.squeeze(batch.y).long().view(-1)
                y_pred = y_pred.view(-1, y_pred.size(-1))
                log_probabilities = torch.log(y_pred)
                nll_loss = nn.NLLLoss()
                val_loss = nll_loss(log_probabilities, y_true)
                val_losses.append(val_loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            all_val_losses.append(val_loss)
            print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}")

            if early_stopping(val_loss, model):
                print("Stopping training early")
                break

    writer.flush()

    return model, all_train_losses, all_val_losses

def model_test(model, test_dataloader, device, config):

    print("test satrt")
    test_rmse, test_mae, test_f1_macro, test_f1_micro, test_accuracy, test_cross_entropy_loss,test_y_pred, test_y_truth = eval(model, device, test_dataloader, config, 'Test')
    
    print("test end")
    return test_rmse, test_mae, test_f1_macro, test_f1_micro, test_accuracy, test_cross_entropy_loss,test_y_pred, test_y_truth


def model_whole(model, whole_dataloader, device, config, id):

    whole_rmse, whole_mae, whole_f1_macro, whole_f1_micro, whole_accuracy, whole_y_pred, whole_y_truth = eval(model, device, whole_dataloader,config, 'whole')
    return whole_rmse, whole_mae, whole_f1_macro, whole_f1_micro, whole_accuracy, whole_y_pred, whole_y_truth



def load_from_checkpoint(checkpoint_path, config):
    """
    Load a model from the checkpoint
    :param checkpoint_path Path to checkpoint
    :param config Configuration to load model with
    """
    #model = LSTM(in_channels=config['N_HIST'], n_classes=config['N_Classes'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'], dropout=config['DROPOUT'])
    #model = GNN(dim_in=config['N_HIST'], periods=config['N_PRED'], n_classes=config['N_Classes'],batch_size=config['BATCH_SIZE'])
    model = ST_GAT2(in_channels=config['N_HIST'], n_classes=config['N_Classes'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'], dropout=config['DROPOUT'])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    #return model

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    


