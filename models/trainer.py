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

            # 分类任务的预测值和真实值
            truth = batch.y.long()  # 真实的 SIR 状态
            pred = pred.squeeze(dim=-1).float()  # 模型输出的概率值

            loss_true = truth.view(-1)  # Flatten to a 1D tensor
            loss_pred = pred.view(-1, pred.size(-1))  # Flatten the first two dimensions while keeping the last one


            log_probabilities = torch.log(loss_pred)
            nll_loss = nn.NLLLoss()
            cross_entropy_loss += nll_loss(log_probabilities, loss_true)  # 使用整数标签

            # 预测类别
            _, predicted_classes = torch.max(pred, dim=2)  # 预测类别
            truth_flat = truth.flatten().cpu()
            pred_flat = predicted_classes.flatten().cpu()

            # 计算准确率和F1分数
            accuracy += (predicted_classes == truth).float().mean()
            f1_macro += f1_score(truth_flat, pred_flat, average='macro')
            f1_micro += f1_score(truth_flat, pred_flat, average='micro')



            # 假设感染者的标签为 1（I 状态）
            truth_infected = (truth == 1).sum(dim=-1).float()  # 真实的感染者人数
            pred_infected = (predicted_classes == 1).sum(dim=-1).float()  # 预测的感染者人数
            
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

            # 计算感染者人数的 RMSE 和 MAE
            rmse += RMSE(truth_infected.flatten().cpu(), pred_infected.flatten().cpu())
            mae += MAE(truth_infected.flatten().cpu(), pred_infected.flatten().cpu())

            n += 1

    # 计算平均 RMSE、MAE、Accuracy、F1-score 和 Cross-Entropy Loss
    rmse = rmse / n
    mae = mae / n
    accuracy = accuracy / n
    f1_macro = f1_macro / n
    f1_micro = f1_micro / n
    cross_entropy_loss = cross_entropy_loss / n  # 平均交叉熵损失

    # 打印结果
    print(f'{type}, MAE: {mae}, RMSE: {rmse}, Accuracy: {accuracy}, F1 (Macro): {f1_macro}, F1 (Micro): {f1_micro}, Cross-Entropy: {cross_entropy_loss}')
    
    return rmse, mae, accuracy, f1_macro, f1_micro, cross_entropy_loss, y_pred, y_truth

def model_train_prox(model, optimizer, train_dataloader, val_dataloader, config, device, global_model_params):

    loss_fn = nn.NLLLoss()

    # 创建 EarlyStopping 实例
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, restore_best_weights=True)

    model.to(device)

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    all_train_losses = []  # 新增用于记录每个epoch的训练损失
    all_val_losses = []  # 新增用于记录每个epoch的验证损失

    for epoch in range(config['EPOCHS']):
        # Train the model
        #train_loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        model.train()
        for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            batch = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(batch, device)  # 直接使用模型输出的概率分布
            y_true = torch.squeeze(batch.y).long()  # 确保标签是整数类型

            y_true = y_true.view(-1)  # Flatten to a 1D tensor

            y_pred = y_pred.view(-1, y_pred.size(-1))  # Flatten the first two dimensions while keeping the last one

            log_probabilities = torch.log(y_pred)

            loss = loss_fn(log_probabilities, y_true)  # 使用整数标签

            # 添加 FedProx 正则项
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

        all_train_losses.append(total_loss.item())  # 记录训练损失

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

    # 创建 EarlyStopping 实例
    early_stopping = EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

    model.to(device)
    print(device)
    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    all_train_losses = []  # 新增用于记录每个epoch的训练损失
    all_val_losses = []  # 新增用于记录每个epoch的验证损失

    for epoch in range(config['EPOCHS']):
        # Train the model
        # train_loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        model.train()
        for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            batch = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(batch, device)  # 直接使用模型输出的概率分布
            y_true = torch.squeeze(batch.y).long()  # 确保标签是整数类型

            y_true = y_true.view(-1)  # Flatten to a 1D tensor

            y_pred = y_pred.view(-1, y_pred.size(-1))  # Flatten the first two dimensions while keeping the last one

            log_probabilities = torch.log(y_pred)

            loss = loss_fn(log_probabilities, y_true)  # 使用整数标签

            writer.add_scalar("Loss/train", loss.item(), epoch)
            loss.backward()
            optimizer.step()

        all_train_losses.append(loss.item())  # 记录训练损失

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
    """
    Test the ST-GAT model
    :param test_dataloader Data loader of test dataset
    :param device Device to evaluate on
    """
    print("test satrt")
    test_rmse, test_mae, test_f1_macro, test_f1_micro, test_accuracy, test_cross_entropy_loss,test_y_pred, test_y_truth = eval(model, device, test_dataloader, config, 'Test')
    
    print("test end")
    # plot_prediction2(test_dataloader, test_y_pred, test_y_truth, 1, config)
    # plot_prediction5(test_dataloader, test_y_pred, test_y_truth, config)
    # nodes = [0, 1, 2, 3, 4]
    # plot_prediction6(test_dataloader, test_y_pred, test_y_truth, nodes, config)
    return test_rmse, test_mae, test_f1_macro, test_f1_micro, test_accuracy, test_cross_entropy_loss,test_y_pred, test_y_truth


def model_whole(model, whole_dataloader, device, config, id):
    
    #Test the ST-GAT model
    #:param test_dataloader Data loader of test dataset
    #:param device Device to evaluate on
    
    whole_rmse, whole_mae, whole_f1_macro, whole_f1_micro, whole_accuracy, whole_y_pred, whole_y_truth = eval(model, device, whole_dataloader,config, 'whole')
    #plot_prediction3(whole_dataloader, whole_y_pred, whole_y_truth, config)

    return whole_rmse, whole_mae, whole_f1_macro, whole_f1_micro, whole_accuracy, whole_y_pred, whole_y_truth


def plot_prediction2(test_dataloader, test_y_pred, test_y_truth, node, config):
    # Calculate the truth
    #node = 19 # node在model_test内定义
    s = test_y_truth.shape
    print(test_y_truth.shape) #[72,10000,1]
    y_truth = test_y_truth.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
    # just get the first prediction out for the nth node
    y_truth = y_truth[:, :, node, 0]
    print(y_truth.shape) #[72, 50]
    # Flatten to get the predictions for entire test dataset
    y_truth = torch.flatten(y_truth)
    #day0_truth = y_truth[:config['N_SLOT']]


    # Calculate the predicted
    s = test_y_pred.shape
    y_pred = test_y_pred.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
    # just get the first prediction out for the nth node
    y_pred = y_pred[:, :, node, 0] #0 是第一次预测的情况
    # Flatten to get the predictions for entire test dataset
    y_pred = torch.flatten(y_pred)
    # Just grab the first day
    #day0_pred = y_pred[:config['N_SLOT']]
    total_time_points = len(y_truth)
    t = [t for t in range(0, total_time_points * 5, 5)]
    #t = [t for t in range(0, config['N_SLOT']*5, 5)]
    plt.plot(t, y_pred, label='ST-GAT')
    plt.plot(t, y_truth, label='truth')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Speed prediction')
    plt.title('Predictions of traffic over time')
    plt.legend()
    plt.savefig('predicted_times.png')
    plt.show()

def plot_prediction3(whole_dataloader, whole_y_pred, whole_y_truth, config, id):
    # Calculate the truth
    s = whole_y_truth.shape
    print(whole_y_truth.shape) #[192,10000,1]
    # Reshape y_truth to match the entire dataset's time steps
    total_time_steps = s[0] * config['BATCH_SIZE']
    y_truth = whole_y_truth.reshape(total_time_steps, config['NODE_PER_CLIENT'], s[-1])
    print(y_truth.shape) #[9600,200,1]
    
    # Calculate the predicted
    s = whole_y_pred.shape
    print(whole_y_pred.shape) #[192,10000,1]
    # Reshape y_pred to match the entire dataset's time steps
    y_pred = whole_y_pred.reshape(total_time_steps, config['NODE_PER_CLIENT'], s[-1])
    print(y_pred.shape) #[9600,200,1]
    
    # Initialize lists to store counts
    y_truth_counts_0 = []
    y_pred_counts_0 = []
    y_truth_counts_1 = []
    y_pred_counts_1 = []
    y_truth_counts_2 = []
    y_pred_counts_2 = []

    # Iterate over each time step
    for t in range(total_time_steps):
        # Count the number of nodes with state 0 at each time step
        truth_count_0 = torch.sum(y_truth[t, :, 0] == 0).item()
        y_truth_counts_0.append(truth_count_0)
        pred_count_0 = torch.sum(y_pred[t, :, 0] == 0).item()
        y_pred_counts_0.append(pred_count_0)

        # Count the number of nodes with state 1 at each time step
        truth_count_1 = torch.sum(y_truth[t, :, 0] == 1).item()
        y_truth_counts_1.append(truth_count_1)
        pred_count_1 = torch.sum(y_pred[t, :, 0] == 1).item()
        y_pred_counts_1.append(pred_count_1)

        # Count the number of nodes with state 2 at each time step
        truth_count_2 = torch.sum(y_truth[t, :, 0] == 2).item()
        y_truth_counts_2.append(truth_count_2)
        pred_count_2 = torch.sum(y_pred[t, :, 0] == 2).item()
        y_pred_counts_2.append(pred_count_2)

    # Calculate the new time vector t
    # Each slot is 5 minutes, so we multiply by 5
    t = [t for t in range(total_time_steps)]

    # Ensure the length of t matches the length of y_pred_counts and y_truth_counts
    #assert len(t) == len(y_pred_counts_0) == len(y_pred_counts_1) == len(y_pred_counts_2) == len(y_truth_counts_0) == len(y_truth_counts_1) == len(y_truth_counts_2), "The lengths of t, y_pred_counts, and y_truth_counts should be the same."
    # Plot the counts
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    axs[0].plot(t, y_pred_counts_0, label='ST-GAT Predicted State 0')
    axs[0].plot(t, y_truth_counts_0, label='Truth State 0')
    axs[0].set_title(f"client{id} State 0")
    axs[0].legend()

    axs[1].plot(t, y_pred_counts_1, label='ST-GAT Predicted State 1')
    axs[1].plot(t, y_truth_counts_1, label='Truth State 1')
    axs[1].set_title(f"client{id} State 1")

    axs[2].plot(t, y_pred_counts_2, label='ST-GAT Predicted State 2')
    axs[2].plot(t, y_truth_counts_2, label='Truth State 2')
    axs[2].set_title(f"client{id} State 2")

    axs[0].set_xlabel('Time (minutes)')
    axs[1].set_xlabel('Time (minutes)')
    axs[2].set_xlabel('Time (minutes)')
    axs[0].set_ylabel('Number of Nodes in State 0')
    axs[1].set_ylabel('Number of Nodes in State 1')
    axs[2].set_ylabel('Number of Nodes in State 2')

    plt.tight_layout()
    plt.savefig('predicted_state_counts.png')
    plt.show()


def plot_prediction4(test_dataloader, test_y_pred, test_y_truth, config):
    # Calculate the truth
    s = test_y_truth.shape
    y_truth = test_y_truth.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
    
    # Flatten to get the predictions for entire test dataset
    y_truth = torch.flatten(y_truth, start_dim=0, end_dim=2)
    print(y_truth.shape)  # 应该是 [720000]

    # Calculate the predicted
    s = test_y_pred.shape
    y_pred = test_y_pred.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
    
    # Flatten to get the predictions for entire test dataset
    y_pred = torch.flatten(y_pred, start_dim=0, end_dim=2)
    print(y_pred.shape)  # 应该是 [720000]

    # Split the tensors into segments corresponding to each time step
    num_time_steps = y_truth.shape[0] // (config['BATCH_SIZE'] * config['N_NODE'])
    y_truth_segments = y_truth.split(config['BATCH_SIZE'] * config['N_NODE'])
    y_pred_segments = y_pred.split(config['BATCH_SIZE'] * config['N_NODE'])

    # Count the number of states that are 1 for both truth and predictions
    state_1_truth = torch.stack([(segment == 1).sum() for segment in y_truth_segments])
    state_1_pred = torch.stack([(segment == 1).sum() for segment in y_pred_segments])

    # Create a list representing time points for the entire dataset
    t = [t for t in range(num_time_steps)]

    plt.plot(t, state_1_pred, label='Prediction')
    plt.plot(t, state_1_truth, label='Truth')
    plt.xlabel('Time Step')
    plt.ylabel('Count of State 1')
    plt.title('Counts of State 1 Over Time')
    plt.legend()
    plt.savefig('state_1_counts_over_time.png')
    plt.show()

def plot_prediction5(test_dataloader, test_y_pred, test_y_truth, config):
    # 提取两个时间点的数据
    first_time_step = 2000
    last_time_step = 2399

    # 重塑数据以便更容易处理
    y_truth = test_y_truth.view(-1, config['N_NODE'])
    y_pred = test_y_pred.view(-1, config['N_NODE'])

    # 统计两个时间点的状态
    def count_states(y_data, time_step):
        state_counts = torch.bincount(y_data[time_step].long(), minlength=3)
        return state_counts.cpu().numpy()  # 将张量转换成NumPy数组以便绘图

    true_first_time_counts = count_states(y_truth, first_time_step)
    pred_first_time_counts = count_states(y_pred, first_time_step)

    true_last_time_counts = count_states(y_truth, last_time_step)
    pred_last_time_counts = count_states(y_pred, last_time_step)

    # 绘制柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 第一个时间点的柱状图
    ax1.bar(np.arange(3), true_first_time_counts, label='True', alpha=0.5)
    ax1.bar(np.arange(3), pred_first_time_counts, bottom=true_first_time_counts, label='Predicted', alpha=0.5)
    ax1.set_title(f'Time Step {first_time_step}')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Count')
    ax1.legend()

    # 最后一个时间点的柱状图
    ax2.bar(np.arange(3), true_last_time_counts, label='True', alpha=0.5)
    ax2.bar(np.arange(3), pred_last_time_counts, bottom=true_last_time_counts, label='Predicted', alpha=0.5)
    ax2.set_title(f'Time Step {last_time_step}')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Count')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('comparation.png')
    plt.show()

def plot_prediction6(test_dataloader, test_y_pred, test_y_truth, nodes, config):
    # 初始化一个包含多个子图的大图
    fig, axs = plt.subplots(len(nodes), 1, figsize=(10, 2.5*len(nodes)))  # 每个节点一个子图

    # 遍历所有感兴趣的节点
    for i, node in enumerate(nodes):
        # Calculate the truth
        s = test_y_truth.shape
        y_truth = test_y_truth.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
        # just get the first prediction out for the nth node
        y_truth = y_truth[:, :, node, 0]
        # Flatten to get the predictions for entire test dataset
        y_truth = torch.flatten(y_truth)

        # Calculate the predicted
        s = test_y_pred.shape
        y_pred = test_y_pred.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
        # just get the first prediction out for the nth node
        y_pred = y_pred[:, :, node, 0] 
        # Flatten to get the predictions for entire test dataset
        y_pred = torch.flatten(y_pred)

        # 计算时间序列
        total_time_points = len(y_truth)
        t = [t for t in range(0, total_time_points * 5, 5)]

        # 绘制预测和真实值
        axs[i].plot(t, y_pred, label='ST-GAT')
        axs[i].plot(t, y_truth, label='Truth')

        # 设置子图的标题
        axs[i].set_title(f'Node {node}: Predictions of traffic over time')
        axs[i].set_xlabel('Time (minutes)')
        axs[i].set_ylabel('Speed prediction')
        axs[i].legend()

    # 调整子图之间的间距
    fig.tight_layout()

    # 保存并显示整个图表
    plt.savefig('predicted_times.png')
    plt.show()


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
    
def plot_client_metrics(client_train_losses, client_val_losses, node_states_filename, n_clients):
    plot_dir = 'plot'
    for client_id in client_train_losses.keys():
        plt.figure(figsize=(10, 6))
        
        train_losses = [loss for sublist in client_train_losses[client_id] for loss in sublist]
        val_losses = [loss for sublist in client_val_losses[client_id] for loss in sublist]
        
        epochs = list(range(1, len(train_losses) + 1))
        
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, val_losses, label='Val Loss', color='red', linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Losses for Client {client_id}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{os.path.basename(node_states_filename)}_NClients{n_clients}_clientid{client_id}_losses.png'))
        plt.show()

def plot_1_prediction(clients, y_pred, y_truth, config, node_states_filename):
    total_time_steps = 0
    y_truth_counts_1 = []
    y_pred_counts_1 = []
    plot_dir = 'lstmprox_plot_1'
    filename = f'{os.path.splitext(os.path.basename(node_states_filename))[0]}_{len(clients)}.png'

    # 遍历每个客户端，处理真实值和预测值
    for client in clients:
        s = y_truth[client.id].shape
        total_time_steps = s[0] * config['BATCH_SIZE']
        client_y_truth = y_truth[client.id].reshape(total_time_steps, client.n_node, s[-1])

        s = y_pred[client.id].shape
        client_y_pred = y_pred[client.id].reshape(total_time_steps, client.n_node, s[-1])

        # 计算每个时间步的1的数量
        for t in range(total_time_steps):
            truth_count_1 = torch.sum(client_y_truth[t, :, 0] == 1).item()
            pred_count_1 = torch.sum(client_y_pred[t, :, 0] == 1).item()
            if len(y_truth_counts_1) <= t:
                y_truth_counts_1.append(truth_count_1)
                y_pred_counts_1.append(pred_count_1)
            else:
                y_truth_counts_1[t] += truth_count_1
                y_pred_counts_1[t] += pred_count_1

    # 创建时间轴
    t = [t for t in range(total_time_steps)]

    # 创建子图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制所有客户端的预测和真实值
    ax.plot(t, y_pred_counts_1, label='Predicted')
    ax.plot(t, y_truth_counts_1, label='Truth')

    # 设置图表标题和标签
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Count of 1s')
    ax.set_title('Predictions vs. Truth for All Clients Combined')
    ax.legend()

    # 保存并显示图表
    plt.savefig(os.path.join(plot_dir, filename))
    plt.show()