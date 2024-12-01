import torch
import pandas as pd
from models.trainer import load_from_checkpoint, model_train ,model_whole, model_test
from torch_geometric.loader import DataLoader
from data_loader.dataloader import TrafficDataset, get_splits, get_splits_by_ratio, distance_to_weight
from models.Clients import Client
from models.Server import Server
import random
from matplotlib import pyplot as plt
import csv
import os
from collections import defaultdict


def write_to_csv(client_acc, test_cross_entropy_loss, client_rmse, client_mae, client_f1_macro, client_f1_micro, node_states_filename, n_client):
    filename = f'{os.path.splitext(os.path.basename(node_states_filename))[0]}_{n_client}_stgat_avg_.csv'
    full_path = os.path.join('result', filename)
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['node_states_filename', 'ClientID'] + [f'Acc{i}' for i in range(len(next(iter(client_acc.values())))) ] + [f'test_loss{i}' for i in range(len(next(iter(test_cross_entropy_loss.values())))) ]
        header += [f'rmse{i}' for i in range(len(next(iter(client_rmse.values())))) ]
        header += [f'mae{i}' for i in range(len(next(iter(client_mae.values())))) ]
        header += [f'f1_macro{i}' for i in range(len(next(iter(client_f1_macro.values()))))]
        header += [f'f1_micro{i}' for i in range(len(next(iter(client_f1_micro.values()))))]
        writer.writerow(header)

        for client_id, acc_values in client_acc.items():
            loss_values = test_cross_entropy_loss.get(client_id, [])
            rmse = client_rmse.get(client_id, [])
            mae = client_mae.get(client_id, [])
            macro = client_f1_macro.get(client_id, [])
            micro = client_f1_micro.get(client_id, [])
            row = [node_states_filename, client_id] + acc_values + loss_values + rmse + mae + macro + micro
            writer.writerow(row)



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
        #plt.savefig(f'plot/{os.path.basename(node_states_filename)}_client_{client_id}_losses.png')
        plt.show()

def main(config, node_states_filename):

    # Load the adjacency matrix and node states
    adj_matrix = pd.read_csv('./node600_10000/airport_600.csv', header=None)
    node_states = pd.read_csv(node_states_filename, header=None)
    print("dataset name:", node_states_filename)

    print("Adjacency Matrix Shape:", adj_matrix.shape)
    print("Node States Shape:", node_states.shape)

    # Number of clients
    n_clients = config['N_CLIENTS']
    # Number of nodes per client
    nodes_per_client = config['NODE_PER_CLIENT']

    # Split the adjacency matrix and node states
    adj_splits = []
    state_splits = []
    W = []

    for i in range(n_clients):
        # Adjacency matrix for this client
        adj_split = adj_matrix.iloc[i * nodes_per_client:(i + 1) * nodes_per_client,
                    i * nodes_per_client:(i + 1) * nodes_per_client]
        adj_splits.append(adj_split)

        # Node states for this client
        state_split = node_states.iloc[:, i * nodes_per_client:(i + 1) * nodes_per_client]
        state_splits.append(state_split)

    # Save each split to a separate file
    output_dir = './ClientDatas/'
    for i in range(n_clients):
        W.append(distance_to_weight(adj_splits[i].values, gat_version=config['USE_GAT_WEIGHTS']))
        adj_splits[i].to_csv(output_dir + f'client{i + 1}_adj_tiny.csv', index=False, header=False)
        state_splits[i].to_csv(output_dir + f'client{i + 1}_node_state_tiny.csv', index=False, header=False)

    print("Data splits have been saved.")

    clients = [Client(i + 1, config, W[i]) for i in range(n_clients)]
    server = Server(config)

    client_train_losses = {i + 1: [] for i in range(n_clients)}
    client_val_losses = {i + 1: [] for i in range(n_clients)}
    client_acc = {i+1:[] for i in range(n_clients)}
    test_cross_entropy_loss = {i+1:[] for i in range(n_clients)}
    client_rmse = {i+1:[] for i in range(n_clients)}
    client_mae = {i+1:[] for i in range(n_clients)}
    client_f1_macro = {i+1:[] for i in range(n_clients)}
    client_f1_micro = {i+1:[] for i in range(n_clients)}
    #rmse_test, mae_test, f1_macro_test, f1_micro_test, accuracy_test, accuracy_whole = 0, 0, 0, 0, 0, 0
    rmse_test, mae_test, f1_macro_test, f1_micro_test = 0, 0, 0, 0
    accuracy_test = []
    total_train_losses = []
    total_val_losses = []
    test_losses = []
    last_10_min_train_losses = {i + 1: [] for i in range(n_clients)}
    last_10_min_val_losses = {i + 1: [] for i in range(n_clients)}

    for round_num in range(config['NUM_ROUND']):
        # print(f"Round {round_num + 1}/{config['NUM_ROUND']}")

        # 随机选择部分客户端参与本轮训练
        selected_clients = random.sample(clients, config['CLIENTS_PER_ROUND'])
        selected_clients.sort(key=lambda client: client.id)
        client_models = []
        #client_train_losses = defaultdict(list)
        #client_val_losses = defaultdict(list)

        for client in clients:
            print(f"Round {round_num + 1}/{config['NUM_ROUND']},client {client.id}")
            client.update_model(server.distribute())
            client_model, train_losses, val_losses = client.train_and_evaluate()

            client_models.append(client_model)

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            #client_train_losses[client.id].append(avg_train_loss)
            #client_val_losses[client.id].append(avg_val_loss)
            client_train_losses[client.id].append(train_losses)
            client_val_losses[client.id].append(val_losses)

            last_10_min_train_losses[client.id].append(min(train_losses[-5:]))
            last_10_min_val_losses[client.id].append(min(val_losses[-5:]))

        #server.aggregate_SGD2(selected_clients)
        server.aggregate(selected_clients)
        #server.aggregate_SGD(client_models)
        for client in selected_clients:
            client.update_model(server.distribute())
            #rmse_test1, mae_test1, f1_macro_test1, f1_micro_test1, acc_test, acc_whole = client.model_predict()
            rmse_test1, mae_test1, f1_macro_test1, f1_micro_test1, acc_test, test_cross_entropy_loss1 = client.model_predict()
            rmse_test += rmse_test1
            mae_test += mae_test1
            f1_macro_test += f1_macro_test1
            f1_micro_test += f1_micro_test1
            accuracy_test.append(acc_test)
            test_losses.append(test_cross_entropy_loss1)
            test_cross_entropy_loss[client.id].append(test_cross_entropy_loss1.cpu().item())
            client_acc[client.id].append(acc_test)
            client_rmse[client.id].append(rmse_test1.cpu().item())
            client_mae[client.id].append(mae_test1.cpu().item())
            client_f1_macro[client.id].append(f1_macro_test1.cpu().item())
            client_f1_micro[client.id].append(f1_micro_test1)


    write_to_csv(client_acc, test_cross_entropy_loss, client_rmse, client_mae, client_f1_macro, client_f1_micro, node_states_filename, n_clients)

        # 遍历每个客户端的训练损失和验证损失
    for client_id in client_train_losses:
        total_train_losses.extend(last_10_min_train_losses[client_id])
        total_val_losses.extend(last_10_min_val_losses[client_id])

        # 计算平均训练损失和验证损失
    num_clients = len(client_train_losses)
    avg_train_loss = sum(total_train_losses) / (num_clients * config['NUM_ROUND'])
    avg_val_loss = sum(total_val_losses) / (num_clients * config['NUM_ROUND'])
    
    avg_rmse_test = rmse_test.cpu().item()/(config['NUM_ROUND']*config['CLIENTS_PER_ROUND'])
    avg_mae_test = mae_test.cpu().item()/(config['NUM_ROUND']*config['CLIENTS_PER_ROUND'])
    avg_f1_macro_test = f1_macro_test.cpu().item()/(config['NUM_ROUND']*config['CLIENTS_PER_ROUND'])
    avg_f1_micro_test = f1_micro_test/(config['NUM_ROUND']*config['CLIENTS_PER_ROUND'])
    avg_accuracy_test = (sum(accuracy_test))/(config['NUM_ROUND']*config['CLIENTS_PER_ROUND'])
    avg_test_cross_entropy_loss = (sum(test_losses)).cpu().item()/(config['NUM_ROUND']*config['CLIENTS_PER_ROUND'])

    LastRound_avg_acc_test = (sum(accuracy_test[-1*config['CLIENTS_PER_ROUND']:]))/(config['CLIENTS_PER_ROUND'])
    max_acc_test = (max(accuracy_test[-1*config['CLIENTS_PER_ROUND']:]))
    min_acc_test = (min(accuracy_test[-1*config['CLIENTS_PER_ROUND']:]))
    
    #avg_accuracy_whole = accuracy_whole.cpu().item()/(config['NUM_ROUND']*config['CLIENTS_PER_ROUND'])

    server.save_model()
    print("Federated learning training completed and global model saved.")

    # Plot client metrics
    #plot_client_metrics(client_train_losses, client_val_losses, node_states_filename, n_clients)

    return avg_rmse_test, avg_mae_test, avg_f1_macro_test, avg_f1_micro_test, avg_accuracy_test, avg_test_cross_entropy_loss,avg_train_loss, avg_val_loss,max_acc_test,min_acc_test,LastRound_avg_acc_test


if __name__ == "__main__":
    config = {
        'BATCH_SIZE': 32,
        'EPOCHS': 5,
        'WEIGHT_DECAY': 5e-5,  # origin 5e-5
        'INITIAL_LR': 1e-3,  # origin 3e-4
        'CHECKPOINT_DIR': './runs',
        'N_PRED': 10,  # change into 5
        'N_HIST': 10,
        'DROPOUT': 0.4,
        # number of possible 5 minute measurements per day
        'N_DAY_SLOT': 250,
        # number of days worth of data in the dataset
        'N_DAYS': 40,
        # If false, use GCN paper weight matrix, if true, use GAT paper weight matrix
        'USE_GAT_WEIGHTS': True,
        'N_NODE': 600,
        # 联邦学习客户端数量
        'N_Classes': 3,

        'N_CLIENTS': 4,
        # 联邦学习轮数
        'NUM_ROUND': 100,
        'MU': 0.01,  # FedProx正则化项
        'CLIENTS_PER_ROUND': 2,  # 每一轮参与聚合的客户端数量
    }
    # Number of possible windows in a day
    config['N_SLOT'] = config['N_DAY_SLOT'] - (config['N_PRED'] + config['N_HIST']) + 1
    
    node_states_datasets = [
        './node600_10000/node_state_SIRS_20_005_005_01_005_100.csv',
    ]
    
    output_csv = 'test2.csv'
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'client_number', 'avg_rmse_test', 'avg_mae_test', 'avg_f1_macro_test', 'avg_f1_micro_test', 'avg_accuracy_test', 'avg_test_cross_entropy_loss',' avg_train_loss', 'avg_val_loss', 'max_acc', 'min_acc','LastRound_avg_acc_test'])
        file.flush()
        os.fsync(file.fileno())
        # 初始化保存准确率和客户端数量的列表

        for node_states_filename in node_states_datasets:
            #for i in [4]:
            for i in [2**(x+1) for x in range(4)]:
                config['N_CLIENTS'] = i
                config['CLIENTS_PER_ROUND'] = config['N_CLIENTS']
                config['NODE_PER_CLIENT'] = int(config['N_NODE'] / config['N_CLIENTS'])
                
                avg_rmse_test, avg_mae_test, avg_f1_macro_test, avg_f1_micro_test, avg_accuracy_test, avg_test_cross_entropy_loss,avg_train_loss, avg_val_loss, max_acc, min_acc,LastRound_avg_acc_test = main(config,node_states_filename)
                
                print(f"Writing to CSV: {node_states_filename}, {config['N_CLIENTS']}, {avg_rmse_test}, {avg_mae_test}, {avg_f1_macro_test}, {avg_f1_micro_test},{avg_accuracy_test},{avg_test_cross_entropy_loss}, {avg_train_loss}, {avg_val_loss}, {max_acc}, {min_acc},{LastRound_avg_acc_test}")
                
                writer.writerow([node_states_filename, config['N_CLIENTS'], avg_rmse_test, avg_mae_test, avg_f1_macro_test, avg_f1_micro_test, avg_accuracy_test, avg_test_cross_entropy_loss,avg_train_loss, avg_val_loss, max_acc, min_acc, LastRound_avg_acc_test])
                file.flush()
                os.fsync(file.fileno())
                print(f"结果已保存到 {output_csv}")
