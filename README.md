# fed

# 论文实验 - Towards_federated_prediction_for_epidemics_on_networks

## 背景
简要介绍你的研究背景、实验目的以及论文的核心内容。

## 环境设置
列出实验所需的软件环境、依赖和库：
- Python 3.10.8
- Torch 2.1.2+cu121 / torch_geometric 2.5.3 /  等
- 其他依赖：`pip install -r requirements.txt`

## 数据集
- 介绍使用的数据集名称、来源、格式等。
- 如果数据集较大，提供数据下载链接或上传至云存储的说明。

## 代码结构
简要介绍仓库中各个文件的功能：
- `main.py`: 主程序文件。
- `models/Clients.py`: 客户端。`models/Server.py`: 服务聚合端。
- `models/lstm.py、st_gat2.py`:模型文件。
- `data_loader/dataloader.py`: 数据预处理文件。
- `requirements.txt`: 所需库和依赖。
  
## 实验步骤
描述如何运行实验：
1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
3. 运行实验：
   ```bash
   python main.py


