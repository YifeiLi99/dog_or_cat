import torch
import torch.nn as nn
import torch.optim as optim
from dataset.dataloader import get_dataloaders
from models.efficientnet_model import EfficientNetBinaryClassifier
from train_utils.train_utils import train_model
import os
from torch.optim.lr_scheduler import StepLR
import pandas as pd

# -----------------------------
# 设置设备：优先使用 GPU
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 全局参数配置
# -----------------------------
data_dir = "./dataset/train"  # 请确保该路径存在且包含 cat/ 与 dog/ 子文件夹
img_size = 224                # 输入图像尺寸
batch_size = 64               # 批次大小
num_epochs = 30               # 总训练轮数
learning_rate = 3e-5          # 学习率

# -----------------------------------------
# Windows 多进程训练保护（必须加！）
# -----------------------------------------
if __name__ == "__main__":

    # 1. 加载训练与验证数据集
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    # 2. 初始化模型（EfficientNet + 二分类头）
    model = EfficientNetBinaryClassifier(pretrained=True, freeze_backbone=False)
    print(f"特征层是否冻结: {all(not p.requires_grad for p in model.backbone.features.parameters())}")

    # 3. 定义损失函数（交叉熵）与优化器（Adam）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 初始化 scheduler
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. 启动训练主流程（含验证评估）
    train_logs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs)

    # 自动创建保存目录
    os.makedirs("weights", exist_ok=True)

    # 保存模型参数（state_dict 仅保存权重，不含结构）
    save_path = os.path.join("weights", "efficientnet_cat_dog03.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

    #保存log文件
    #确保 log 文件夹存在
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    logs_df = pd.DataFrame(train_logs)
    logs_df.to_csv('log/train_log.csv', index=False)

