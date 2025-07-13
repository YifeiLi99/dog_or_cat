import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    执行模型的单轮训练过程。

    包括前向传播、反向传播、参数更新，并计算当前训练轮次的平均损失与准确率。

    :param model: 需要训练的神经网络模型
    :type model: torch.nn.Module
    :param dataloader: 训练数据加载器，提供按批次的训练数据
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: 损失函数，用于衡量预测结果与真实标签之间的误差
    :type criterion: torch.nn.Module
    :param optimizer: 优化器，用于更新模型参数
    :type optimizer: torch.optim.Optimizer
    :param device: 运算设备，通常为 "cpu" 或 "cuda"
    :type device: torch.device

    :return: 当前轮次的平均损失和准确率（范围为 0–1）
    :rtype: Tuple[float, float]
    """

    #切换为训练模式
    model.train()
    #初始化累计值：总损失、正确预测数、样本总数
    running_loss = 0.0
    correct = 0
    total = 0

    #将图像和标签移动到 GPU 或 CPU
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 手动清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 根据梯度更新权重
        optimizer.step()

        # 计算总loss
        running_loss += loss.item() * images.size(0)
        # 获取预测类别（按最大概率）
        _, predicted = torch.max(outputs, 1)
        # 统计预测对的个数
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # 计算平均损失和准确率
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    在验证集上评估模型性能，计算平均损失与准确率。

    :param model: 要评估的神经网络模型
    :type model: torch.nn.Module
    :param dataloader: 验证集的数据加载器
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: 用于计算损失的损失函数（如 nn.CrossEntropyLoss）
    :type criterion: torch.nn.Module
    :param device: 计算设备（如 "cuda" 或 "cpu"）
    :type device: torch.device
    :return: 平均损失与准确率
    :rtype: Tuple[float, float]
    """
    # 评估模式，不启用 Dropout/BN 更新
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度，节省显存与加速推理
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            #累加总损失时乘以样本数
            running_loss += loss.item() * images.size(0)
            #返回最大概率的类别索引
            _, predicted = torch.max(outputs, 1)
            #统计预测正确的样本数
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    完整训练流程，包括多个 epoch 的训练与验证，并输出每一轮的性能。

    :param model: 要训练的模型
    :type model: nn.Module
    :param train_loader: 训练集的 DataLoader
    :type train_loader: torch.utils.data.DataLoader
    :param val_loader: 验证集的 DataLoader
    :type val_loader: torch.utils.data.DataLoader
    :param criterion: 损失函数
    :type criterion: nn.Module
    :param optimizer: 优化器
    :type optimizer: torch.optim.Optimizer
    :param device: 运行设备 如 torch.device("cuda") 或 torch.device("cpu")
    :type device: torch.device
    :param num_epochs: 总训练轮数
    :type num_epochs: int

    :return: 无（但会打印每轮的损失与准确率）
    :rtype: None
    """
    #确保模型在 CPU 或 GPU 上运行
    model.to(device)

    #控制整个训练过程
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # 训练一个完整 epoch，返回损失与准确率
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # 在验证集上评估模型
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")