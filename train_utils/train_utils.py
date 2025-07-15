import torch
import torch.nn as nn
from tqdm import tqdm

class EarlyStopping:
    """
    早停机制，用于在验证集指标长时间未提升时提前终止训练，防止过拟合或无效训练。

    :param patience: 在验证集损失未提升的情况下，允许的连续最大 epoch 数
    :type patience: int
    :param verbose: 是否打印早停过程信息
    :type verbose: bool

    Attributes:
    - best_loss (float): 当前训练过程中记录的最低验证损失
    - best_state_dict (dict): 最佳模型状态字典
    - counter (int): 连续未提升的 epoch 次数
    - early_stop (bool): 是否触发早停
    """

    def __init__(self, patience=5, verbose=True):
        """
        初始化 EarlyStopping 类。

        :param patience: 验证集损失连续未提升时允许的最大 epoch 数（默认 5）
        :type patience: int
        :param verbose: 是否输出早停监测信息（默认 True）
        :type verbose: bool
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        """
        进行早停判断逻辑。

        :param val_loss: 当前 epoch 的验证集损失
        :type val_loss: float
        :param model: 当前训练中的模型实例
        :type model: torch.nn.Module
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict()
            self.counter = 0
            if self.verbose:
                tqdm.write(f"Validation loss improved to {val_loss:.4f}, saving model checkpoint.")
        else:
            self.counter += 1
            if self.verbose:
                tqdm.write(f"EarlyStopping counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num=None):
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
    :param epoch_num: 当前epoch轮数
    :type epoch_num: int

    :return: 当前轮次的平均损失和准确率（范围为 0–1）
    :rtype: Tuple[float, float]
    """

    #切换为训练模式
    model.train()
    #初始化累计值：总损失、正确预测数、样本总数
    running_loss = 0.0
    correct = 0
    total = 0

    #进度条
    #desc = f"Epoch {epoch_num}" if epoch_num is not None else "Training"
    pbar = tqdm(dataloader, desc="Training", dynamic_ncols=True)

    #将图像和标签移动到 GPU 或 CPU
    for images, labels in pbar:
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

        # 实时展示 loss 和当前 batch 的准确率
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

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
        pbar = tqdm(dataloader, desc="Validation", dynamic_ncols=True)
        for images, labels in pbar:
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

            #进度条展示数据
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
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
    :param scheduler: 学习率调整
    :type scheduler: StepLR
    :param device: 运行设备 如 torch.device("cuda") 或 torch.device("cpu")
    :type device: torch.device
    :param num_epochs: 总训练轮数
    :type num_epochs: int

    :return: 无（但会打印每轮的损失与准确率）
    :rtype: None
    """
    #确保模型在 CPU 或 GPU 上运行
    model.to(device)

    #初始化早停
    early_stopping = EarlyStopping(patience=5)

    #控制整个训练过程
    for epoch in range(num_epochs):
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
        tqdm.write("-" * 30)

        # 训练一个完整 epoch，返回损失与准确率
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_num=epoch+1)

        # 在验证集上评估模型
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 调整学习率
        scheduler.step()

        #查看指标
        tqdm.write(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        tqdm.write(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        tqdm.write(f"{'=' * 50}\n")

        #看看是否需要早停
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            tqdm.write(f"\n⏹️ Early stopping triggered at epoch {epoch + 1}")
            break

    # 恢复最佳模型
    model.load_state_dict(early_stopping.best_state_dict)
    tqdm.write(f"\nLoaded best model with val_loss = {early_stopping.best_loss:.4f}")

