import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetBinaryClassifier(nn.Module):
    """
    基于 EfficientNet-B0 构建的二分类神经网络模型。

    用于猫狗图像分类任务，支持加载 ImageNet 预训练权重并选择是否冻结特征提取层。
    """

    def __init__(self, pretrained=True, freeze_backbone=True):
        """
        初始化 EfficientNetBinaryClassifier 模型

        :param pretrained: 是否加载 ImageNet 预训练权重
        :type pretrained: bool
        :param freeze_backbone: 是否冻结特征提取部分参数（提高训练速度，适用于小样本）
        :type freeze_backbone: bool
        """
        super(EfficientNetBinaryClassifier, self).__init__()

        # 加载 EfficientNet-B0 预训练模型（包含 features + classifier）
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = efficientnet_b0(weights=weights)

        # 冻结特征提取部分参数（即 self.backbone.features）
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False  # 不参与反向传播

        # 获取原始分类器中 Linear 层的输入特征维度（默认 1280）
        in_features = self.backbone.classifier[1].in_features

        # 替换原始 1000 类输出头为二分类结构
        # 保留 Dropout 以抑制过拟合（0.3 是官方默认值）
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)  # 输出为两个类别：猫/狗
        )

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入图像张量，形状为 (B, C, H, W)
        :type x: torch.Tensor
        :return: 分类输出的 logits 张量，形状为 (B, 2)
        :rtype: torch.Tensor
        """
        return self.backbone(x)
