import torch
import torch.nn as nn
from torchvision import models

#构建基于efficient预训练模型的二分类器类
class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(EfficientNetBinaryClassifier, self).__init__()

        # 1. 加载预训练模型
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # 2. 冻结特征提取部分（可选）
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # 3. 替换分类头（原来是 1000 类）
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)  # 二分类
        )

    def forward(self, x):
        return self.backbone(x)
