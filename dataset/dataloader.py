import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DogOrCatDataset(Dataset):
    """
    自定义数据集类，用于加载猫狗图像分类数据

    继承自 torch.utils.data.Dataset，用于将图像路径和标签配对并提供数据加载功能
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        初始化数据集对象

        :param image_paths: 图像文件路径列表，每个元素为一个图像的绝对路径
        :type image_paths: List[str]
        :param labels: 与 image_paths 一一对应的标签列表（0 表示猫，1 表示狗）
        :type labels: List[int]
        :param transform: 图像预处理或增强操作（如 Resize、ToTensor、Normalize）
        :type transform: torchvision.transforms.Compose 或 None
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        返回数据集的样本数量

        :return: 样本总数
        :rtype: int
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本，包括图像和标签。

        :param idx: 样本索引
        :type idx: int
        :return: (图像张量, 标签整数)
        :rtype: Tuple[torch.Tensor, int]
        """
        # 从 image_paths 和 labels 中取出当前索引对应的图像路径和标签
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像并转换为 RGB 模式，确保一致性（防止灰度图出错）
        image = Image.open(img_path).convert("RGB")

        # 应用 transform 操作
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir, img_size=224, batch_size=32, valid_ratio=0.2):
    """
    构建训练集与验证集的 DataLoader，适用于 ImageNet-style 分类结构

    目录结构要求如下：

    data_dir/
    ├── cat/
    │   ├── cat.0.jpg
    │   └── ...
    └── dog/
        ├── dog.0.jpg
        └── ...

    :param data_dir: 包含 cat/ 和 dog/ 子文件夹的主目录路径
    :type data_dir: str
    :param img_size: 图像将被 resize 到的尺寸，适配模型输入
    :type img_size: int
    :param batch_size: 每个 batch 中的数据量
    :type batch_size: int
    :param valid_ratio: 验证集占总数据的比例
    :type valid_ratio: float
    :return: 用于训练和验证的两个 DataLoader
    :rtype: Tuple[DataLoader, DataLoader]
    """

    # 初始化图像路径与标签列表；从子文件夹名确定标签（cat=0，dog=1）
    all_images = []
    all_labels = []
    class_to_label = {"cat": 0, "dog": 1}

    # 遍历子文件夹（每类一个），如 "cat"、"dog"
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        # lower 避免大小写差异导致错误
        # 抽取标签，cat-0，dog-1
        label = class_to_label[class_name.lower()]

        # 遍历当前类别文件夹下所有 .jpg 图像
        for fname in os.listdir(class_dir):
            if fname.endswith(".jpg"):
                # 添加图像路径和对应标签
                all_images.append(os.path.join(class_dir, fname))
                all_labels.append(label)

    # 划分训练/验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_images, all_labels,
        test_size=valid_ratio,
        # 使用分层抽样（stratify）按类别比例划分训练集和验证集
        stratify=all_labels,
        # 设置随机种子以保证划分结果可复现
        random_state=42
    )

    # 训练集 transform：包括数据增强与标准化
    train_transform = transforms.Compose([
        # 缩放图像尺寸到统一大小
        transforms.Resize((img_size, img_size)),
        # 增加图像变换的多样性（增强泛化）
        transforms.RandomHorizontalFlip(),
        # 转为 torch.Tensor，且归一化到 [0, 1]
        transforms.ToTensor(),
        # 冻结时，用 ImageNet 归一化
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # 验证集 transform：不使用数据增强，仅进行 resize 和标准化
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # 构建 Dataset 和 DataLoader
    train_dataset = DogOrCatDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DogOrCatDataset(val_paths, val_labels, transform=val_transform)

    # 构建训练集加载器：启用随机打乱；使用 4 个子进程并行加载数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 构建验证集加载器：不打乱顺序，便于结果复现
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
