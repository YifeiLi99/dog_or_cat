import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class DogOrCatDataset(Dataset):
    """
    自定义数据集类，用于加载猫狗图像分类数据。

    继承自 torch.utils.data.Dataset，用于将图像路径和标签配对并提供数据加载功能。
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
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像并转换为 RGB 模式，确保一致性（防止灰度图出错）
        image = Image.open(img_path).convert("RGB")

        # 应用 transform 操作（若指定）
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(data_dir, img_size=224, batch_size=32, valid_ratio=0.2):
    """
    构建训练集和验证集的 DataLoader。

    自动读取图像文件路径，解析猫狗标签，并划分训练集与验证集。

    :param data_dir: 图像数据所在目录，应包含 cat.x.jpg 和 dog.x.jpg 命名的文件
    :type data_dir: str
    :param img_size: 输入图像的目标大小，图像将被 resize 为 (img_size, img_size)
    :type img_size: int
    :param batch_size: 每个批次加载的图像数量
    :type batch_size: int
    :param valid_ratio: 验证集所占的比例（如 0.2 表示 20% 用于验证）
    :type valid_ratio: float
    :return: 训练集 DataLoader 和 验证集 DataLoader
    :rtype: Tuple[DataLoader, DataLoader]
    """

    # 1. 收集所有图像文件路径（仅限 .jpg 格式）
    all_images = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".jpg")
    ]

    # 2. 通过文件名判断标签：含 'dog' 为 1，'cat' 为 0
    all_labels = [1 if "dog" in os.path.basename(p).lower() else 0 for p in all_images]

    # 3. 使用 stratified 分层抽样划分训练集与验证集，保持猫狗比例一致
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_images,
        all_labels,
        test_size=valid_ratio,
        stratify=all_labels,
        random_state=42
    )

    # 4. 定义图像变换（图像增强 + 标准化）
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # 像素归一化到 [-1, 1]
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 5. 构建数据集对象
    train_dataset = DogOrCatDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DogOrCatDataset(val_paths, val_labels, transform=val_transform)

    # 6. 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
