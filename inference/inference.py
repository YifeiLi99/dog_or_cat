import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.efficientnet_model import EfficientNetBinaryClassifier

# ----------- 路径配置 -----------
BASE_DIR = Path(__file__).resolve().parent.parent  # 获取项目根目录
WEIGHTS_PATH = BASE_DIR / "weights" / "efficientnet_cat_dog.pth"
IMG_SIZE = 224  # 与训练保持一致

# ----------- 图像预处理 -----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# ----------- 推理函数 -----------
def predict_image(image_path: Path, device: str = 'cpu') -> int:
    """
    对单张图像进行推理分类（猫或狗）

    :param image_path: 图像路径
    :type image_path: Path
    :param device: 推理设备（'cpu' 或 'cuda'）
    :type device: str
    :return: 类别标签（0 = Cat, 1 = Dog）
    :rtype: int
    """
    if not image_path.exists():
        raise FileNotFoundError(f"图像路径不存在: {image_path}")

    # 加载图像并预处理
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    # 加载模型并加载权重
    model = EfficientNetBinaryClassifier(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 推理
    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return predicted

# ----------- 推理测试入口 -----------
if __name__ == "__main__":
    # 输入图像路径
    image_path = BASE_DIR / "dataset" / "test" / "1.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred = predict_image(image_path, device)
    label = "Dog 🐶" if pred == 1 else "Cat 🐱"
    print(f"预测结果: {label}")
