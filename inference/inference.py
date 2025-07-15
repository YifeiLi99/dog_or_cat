import shutil
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.efficientnet_model import EfficientNetBinaryClassifier
import torch.nn.functional as F

# ----------- 路径配置 -----------
BASE_DIR = Path(__file__).resolve().parent.parent  # 获取项目根目录
WEIGHTS_PATH = BASE_DIR / "weights" / "efficientnet_cat_dog03.pth"
IMG_SIZE = 224  # 与训练保持一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- 图像预处理 -----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------- 推理函数 -----------
def predict_image(model, image, device):
    """
        对单张图像进行推理分类（猫或狗），返回类别和预测概率。

        :param model: 模型
        :type model: EfficientNetBinaryClassifier
        :param image: 图像
        :type image: Path
        :param device: 推理设备（'cpu' 或 'cuda'）
        :type device: device
        :return: (类别标签（0 = Cat, 1 = Dog）, 预测概率)
        :rtype: Tuple[int, float]
    """
    # 加载图像并预处理
    image_tensor = transform(image).unsqueeze(0).to(device)
    # 变成推理模式
    model.eval()

    # 推理
    with torch.no_grad():
        # logits 输出
        output = model(image_tensor)
        # 转为概率
        probs = torch.softmax(output, dim=1)
        prob, cls = torch.max(probs, dim=1)

    return cls.item(), prob.item()

# ----------- 图片分类函数 -----------
def classify_and_organize_images(model, src_folder, dst_folder, weight_path, device="cpu", img_size=224):
    """
    使用训练好的模型对 src_folder 中的所有图像进行推理（cat/dog），并将图像复制到
    dst_folder/cat 或 dst_folder/dog 文件夹中。

    :param model: 训练好的 PyTorch 模型实例
    :param src_folder: 源图像文件夹路径（Path 或 str）
    :param dst_folder: 分类后图像保存的目标根路径（Path 或 str）
    :param weight_path: 模型权重文件路径（Path 或 str）
    :param device: 推理设备（"cuda" 或 "cpu"）
    :param img_size: 输入图像尺寸（需与训练时一致）
    """
    # 转换为 Path 对象，确保兼容性
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    weight_path = Path(weight_path)

    # 加载模型权重并切换到推理模式
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval().to(device)

    # 创建目标文件夹（cat 和 dog）
    (dst_folder / "cat").mkdir(parents=True, exist_ok=True)
    (dst_folder / "dog").mkdir(parents=True, exist_ok=True)

    # 定义图像预处理流程（必须与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # 遍历所有图像文件进行推理与保存
    for img_path in src_folder.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[跳过] 无法读取图像: {img_path.name}，原因: {e}")
            continue

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, dim=1).item()

        label = "dog" if predicted == 1 else "cat"
        target_path = dst_folder / label / img_path.name
        shutil.copy2(img_path, target_path)

    print(f"✅ 分类完成！结果保存在：{dst_folder.resolve()}")

# ----------- 推理测试入口 -----------
if __name__ == "__main__":
    # 输入图像路径
    image_path = BASE_DIR / "dataset" / "test" / "cat" / "7846.jpg"

    # ----------- 模型加载 -----------
    model = EfficientNetBinaryClassifier(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    #开始推理
    cls, prob = predict_image(model, image_path, DEVICE)
    label = "Dog 🐶" if cls == 1 else "Cat 🐱"
    print(f"预测结果: {label} (置信度: {prob:.4f})")

