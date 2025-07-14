import torch
from pathlib import Path
from models.efficientnet_model import EfficientNetBinaryClassifier
from inference.inference import classify_and_organize_images

# ----------- 路径配置 -----------
BASE_DIR = Path(__file__).resolve().parent.parent  # 获取项目根目录
TEST_DIR = BASE_DIR / "dataset" / "test"
WEIGHTS_PATH = BASE_DIR / "weights" / "efficientnet_cat_dog.pth"
IMG_SIZE = 224  # 与训练保持一致
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- 初始化模型 -----------
model = EfficientNetBinaryClassifier(pretrained=False)

# ----------- 推理并分类整理图像 -----------
classify_and_organize_images(
    model=model,
    src_folder=TEST_DIR,
    dst_folder=TEST_DIR,
    weight_path=WEIGHTS_PATH,
    device=DEVICE,
    img_size=IMG_SIZE
)
