import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from models.efficientnet_model import EfficientNetBinaryClassifier
from tqdm import tqdm

# ----------- 路径配置 -----------
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset" / "test"
WEIGHTS_PATH = BASE_DIR / "weights" / "efficientnet_cat_dog01.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- 数据准备 -----------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"测试集样本数: {len(test_dataset)}, 类别: {test_dataset.classes}")

# ----------- 模型加载 -----------
model = EfficientNetBinaryClassifier(pretrained=False)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ----------- 推理 + 计算准确率 -----------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"测试集准确率: {accuracy:.2f}%")
