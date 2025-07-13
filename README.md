# 🐶🐱 猫狗图像分类器项目（基于 EfficientNet 和 PyTorch）

本项目实现了一个基于深度学习的图像二分类器，用于判断输入图片是猫还是狗。模型采用迁移学习方法，加载预训练的 EfficientNet-B0 作为特征提取器，并通过自定义分类头完成二分类任务。整个项目使用 PyTorch 构建，支持训练、预测、模型保存、命令行调用等功能。

---

## 📁 项目结构说明

之后写

---

## 🧠 项目亮点

- ✅ 使用 ImageNet 预训练 EfficientNet-B0，迁移学习提升精度
- ✅ 支持冻结/解冻主干网络，自由调整训练策略
- ✅ 支持命令行一键训练与推理
- ✅ 项目结构清晰，便于部署与迁移

---

## 📦 环境要求

建议使用 Python 虚拟环境（如 venv 或 conda）运行本项目。

- Python >= 3.8
- torch >= 2.0
- torchvision
- matplotlib
- pillow
- tqdm

安装依赖：

```bash
pip install -r requirements.txt

```

---

## 📂 数据准备
请准备如下结构的数据集文件夹，并放置于项目根目录下：

之后再说


图片命名格式需包含 "cat" 或 "dog" 字样以供自动标签生成
数据集较大，未上传至 GitHub，请本地准备（如来自 Kaggle 的 Dogs vs. Cats 数据集）

---

## 🚀 模型训练
运行以下命令启动训练：

```bash
python train.py --data_dir ./dataset/train --epochs 10 --batch_size 32 --lr 0.0003

```
训练结束后模型权重将自动保存为 model_best.pth。

---

## 🔍 模型预测（推理）

之后再写


---

## 📈 后续可扩展方向

不知道