import pandas as pd
import matplotlib.pyplot as plt
import os

"""
训练日志可视化脚本

作用：读取 log/train_log.csv 文件，绘制训练过程中的 Loss 和 Accuracy 曲线，
并保存为 log/training_curves.png，用于项目总结和训练效果展示。
"""

# 脚本固定读取 log/train_log.csv
project_root = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(project_root, 'train_log.csv')
logs = pd.read_csv(log_path)

# ✅ 创建画布，左右两张子图：Loss 曲线 + Accuracy 曲线
plt.figure(figsize=(12, 5))

# 📌 绘制 Loss 曲线（左图）
plt.subplot(1, 2, 1)
plt.plot(logs['epoch'], logs['train_loss'], label='Train Loss', color='blue')
plt.plot(logs['epoch'], logs['val_loss'], label='Validation Loss', color='orange')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 📌 绘制 Accuracy 曲线（右图）
plt.subplot(1, 2, 2)
plt.plot(logs['epoch'], logs['train_acc'], label='Train Accuracy', color='green')
plt.plot(logs['epoch'], logs['val_acc'], label='Validation Accuracy', color='red')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# ✅ 自适应布局，防止子图重叠
plt.tight_layout()

# 输出图片
plot_path = os.path.join(project_root, 'training_curves.png')
plt.savefig(plot_path)

# ✅ 可视化显示（可选，如果本地运行建议保留）
plt.show()