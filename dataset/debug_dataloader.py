from dataset.dataloader import get_dataloaders
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # 相对路径构建（假设当前文件在 dataset/ 中）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "train")

    # 加载数据
    train_loader, _ = get_dataloaders(train_dir)

    # 获取一批图像
    images, labels = next(iter(train_loader))

    # 可视化前4张
    for i in range(4):
        img = images[i].permute(1, 2, 0) * 0.5 + 0.5  # [-1,1] → [0,1]
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title("Dog" if labels[i] == 1 else "Cat")
        plt.axis("off")

    plt.show()
