import os
import shutil
from pathlib import Path

# 源目录：原始图片都在这里
source_dir = Path(r"/\dataset\test\test")

# 目标目录：猫狗分类后的目标路径
cat_dir = Path(r"/\dataset\test\cat")
dog_dir = Path(r"/\dataset\test\dog")

# 创建目标目录（如果不存在）
cat_dir.mkdir(parents=True, exist_ok=True)
dog_dir.mkdir(parents=True, exist_ok=True)

# 遍历文件并分类
for filename in os.listdir(source_dir):
    if filename.lower().endswith(".jpg"):
        src_path = source_dir / filename
        if "cat" in filename.lower():
            dst_path = cat_dir / filename
        elif "dog" in filename.lower():
            dst_path = dog_dir / filename
        else:
            continue  # 如果不是cat或dog，跳过
        shutil.move(str(src_path), str(dst_path))

print("图片已成功分类移动到 cat/ 和 dog/ 文件夹")
