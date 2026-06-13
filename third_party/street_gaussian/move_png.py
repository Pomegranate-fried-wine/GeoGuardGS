import os
import shutil
from glob import glob

# 确认路径
image_dir = "/home/hch/projects/street_gaussians-main/data/waymo/002/images"

# 1. 使用 ** 递归搜索子文件夹下所有的 png
# 例如它会找到 images/0/000000.png
sub_images = glob(os.path.join(image_dir, "**", "*.png"), recursive=True)

print(f"DEBUG: 扫描到图片总数: {len(sub_images)} 张")

move_count = 0
for img_path in sub_images:
    # 排除已经在根目录下的图片，防止重复处理
    if os.path.dirname(img_path) == image_dir:
        continue
        
    filename = os.path.basename(img_path) # 000000.png
    cam_id = os.path.basename(os.path.dirname(img_path)) # 获取文件夹名，如 '0'
    
    # 构造原作者代码最喜欢的平铺名: 000000_0.png
    # 这样代码里的 x.split('.')[0][-1] 就能准确拿到最后的 '0'
    new_name = f"{filename.split('.')[0]}_{cam_id}.png"
    target_path = os.path.join(image_dir, new_name)
    
    # 移动并改名
    shutil.move(img_path, target_path)
    move_count += 1

print(f"成功复原了 {move_count} 张图片到平铺结构。")

# 2. 清理空的子文件夹
for i in range(5):
    subdir = os.path.join(image_dir, str(i))
    if os.path.exists(subdir):
        try:
            os.rmdir(subdir)
            print(f"已清理空目录: {subdir}")
        except:
            pass