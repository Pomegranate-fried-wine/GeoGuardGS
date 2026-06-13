# LD_LIBRARY_PATH=/usr/lib/wsl/lib python depth_anything.py
import os
import torch
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3
from sklearn.linear_model import LinearRegression

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 正在启动 DA3 官方对齐流水线 (Device: {device})...")
    
    # --- 1. 模型加载 ---
    try:
        model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to(device).eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # --- 2. 数据读取 ---
    img_path = "test.png"
    lidar_path = "test_lidar.npy"
    
    if not os.path.exists(img_path) or not os.path.exists(lidar_path):
        print("❌ 错误：当前目录下缺少 test.png 或 test_lidar.npy")
        return
        
    raw_img = cv2.imread(img_path)
    img_h, img_w = raw_img.shape[:2]

    # --- 3. 模型推理 (获取仿射不变深度) ---
    print("🧠 正在进行稠密深度推理...")
    with torch.no_grad():
        prediction = model.inference([raw_img])
        raw_da3 = prediction.depth[0] # 模型原始输出 (Inverse-like)

    # --- 4. 严格物理标定 (基于 17661 个 LiDAR 点) ---
    print("🎯 正在根据 LiDAR 进行仿射对齐 (Affine Alignment)...")
    try:
        lidar_data = np.load(lidar_path, allow_pickle=True).item()
        mask = lidar_data['mask']
        li_val = lidar_data['value'].flatten()
        
        # 尺寸对齐：DA3 -> LiDAR 分辨率
        da3_res = cv2.resize(raw_da3, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 提取点对点对齐样本
        rows, cols = np.where(mask)
        da3_samples = da3_res[rows, cols]
        
        # 线性回归：Metric_Depth = s * DA3_Raw + t
        X = da3_samples.reshape(-1, 1)
        y = li_val.reshape(-1, 1)
        
        reg = LinearRegression().fit(X, y)
        s = reg.coef_[0][0]
        t = reg.intercept_[0]
        r2 = reg.score(X, y)
        
        print(f"📊 标定完成：Scale={s:.4f}, Shift={t:.4f}, R²={r2:.4f}")

        # --- 5. 生成全局物理 NPY (用于误差计算) ---
        # 直接使用全图原始输出进行变换，保留所有棱角细节
        metric_depth = s * da3_res + t
        # 物理量程截断 (与 3DGS 渲染器对齐)
        metric_depth = np.clip(metric_depth, 0.1, 80.0)
        
        # 强制保存为 float32 的 npy
        np.save("test_depth_metric.npy", metric_depth.astype(np.float32))
        print("💾 物理深度数据已保存: test_depth_metric.npy")

        # --- 6. 3DGS 风格化绘图 (预览一致性) ---
        # 遵循 3DGS 线性映射 [0, 80] -> [0, 255]
        # 配色方案：近处(值小)蓝色 -> 远处(值大)红色
        max_viz_range = 80.0
        norm_val = np.clip(metric_depth / max_viz_range, 0, 1)
        depth_viz_255 = (norm_val * 255).astype(np.uint8)
        
        # 应用 TURBO 色图
        color_map_bgr = cv2.applyColorMap(depth_viz_255, cv2.COLORMAP_TURBO)
        
        # 存图逻辑：按照你的 LiDAR 代码习惯，直接 cv2 存 BGR
        # 确保图片在预览器里极性正确
        cv2.imwrite("da3_final_bob_scheme.png", color_map_bgr)
        print("✨ 3DGS 风格预览图已生成: da3_final_bob_scheme.png")

    except Exception as e:
        print(f"❌ 对齐过程发生逻辑错误: {e}")

if __name__ == "__main__":
    main()