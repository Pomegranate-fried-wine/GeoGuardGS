import numpy as np
import cv2
import os
import argparse

def visualize_lidar_final(depth_path, save_path=None, max_depth=80.0):
    if not os.path.exists(depth_path):
        print(f"File not found: {depth_path}")
        return

    try:
        # 1. 加载字典并还原深度
        raw_data = np.load(depth_path, allow_pickle=True)
        data_dict = raw_data.item()
        
        values = np.array(data_dict['value']).flatten()
        mask = np.array(data_dict['mask'])
        H, W = mask.shape
        
        # 2. 物理深度转归一化灰度 (0-255)
        # 这一步必须严格匹配渲染器的归一化逻辑
        depth_map = np.zeros((H, W), dtype=np.float32)
        depth_map[mask] = values
        
        # 线性映射到 0-255
        depth_viz = np.clip(depth_map / max_depth, 0, 1) * 255
        depth_viz = depth_viz.astype(np.uint8)

        # 3. 应用颜色映射 (使用与代码一致的 TURBO)
        # cv2.applyColorMap 返回的是 BGR 格式
        color_depth = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
        
        # 4. 关键：修正通道顺序 (从 BGR 转为 RGB)
        # 这样在 VS Code 或普通看图软件里颜色才正确
        color_depth_rgb = cv2.cvtColor(color_depth, cv2.COLOR_BGR2RGB)

        # 5. 背景处理：将无效点强行设为黑色 (0, 0, 0)
        color_depth_rgb[~mask] = 0

    except Exception as e:
        print(f"Logic Error during processing: {e}")
        return

    # 6. 保存
    if save_path:
        # 使用 OpenCV 保存时需要转回 BGR，或者用 imageio 保存 RGB
        # 这里为了简单统一，直接用 cv2 保存，所以存之前再转回 BGR
        final_save = cv2.cvtColor(color_depth_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_save)
        print(f"Successfully saved to {save_path}")
    else:
        # 弹窗显示通常需要 BGR
        cv2.imshow("Check Consistency", color_depth) 
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, default="./final_consistent_gt.png")
    parser.add_argument("--max_depth", type=float, default=80.0) # 务必确认配置里的值
    args = parser.parse_args()
    visualize_lidar_final(args.inp, args.out, args.max_depth)