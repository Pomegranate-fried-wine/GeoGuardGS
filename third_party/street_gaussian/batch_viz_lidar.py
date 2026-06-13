import numpy as np
import cv2
import os
import argparse
from pathlib import Path

def process_batch(input_dir, output_dir, start_idx, end_idx, cam_id=0, max_depth=80.0):
    """
    批量将 LiDAR .npy 文件转换为统一映射的深度图 PNG
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始批量处理：{start_idx} -> {end_idx}, 目标文件夹: {output_dir}")

    for i in range(start_idx, end_idx + 1):
        # 构造文件名，例如 000005_0.npy
        file_name = f"{i:06d}_{cam_id}"
        inp_path = os.path.join(input_dir, f"{file_name}.npy")
        out_path = os.path.join(output_dir, f"{file_name}_gt_depth.png")

        if not os.path.exists(inp_path):
            print(f"跳过：找不到文件 {inp_path}")
            continue

        try:
            # 1. 加载数据
            raw_data = np.load(inp_path, allow_pickle=True)
            data_dict = raw_data.item()
            
            values = np.array(data_dict['value']).flatten()
            mask = np.array(data_dict['mask'])
            H, W = mask.shape
            
            # 2. 物理深度映射到 0-255 灰度
            depth_map = np.zeros((H, W), dtype=np.float32)
            depth_map[mask] = values
            depth_viz = (np.clip(depth_map / max_depth, 0, 1) * 255).astype(np.uint8)

            # 3. 应用 Turbo 颜色映射 (OpenCV 逻辑)
            color_depth = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
            
            # 4. 后置处理：背景设为黑色，并确保通道顺序
            # 注意：cv2.imwrite 期望 BGR，所以我们不需要转 RGB
            color_depth[~mask] = 0
            
            # 5. 保存结果
            cv2.imwrite(out_path, color_depth)
            print(f"已生成 [{i}/{end_idx}]: {out_path}")

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    print("\n批量处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--inp_dir", type=str, default="./data/waymo/002/lidar_depth", help="LiDAR npy 文件夹路径")
    parser.add_argument("--out_dir", type=str, default="./lidar_viz_batch", help="保存图片的文件夹")
    # 范围参数
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=20, help="结束索引")
    parser.add_argument("--cam", type=int, default=0, help="相机编号 (_0, _1 等)")
    # 尺度参数
    parser.add_argument("--max_depth", type=float, default=80.0, help="统一的最大深度尺度")
    
    args = parser.parse_args()
    
    process_batch(args.inp_dir, args.out_dir, args.start, args.end, args.cam, args.max_depth)