#python "eval_depth_3dgs&lidar.py"
import numpy as np
import os
import cv2
from tqdm import tqdm

def calculate_comprehensive_metrics():
    # --- 1. 路径配置 (严格匹配你的文件系统) ---
    gt_dir = "data/waymo/002/lidar_depth"
    pred_dir = "output/Waymo/002_baseline/train/ours_7000"
    img_dir = "data/waymo/002/images" # 必须指定图片路径用于 RGB Canny
    
    # 严格遵循你之前要求的参数
    CANNY_LOW = 50
    CANNY_HIGH = 150
    
    frames = range(0, 21)
    cam_id = 0
    
    # 统计容器
    stats = {
        'rmse': [], 'abs_rel': [], 'mae': [],
        'edge_rmse': [], 'edge_abs_rel': [], 'edge_mae': []
    }

    # 保持你原始的输出表头
    print(f"{'帧编号':<10} | {'G-RMSE':<8} | {'E-RMSE':<8} | {'G-MAE':<8} | {'E-MAE':<8} | {'E-AbsRel'}")
    print("-" * 85)

    for i in frames:
        name = f"{i:06d}_{cam_id}"
        gt_path = os.path.join(gt_dir, f"{name}.npy")
        pred_path = os.path.join(pred_dir, f"{name}_depth.npy")
        img_path = os.path.join(img_dir, f"{name}.png")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path) or not os.path.exists(img_path):
            continue

        # 1. 加载数据
        gt_data = np.load(gt_path, allow_pickle=True).item()
        gt_mask = gt_data['mask']
        gt_values = gt_data['value']
        pred_full = np.load(pred_path).squeeze()
        raw_img = cv2.imread(img_path)

        # 2. 分辨率强制对齐 (Pred -> GT 尺寸)
        if pred_full.shape != gt_mask.shape:
            target_size = (gt_mask.shape[1], gt_mask.shape[0])
            pred_full = cv2.resize(pred_full, target_size, interpolation=cv2.INTER_LINEAR)

        # 3. 将一维 GT 深度值还原到二维平面
        gt_depth_2d = np.zeros(gt_mask.shape, dtype=np.float32)
        gt_depth_2d[gt_mask] = gt_values.flatten()

        # 4. 提取边缘掩码 (改动点：从 RGB 提取，而非从深度图提取)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        
        # 5. 定义掩码逻辑
        # 全局有效点：有GT、有预测、且在 80m 范围内
        valid_mask = gt_mask & (pred_full > 0) & (gt_depth_2d < 80)
        # 边缘有效点：全局有效点中的 RGB 边缘部分
        edge_mask = valid_mask & (edge_map > 0)

        # 6. 提取像素值
        g_gt, g_pred = gt_depth_2d[valid_mask], pred_full[valid_mask]
        e_gt, e_pred = gt_depth_2d[edge_mask], pred_full[edge_mask]

        # 7. 计算 Global 指标
        g_rmse = np.sqrt(np.mean((g_gt - g_pred) ** 2))
        g_abs_rel = np.mean(np.abs(g_gt - g_pred) / g_gt)
        g_mae = np.mean(np.abs(g_gt - g_pred))
        
        stats['rmse'].append(g_rmse)
        stats['abs_rel'].append(g_abs_rel)
        stats['mae'].append(g_mae)

        # 8. 计算 Edge 指标
        if len(e_gt) > 0:
            e_rmse = np.sqrt(np.mean((e_gt - e_pred) ** 2))
            e_abs_rel = np.mean(np.abs(e_gt - e_pred) / e_gt)
            e_mae = np.mean(np.abs(e_gt - e_pred))
            
            stats['edge_rmse'].append(e_rmse)
            stats['edge_abs_rel'].append(e_abs_rel)
            stats['edge_mae'].append(e_mae)
        else:
            e_rmse, e_mae, e_abs_rel = 0.0, 0.0, 0.0

        # 保持原始输出结构
        print(f"{name:<10} | {g_rmse:<8.3f} | {e_rmse:<8.3f} | {g_mae:<8.3f} | {e_mae:<8.3f} | {e_abs_rel:.4f}")

    # --- 最终汇总 ---
    if stats['rmse']:
        print("-" * 85)
        print(f"【学术汇报汇总指标 - Waymo 002 Sequence (RGB Edge Base)】")
        print(f"1. Global Metrics: RMSE: {np.mean(stats['rmse']):.4f}m, MAE: {np.mean(stats['mae']):.4f}m, AbsRel: {np.mean(stats['abs_rel']):.4f}")
        print(f"2. Edge Metrics:   RMSE: {np.mean(stats['edge_rmse']):.4f}m, MAE: {np.mean(stats['edge_mae']):.4f}m, AbsRel: {np.mean(stats['edge_abs_rel']):.4f}")
        print("-" * 85)
    else:
        print("未检测到有效数据，请检查路径。")

if __name__ == "__main__":
    calculate_comprehensive_metrics()