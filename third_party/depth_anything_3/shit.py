#用于衡量da3与lidar的深度信息指标，引入canny算子

import numpy as np
import cv2
import os

def calculate_metrics_core(gt_values, pred_values):
    """核心指标计算：RMSE, MAE, AbsRel"""
    if len(gt_values) == 0:
        return 0.0, 0.0, 0.0
    rmse = np.sqrt(np.mean((gt_values - pred_values) ** 2))
    mae = np.mean(np.abs(gt_values - pred_values))
    abs_rel = np.mean(np.abs(gt_values - pred_values) / gt_values)
    return rmse, mae, abs_rel

def main():
    # --- 1. 路径配置 (严格按照你的 DA3 环境路径) ---
    root_path = "/home/hch/projects/street_gaussians-main/Depth-Anything-3/Depth-Anything-3-main"
    gt_path = os.path.join(root_path, "test_lidar.npy")
    pred_path = os.path.join(root_path, "test_depth_metric.npy")
    img_path = os.path.join(root_path, "test.png")
    
    # --- 2. 严格参数对齐 (必须与 3DGS 脚本的 50/150 一致) ---
    CANNY_LOW = 50
    CANNY_HIGH = 150

    if not all(os.path.exists(p) for p in [gt_path, pred_path, img_path]):
        print("❌ 错误：当前目录下缺少 test_lidar.npy, test_depth_metric.npy 或 test.png")
        return

    # --- 3. 加载与格式对齐 ---
    # 加载 LiDAR 数据
    gt_data = np.load(gt_path, allow_pickle=True).item()
    gt_mask = gt_data['mask']
    gt_values_1d = gt_data['value'].flatten()
    
    # 加载预测深度 (DA3 物理深度)
    pred_full = np.load(pred_path).squeeze()
    
    # 加载 RGB 原图
    raw_img = cv2.imread(img_path)

    # 尺寸强制对齐 (Pred -> GT 尺寸)
    if pred_full.shape != gt_mask.shape:
        pred_full = cv2.resize(pred_full, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 将一维 GT 还原到二维平面
    gt_depth_2d = np.zeros(gt_mask.shape, dtype=np.float32)
    gt_depth_2d[gt_mask] = gt_values_1d

    # --- 4. 提取 RGB 边缘掩码 (核心：与 3DGS 脚本逻辑完全闭环) ---
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    #edge_map = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    # 修改 shit.py 这一行再跑一次
    edge_map = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_map = cv2.dilate(edge_map, np.ones((3,3), np.uint8), iterations=2)
    # --- 5. 掩码筛选 ---
    # 全局有效点：有GT、有预测、且在 80m 范围内
    valid_mask = gt_mask & (pred_full > 0) & (gt_depth_2d < 80)
    # 边缘有效点：全局点中的 RGB 边缘部分
    edge_mask = valid_mask & (edge_map > 0)

    # --- 6. 提取像素值 ---
    g_gt, g_pred = gt_depth_2d[valid_mask], pred_full[valid_mask]
    e_gt, e_pred = gt_depth_2d[edge_mask], pred_full[edge_mask]

    # --- 7. 计算指标 ---
    g_rmse, g_mae, g_abs_rel = calculate_metrics_core(g_gt, g_pred)
    e_rmse, e_mae, e_abs_rel = calculate_metrics_core(e_gt, e_pred)

    # --- 8. 格式化输出报告 ---
    print("\n" + "="*55)
    print("      DA3 vs LiDAR 评估报告 (RGB 边缘对齐版)")
    print("="*55)
    print(f"{'指标类型':<12} | {'RMSE':<8} | {'MAE':<8} | {'AbsRel':<8} | {'点数'}")
    print("-" * 55)
    print(f"{'Global':<12} | {g_rmse:<8.3f} | {g_mae:<8.3f} | {g_abs_rel:<8.4f} | {len(g_gt)}")
    print(f"{'Edge (RGB)':<12} | {e_rmse:<8.3f} | {e_mae:<8.3f} | {e_abs_rel:<8.4f} | {len(e_gt)}")
    print("="*55)

    # 对比提示
    print(f"\n💡 请记录上面的 Edge AbsRel ({e_abs_rel:.4f})，")
    print(f"   并与运行 python \"eval_depth_3dgs&lidar.py\" 得到的汇总 E-AbsRel 进行对比。")

if __name__ == "__main__":
    main()