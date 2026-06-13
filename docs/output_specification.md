# 输出规范

最终实验应能展示以下内容：

1. 不同分支 RGB 渲染对比图。
2. 不同分支 depth map。
3. depth error heatmap。
4. DA3 boundary-risk overlay。
5. LiDAR sparse depth overlay。
6. high-error / high-risk selected region panels。
7. contribution top-K Gaussian 可视化。
8. group responsibility 可视化。
9. opacity decay 前后 panel。
10. controlled Gaussian opacity trace。
11. global RGB metrics 表。
12. LiDAR-valid geometry metrics 表。
13. DA3 top-risk region metrics 表。
14. boundary / depth-edge metrics 表。
15. repair candidate 统计表。
16. safety audit 表。
17. 每 500 轮训练趋势曲线。
18. final comparison table。

推荐目录：

```text
outputs/<exp_name>/
  checkpoints/
  periodic_eval/
  feedback_controller/
  metrics/
  final_eval/
```
