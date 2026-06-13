# StreetGS 权重 / checkpoint 放置说明

本仓库不上传大型 checkpoint。

如需 resume 或评估，请将 checkpoint 放到：

```text
weights/streetgs/<experiment_name>/
```

或在配置中显式指定：

```yaml
train:
  start_checkpoint: /path/to/checkpoint.pth
```

请勿将 `.pth`、`.pt`、`.ckpt` 文件提交到 GitHub。
