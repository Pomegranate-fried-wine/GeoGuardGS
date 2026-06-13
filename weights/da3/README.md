# DA3 权重放置说明

本仓库不包含 Depth Anything 3 权重。

推荐放置为 Hugging Face snapshot 目录，而不是单个 `.safetensors` 文件：

```text
weights/da3/DA3-LARGE-1.1/
  config.json
  model.safetensors
  ...
```

配置中使用：

```yaml
geovit:
  enabled: true
  model_dir: weights/da3/DA3-LARGE-1.1
  local_files_only: true
```

DA3 在本项目中作为 boundary / ranking / side-structure prior，不作为 metric depth ground truth。
