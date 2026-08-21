# Part 3 遥感数据流水线实操

本目录提供一个不依赖外部下载的端到端示例：

```text
合成 4 波段 GeoTIFF + GeoJSON 标注
-> 对齐栅格化 mask
-> 滑窗切片
-> 影像/mask 同步增强
-> PyTorch Dataset
```

安装依赖：

```bash
pip install -r scripts/part3_data_pipeline/requirements.txt
```

运行快速测试：

```bash
python scripts/part3_data_pipeline/smoke_test.py
```

随后从仓库根目录打开：

```text
notebooks/practices/03-remote-sensing-data-pipeline.ipynb
```

Notebook 的运行文件写入 `outputs/part3_data_pipeline/`，该目录不会提交到 Git。
