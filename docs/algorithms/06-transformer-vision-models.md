    # Part 6: Transformer Vision Models

    从 patch token 与 self-attention 理解 ViT、Swin 和 SegFormer，并关注遥感大图计算复杂度。

    对应 Notebook：[运行本章](../../notebooks/algorithms/06-transformer-vision-models.ipynb)

    ---

    ## 1. 本章目标

    - 说清核心问题、输入输出 shape 与评估协议
    - 运行一个无需外部下载的最小实验
    - 把方法迁移到遥感场景并识别数据泄漏风险
    - 完成本章练习，为下一部分建立可复用实验习惯

    ## 2. 方法流程

    ```text
    image → patch embedding → tokens + position → attention/MLP blocks → multi-scale decoder
    ```

    ## 3. 核心概念

    | 概念 | 遥感直觉 |
    | --- | --- |
    | Patch embedding | 把图像切块并投影为 token 序列。 |
| Self-attention | 用 QKᵀ/√d 计算 token 间关系，再加权 V。 |
| Hierarchical features | Swin/SegFormer 逐级降采样，更适合密集预测。 |
| Complexity | 全局 attention 随 token 数平方增长，大图需窗口、稀疏或多尺度策略。 |

    ## 4. 真实训练实验

    本章使用真实 PyTorch/CUDA 训练。长训练由脚本执行，避免 Notebook 重启导致结果丢失：

    ```powershell
    conda run --no-capture-output -n remote-sensing-dl python scripts/real_training/train_segmentation.py --model segformer_b0 --data-root data/landcoverai --output outputs/part6_segformer_b0
    ```

    本地结果写入 `outputs/part6_segformer_b0`：`best.pt`、`last.pt`、`config.json`、`history.csv`、`metrics.json`、训练曲线和预测图。`data/` 与 `outputs/` 均由 Git 忽略，绝不提交权重或结果图。

    ## 5. 实验与评估建议

    1. 先让最小样本过拟合或通过确定性 sanity check。
    2. 固定 train/val/test，并优先按场景或空间区域划分。
    3. 同时报主要指标、类别指标、参数量、计算量与运行时间。
    4. 使用多个随机种子，并保存配置、日志、checkpoint 和代码版本。
    5. 可视化输入、标签与预测，特别检查边界、小目标和失败案例。

    ## 6. 常见坑

    位置编码尺寸不匹配；忽略多光谱通道适配；token 太多导致显存平方增长；只看自然图像预训练而不做域验证。

    ## 7. 遥感迁移

    真实遥感任务还需核对空间分辨率、波段顺序、NoData、CRS、成像季节、传感器差异和跨区域泛化。公开 benchmark 的随机划分不能自动代表真实部署性能。

    ## 8. 作业与扩展

    分别计算 512×512 影像在 patch size 16 与 8 下的 token 数及 attention 矩阵元素数量。

    ## 9. 本章小结

    ```text
    image → patch embedding → tokens + position → attention/MLP blocks → multi-scale decoder
    ```

    学会模型名称不是终点；可靠的数据划分、指标实现、对照实验和失败分析共同决定结论是否可信。
