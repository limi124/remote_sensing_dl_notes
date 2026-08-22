    # Part 5: DeepLabV3+ Land Cover Segmentation

    理解空洞卷积、ASPP、多尺度上下文与轻量 decoder，面向多类别土地覆盖分割。

    对应 Notebook：[运行本章](../../notebooks/algorithms/05-deeplabv3plus-land-cover.ipynb)

    ---

    ## 1. 本章目标

    - 说清核心问题、输入输出 shape 与评估协议
    - 运行一个无需外部下载的最小实验
    - 把方法迁移到遥感场景并识别数据泄漏风险
    - 完成本章练习，为下一部分建立可复用实验习惯

    ## 2. 方法流程

    ```text
    backbone → low/high-level features → ASPP(rates) → upsample → decoder → land-cover logits
    ```

    ## 3. 核心概念

    | 概念 | 遥感直觉 |
    | --- | --- |
    | Atrous convolution | 在不继续下采样的前提下扩大感受野。 |
| ASPP | 并行使用多个 dilation rate 和全局池化捕捉多尺度上下文。 |
| Output stride | 控制特征图分辨率与计算量，常见 8 或 16。 |
| Decoder | 融合浅层边界特征，改善道路、岸线和地块边缘。 |

    ## 4. 真实训练实验

    本章使用真实 PyTorch/CUDA 训练。长训练由脚本执行，避免 Notebook 重启导致结果丢失：

    ```powershell
    conda run --no-capture-output -n remote-sensing-dl python scripts/real_training/train_segmentation.py --model deeplabv3plus --data-root data/landcoverai --output outputs/part5_deeplabv3plus
    ```

    本地结果写入 `outputs/part5_deeplabv3plus`：`best.pt`、`last.pt`、`config.json`、`history.csv`、`metrics.json`、训练曲线和预测图。`data/` 与 `outputs/` 均由 Git 忽略，绝不提交权重或结果图。

    ## 5. 实验与评估建议

    1. 先让最小样本过拟合或通过确定性 sanity check。
    2. 固定 train/val/test，并优先按场景或空间区域划分。
    3. 同时报主要指标、类别指标、参数量、计算量与运行时间。
    4. 使用多个随机种子，并保存配置、日志、checkpoint 和代码版本。
    5. 可视化输入、标签与预测，特别检查边界、小目标和失败案例。

    ## 6. 常见坑

    dilation 过大出现 gridding；忽略类别不平衡；上采样 align_corners 设置不一致；训练/验证地块重叠。

    ## 7. 遥感迁移

    真实遥感任务还需核对空间分辨率、波段顺序、NoData、CRS、成像季节、传感器差异和跨区域泛化。公开 benchmark 的随机划分不能自动代表真实部署性能。

    ## 8. 作业与扩展

    为 10 m 分辨率、目标尺度约 30–300 m 的影像提出一组 ASPP rates，并说明 output stride 的影响。

    ## 9. 本章小结

    ```text
    backbone → low/high-level features → ASPP(rates) → upsample → decoder → land-cover logits
    ```

    学会模型名称不是终点；可靠的数据划分、指标实现、对照实验和失败分析共同决定结论是否可信。
