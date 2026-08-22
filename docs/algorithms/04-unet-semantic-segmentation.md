    # Part 4: U-Net Semantic Segmentation

    把 Part 3 的影像 tile 与整数 mask 映射为逐像素类别，理解 encoder、decoder、skip connection、Dice 与 IoU。

    对应 Notebook：[运行本章](../../notebooks/algorithms/04-unet-semantic-segmentation.ipynb)

    ---

    ## 1. 本章目标

    - 说清核心问题、输入输出 shape 与评估协议
    - 运行一个无需外部下载的最小实验
    - 把方法迁移到遥感场景并识别数据泄漏风险
    - 完成本章练习，为下一部分建立可复用实验习惯

    ## 2. 方法流程

    ```text
    image [B,C,H,W] → encoder → bottleneck → decoder + skips → logits [B,K,H,W]
    ```

    ## 3. 核心概念

    | 概念 | 遥感直觉 |
    | --- | --- |
    | Encoder–decoder | 编码器降低空间分辨率并增加通道；解码器恢复分辨率。 |
| Skip connection | 拼接同尺度浅层细节，帮助恢复边界与小目标。 |
| Logits 与 mask | 模型输出每类 logits；target 是 [B,H,W] 的 int64 类别索引。 |
| Dice / IoU | 比 accuracy 更适合前景稀少的建筑、水体和道路分割。 |

    ## 4. 真实训练实验

    本章使用真实 PyTorch/CUDA 训练。长训练由脚本执行，避免 Notebook 重启导致结果丢失：

    ```powershell
    conda run --no-capture-output -n remote-sensing-dl python scripts/real_training/train_segmentation.py --model unet --data-root data/landcoverai --output outputs/part4_unet
    ```

    本地结果写入 `outputs/part4_unet`：`best.pt`、`last.pt`、`config.json`、`history.csv`、`metrics.json`、训练曲线和预测图。`data/` 与 `outputs/` 均由 Git 忽略，绝不提交权重或结果图。

    ## 5. 实验与评估建议

    1. 先让最小样本过拟合或通过确定性 sanity check。
    2. 固定 train/val/test，并优先按场景或空间区域划分。
    3. 同时报主要指标、类别指标、参数量、计算量与运行时间。
    4. 使用多个随机种子，并保存配置、日志、checkpoint 和代码版本。
    5. 可视化输入、标签与预测，特别检查边界、小目标和失败案例。

    ## 6. 常见坑

    skip feature 尺寸不一致；mask 被双线性插值；把 logits 先 argmax 再送入 loss；随机 tile 划分造成空间泄漏。

    ## 7. 遥感迁移

    真实遥感任务还需核对空间分辨率、波段顺序、NoData、CRS、成像季节、传感器差异和跨区域泛化。公开 benchmark 的随机划分不能自动代表真实部署性能。

    ## 8. 作业与扩展

    把阈值从 0.3 扫到 0.7，比较 Dice 最优阈值，并解释为什么测试集不能用于选阈值。

    ## 9. 本章小结

    ```text
    image [B,C,H,W] → encoder → bottleneck → decoder + skips → logits [B,K,H,W]
    ```

    学会模型名称不是终点；可靠的数据划分、指标实现、对照实验和失败分析共同决定结论是否可信。
