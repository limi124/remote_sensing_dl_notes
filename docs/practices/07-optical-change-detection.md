    # Part 7: Optical Change Detection

    建立双时相光学变化检测最小闭环：配准、Siamese 特征、差异融合、阈值与 F1/IoU。

    对应 Notebook：[运行本章](../../notebooks/practices/07-optical-change-detection.ipynb)

    ---

    ## 1. 本章目标

    - 说清核心问题、输入输出 shape 与评估协议
    - 运行一个无需外部下载的最小实验
    - 把方法迁移到遥感场景并识别数据泄漏风险
    - 完成本章练习，为下一部分建立可复用实验习惯

    ## 2. 方法流程

    ```text
    image_t1 + image_t2 → shared encoder → feature difference/fusion → change logits → binary mask
    ```

    ## 3. 核心概念

    | 概念 | 遥感直觉 |
    | --- | --- |
    | Co-registration | 亚像素错位也会制造伪变化，进入模型前先对齐。 |
| Siamese encoder | 两个时相共享权重，便于比较一致的特征空间。 |
| Change fusion | 可用绝对差、拼接、注意力或双向交互融合。 |
| Metrics | 变化像素稀少，应报告 precision、recall、F1、IoU，而非只报 accuracy。 |

    ## 4. 真实训练实验

    本章使用真实 PyTorch/CUDA 训练。长训练由脚本执行，避免 Notebook 重启导致结果丢失：

    ```powershell
    conda run --no-capture-output -n remote-sensing-dl python scripts/real_training/train_change.py --data-root data/levir_cd --output outputs/part7_change_weighted
    ```

    本地结果写入 `outputs/part7_change_weighted`：`best.pt`、`last.pt`、`config.json`、`history.csv`、`metrics.json`、训练曲线和预测图。`data/` 与 `outputs/` 均由 Git 忽略，绝不提交权重或结果图。

    ## 5. 实验与评估建议

    1. 先让最小样本过拟合或通过确定性 sanity check。
    2. 固定 train/val/test，并优先按场景或空间区域划分。
    3. 同时报主要指标、类别指标、参数量、计算量与运行时间。
    4. 使用多个随机种子，并保存配置、日志、checkpoint 和代码版本。
    5. 可视化输入、标签与预测，特别检查边界、小目标和失败案例。

    ## 6. 常见坑

    时相未配准；季节/光照变化被当成地物变化；正负样本极不平衡；同一区域跨集合泄漏。

    ## 7. 遥感迁移

    真实遥感任务还需核对空间分辨率、波段顺序、NoData、CRS、成像季节、传感器差异和跨区域泛化。公开 benchmark 的随机划分不能自动代表真实部署性能。

    ## 8. 作业与扩展

    人为将 t2 平移 1 个像素，比较伪变化数量；再讨论真实项目的配准质量检查。

    ## 9. 本章小结

    ```text
    image_t1 + image_t2 → shared encoder → feature difference/fusion → change logits → binary mask
    ```

    学会模型名称不是终点；可靠的数据划分、指标实现、对照实验和失败分析共同决定结论是否可信。
