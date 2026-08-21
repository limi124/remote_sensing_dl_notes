    # Part 8: Remote Sensing Foundation Models

    理解 MAE、DINO、对比学习和视觉语言预训练如何形成通用遥感表征，以及如何做线性探测与微调。

    对应 Notebook：[运行本章](../../notebooks/papers/08-remote-sensing-foundation-models.ipynb)

    ---

    ## 1. 本章目标

    - 说清核心问题、输入输出 shape 与评估协议
    - 运行一个无需外部下载的最小实验
    - 把方法迁移到遥感场景并识别数据泄漏风险
    - 完成本章练习，为下一部分建立可复用实验习惯

    ## 2. 方法流程

    ```text
    large unlabeled corpus → self-supervised pretraining → encoder → linear probe / fine-tune / prompt adaptation
    ```

    ## 3. 核心概念

    | 概念 | 遥感直觉 |
    | --- | --- |
    | MAE | 遮挡大部分 patch，从可见 token 重建缺失内容，学习空间与光谱结构。 |
| DINO | 教师–学生自蒸馏，不依赖人工标签形成语义表征。 |
| Contrastive/VLM | 拉近匹配影像–文本或同一地点多视图，拉远不匹配样本。 |
| Transfer protocol | 先 linear probe，再全量/参数高效微调，并与从头训练公平比较。 |

    ## 4. 最小实验

    本章 Notebook 使用固定随机种子与 NumPy 合成数据，先验证数学、shape 和指标，再替换为真实数据与 PyTorch 模型。这样可以把环境问题与方法问题分开排查。

    ## 5. 实验与评估建议

    1. 先让最小样本过拟合或通过确定性 sanity check。
    2. 固定 train/val/test，并优先按场景或空间区域划分。
    3. 同时报主要指标、类别指标、参数量、计算量与运行时间。
    4. 使用多个随机种子，并保存配置、日志、checkpoint 和代码版本。
    5. 可视化输入、标签与预测，特别检查边界、小目标和失败案例。

    ## 6. 常见坑

    预训练数据与下游测试泄漏；波段/分辨率不匹配；只比较单一微调策略；忽略预训练算力和数据许可。

    ## 7. 遥感迁移

    真实遥感任务还需核对空间分辨率、波段顺序、NoData、CRS、成像季节、传感器差异和跨区域泛化。公开 benchmark 的随机划分不能自动代表真实部署性能。

    ## 8. 作业与扩展

    设计一个 linear probe 与 full fine-tuning 的公平对比表，固定数据划分、增强、epoch 与评估指标。

    ## 9. 本章小结

    ```text
    large unlabeled corpus → self-supervised pretraining → encoder → linear probe / fine-tune / prompt adaptation
    ```

    学会模型名称不是终点；可靠的数据划分、指标实现、对照实验和失败分析共同决定结论是否可信。
