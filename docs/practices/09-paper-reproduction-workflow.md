    # Part 9: Paper Reproduction Workflow

    把论文方法转成可审计实验：主张拆解、环境锁定、数据划分、baseline、消融、重复实验与结论边界。

    对应 Notebook：[运行本章](../../notebooks/practices/09-paper-reproduction-workflow.ipynb)

    ---

    ## 1. 本章目标

    - 说清核心问题、输入输出 shape 与评估协议
    - 运行一个无需外部下载的最小实验
    - 把方法迁移到遥感场景并识别数据泄漏风险
    - 完成本章练习，为下一部分建立可复用实验习惯

    ## 2. 方法流程

    ```text
    claim → protocol → baseline → implementation → sanity checks → repeated runs → ablation → report
    ```

    ## 3. 核心概念

    | 概念 | 遥感直觉 |
    | --- | --- |
    | Claim matrix | 把论文的每个核心主张映射到数据、指标、对照和证据。 |
| Reproducibility | 固定代码版本、依赖、随机种子、硬件、数据 checksum 与划分。 |
| Ablation | 一次只改变一个因素，并保留强 baseline 与等预算比较。 |
| Uncertainty | 至少多随机种子报告均值±标准差，避免把偶然结果写成结论。 |

    ## 4. 真实训练实验

    本章使用真实 PyTorch/CUDA 训练。长训练由脚本执行，避免 Notebook 重启导致结果丢失：

    ```powershell
    conda run --no-capture-output -n remote-sensing-dl python powershell -ExecutionPolicy Bypass -File scripts/real_training/run_part9.ps1
    ```

    本地结果写入 `outputs/part9_reproduction`：`best.pt`、`last.pt`、`config.json`、`history.csv`、`metrics.json`、训练曲线和预测图。`data/` 与 `outputs/` 均由 Git 忽略，绝不提交权重或结果图。

    ## 5. 实验与评估建议

    1. 先让最小样本过拟合或通过确定性 sanity check。
    2. 固定 train/val/test，并优先按场景或空间区域划分。
    3. 同时报主要指标、类别指标、参数量、计算量与运行时间。
    4. 使用多个随机种子，并保存配置、日志、checkpoint 和代码版本。
    5. 可视化输入、标签与预测，特别检查边界、小目标和失败案例。

    ## 6. 常见坑

    只跑一个 seed；复现时悄悄改变数据划分；弱 baseline；未记录失败实验；把未复现等同于论文错误。

    ## 7. 遥感迁移

    真实遥感任务还需核对空间分辨率、波段顺序、NoData、CRS、成像季节、传感器差异和跨区域泛化。公开 benchmark 的随机划分不能自动代表真实部署性能。

    ## 8. 作业与扩展

    选一篇目标论文，写三行 claim matrix：主张、必要对照、指标、失败时如何解释。

    ## 9. 本章小结

    ```text
    claim → protocol → baseline → implementation → sanity checks → repeated runs → ablation → report
    ```

    学会模型名称不是终点；可靠的数据划分、指标实现、对照实验和失败分析共同决定结论是否可信。
