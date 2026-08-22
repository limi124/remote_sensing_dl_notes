"""Generate the tutorial notebooks and Markdown chapters for Parts 4-9."""

from __future__ import annotations

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

# Long runs are deliberately script-first: notebooks stay restartable and only
# invoke a one-epoch smoke run when RUN_SHORT_TRAIN is set to True.
REAL_EXPERIMENTS = {
    4: ("scripts/real_training/train_segmentation.py --model unet --data-root data/landcoverai --output outputs/part4_unet", "outputs/part4_unet"),
    5: ("scripts/real_training/train_segmentation.py --model deeplabv3plus --data-root data/landcoverai --output outputs/part5_deeplabv3plus", "outputs/part5_deeplabv3plus"),
    6: ("scripts/real_training/train_segmentation.py --model segformer_b0 --data-root data/landcoverai --output outputs/part6_segformer_b0", "outputs/part6_segformer_b0"),
    7: ("scripts/real_training/train_change.py --data-root data/levir_cd --output outputs/part7_change_weighted", "outputs/part7_change_weighted"),
    8: ("scripts/real_training/train_foundation.py --data-root data/eurosat/EuroSAT_RGB --output outputs/part8_scalemae", "outputs/part8_scalemae"),
    9: ("powershell -ExecutionPolicy Bypass -File scripts/real_training/run_part9.ps1", "outputs/part9_reproduction"),
}


SPECS = [
    {
        "part": 4, "title": "U-Net Semantic Segmentation",
        "doc": "docs/algorithms/04-unet-semantic-segmentation.md",
        "nb": "notebooks/algorithms/04-unet-semantic-segmentation.ipynb",
        "problem": "把 Part 3 的影像 tile 与整数 mask 映射为逐像素类别，理解 encoder、decoder、skip connection、Dice 与 IoU。",
        "pipeline": "image [B,C,H,W] → encoder → bottleneck → decoder + skips → logits [B,K,H,W]",
        "concepts": [
            ("Encoder–decoder", "编码器降低空间分辨率并增加通道；解码器恢复分辨率。"),
            ("Skip connection", "拼接同尺度浅层细节，帮助恢复边界与小目标。"),
            ("Logits 与 mask", "模型输出每类 logits；target 是 [B,H,W] 的 int64 类别索引。"),
            ("Dice / IoU", "比 accuracy 更适合前景稀少的建筑、水体和道路分割。"),
        ],
        "code": '''import numpy as np
rng = np.random.default_rng(4)
mask = np.zeros((64, 64), dtype=np.uint8)
mask[12:44, 18:50] = 1
score = mask * 0.75 + rng.normal(0, 0.18, mask.shape)
pred = (score > 0.5).astype(np.uint8)

def segmentation_metrics(pred, target):
    intersection = np.logical_and(pred == 1, target == 1).sum()
    union = np.logical_or(pred == 1, target == 1).sum()
    dice = 2 * intersection / ((pred == 1).sum() + (target == 1).sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return {"dice": dice, "iou": iou}

metrics = segmentation_metrics(pred, mask)
print({k: round(v, 4) for k, v in metrics.items()})
assert metrics["dice"] > metrics["iou"] > 0.8''',
        "exercise": "把阈值从 0.3 扫到 0.7，比较 Dice 最优阈值，并解释为什么测试集不能用于选阈值。",
        "pitfalls": "skip feature 尺寸不一致；mask 被双线性插值；把 logits 先 argmax 再送入 loss；随机 tile 划分造成空间泄漏。",
    },
    {
        "part": 5, "title": "DeepLabV3+ Land Cover Segmentation",
        "doc": "docs/algorithms/05-deeplabv3plus-land-cover.md",
        "nb": "notebooks/algorithms/05-deeplabv3plus-land-cover.ipynb",
        "problem": "理解空洞卷积、ASPP、多尺度上下文与轻量 decoder，面向多类别土地覆盖分割。",
        "pipeline": "backbone → low/high-level features → ASPP(rates) → upsample → decoder → land-cover logits",
        "concepts": [
            ("Atrous convolution", "在不继续下采样的前提下扩大感受野。"),
            ("ASPP", "并行使用多个 dilation rate 和全局池化捕捉多尺度上下文。"),
            ("Output stride", "控制特征图分辨率与计算量，常见 8 或 16。"),
            ("Decoder", "融合浅层边界特征，改善道路、岸线和地块边缘。"),
        ],
        "code": '''import numpy as np

def receptive_positions(center, kernel_size=3, dilation=1):
    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1) * dilation
    return center + offsets

for rate in [1, 2, 4, 8]:
    positions = receptive_positions(20, dilation=rate)
    effective = 3 + (3 - 1) * (rate - 1)
    print(f"rate={rate}: samples={positions.tolist()}, effective_kernel={effective}")

assert (receptive_positions(20, dilation=4) == [16, 20, 24]).all()''',
        "exercise": "为 10 m 分辨率、目标尺度约 30–300 m 的影像提出一组 ASPP rates，并说明 output stride 的影响。",
        "pitfalls": "dilation 过大出现 gridding；忽略类别不平衡；上采样 align_corners 设置不一致；训练/验证地块重叠。",
    },
    {
        "part": 6, "title": "Transformer Vision Models",
        "doc": "docs/algorithms/06-transformer-vision-models.md",
        "nb": "notebooks/algorithms/06-transformer-vision-models.ipynb",
        "problem": "从 patch token 与 self-attention 理解 ViT、Swin 和 SegFormer，并关注遥感大图计算复杂度。",
        "pipeline": "image → patch embedding → tokens + position → attention/MLP blocks → multi-scale decoder",
        "concepts": [
            ("Patch embedding", "把图像切块并投影为 token 序列。"),
            ("Self-attention", "用 QKᵀ/√d 计算 token 间关系，再加权 V。"),
            ("Hierarchical features", "Swin/SegFormer 逐级降采样，更适合密集预测。"),
            ("Complexity", "全局 attention 随 token 数平方增长，大图需窗口、稀疏或多尺度策略。"),
        ],
        "code": '''import numpy as np
rng = np.random.default_rng(6)
tokens = rng.normal(size=(16, 8))
Q, K, V = tokens, tokens, tokens
scores = Q @ K.T / np.sqrt(Q.shape[-1])
scores -= scores.max(axis=-1, keepdims=True)
weights = np.exp(scores); weights /= weights.sum(axis=-1, keepdims=True)
context = weights @ V
print("tokens:", tokens.shape, "attention:", weights.shape, "context:", context.shape)
print("row sums:", np.round(weights.sum(axis=1)[:4], 6))
assert np.allclose(weights.sum(axis=1), 1.0)''',
        "exercise": "分别计算 512×512 影像在 patch size 16 与 8 下的 token 数及 attention 矩阵元素数量。",
        "pitfalls": "位置编码尺寸不匹配；忽略多光谱通道适配；token 太多导致显存平方增长；只看自然图像预训练而不做域验证。",
    },
    {
        "part": 7, "title": "Optical Change Detection",
        "doc": "docs/practices/07-optical-change-detection.md",
        "nb": "notebooks/practices/07-optical-change-detection.ipynb",
        "problem": "建立双时相光学变化检测最小闭环：配准、Siamese 特征、差异融合、阈值与 F1/IoU。",
        "pipeline": "image_t1 + image_t2 → shared encoder → feature difference/fusion → change logits → binary mask",
        "concepts": [
            ("Co-registration", "亚像素错位也会制造伪变化，进入模型前先对齐。"),
            ("Siamese encoder", "两个时相共享权重，便于比较一致的特征空间。"),
            ("Change fusion", "可用绝对差、拼接、注意力或双向交互融合。"),
            ("Metrics", "变化像素稀少，应报告 precision、recall、F1、IoU，而非只报 accuracy。"),
        ],
        "code": '''import numpy as np
rng = np.random.default_rng(7)
t1 = rng.normal(0.3, 0.03, (64, 64))
t2 = t1.copy(); t2[20:42, 28:50] += 0.45
t2 += rng.normal(0, 0.02, t2.shape)
truth = np.zeros_like(t1, dtype=bool); truth[20:42, 28:50] = True
change_score = np.abs(t2 - t1)
pred = change_score > 0.2
tp = np.logical_and(pred, truth).sum(); fp = np.logical_and(pred, ~truth).sum(); fn = np.logical_and(~pred, truth).sum()
precision = tp / (tp + fp + 1e-8); recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
print({"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)})
assert f1 > 0.95''',
        "exercise": "人为将 t2 平移 1 个像素，比较伪变化数量；再讨论真实项目的配准质量检查。",
        "pitfalls": "时相未配准；季节/光照变化被当成地物变化；正负样本极不平衡；同一区域跨集合泄漏。",
    },
    {
        "part": 8, "title": "Remote Sensing Foundation Models",
        "doc": "docs/papers/08-remote-sensing-foundation-models.md",
        "nb": "notebooks/papers/08-remote-sensing-foundation-models.ipynb",
        "problem": "理解 MAE、DINO、对比学习和视觉语言预训练如何形成通用遥感表征，以及如何做线性探测与微调。",
        "pipeline": "large unlabeled corpus → self-supervised pretraining → encoder → linear probe / fine-tune / prompt adaptation",
        "concepts": [
            ("MAE", "遮挡大部分 patch，从可见 token 重建缺失内容，学习空间与光谱结构。"),
            ("DINO", "教师–学生自蒸馏，不依赖人工标签形成语义表征。"),
            ("Contrastive/VLM", "拉近匹配影像–文本或同一地点多视图，拉远不匹配样本。"),
            ("Transfer protocol", "先 linear probe，再全量/参数高效微调，并与从头训练公平比较。"),
        ],
        "code": '''import numpy as np
rng = np.random.default_rng(8)
image_embeddings = rng.normal(size=(5, 8))
text_embeddings = image_embeddings + rng.normal(scale=0.08, size=(5, 8))
image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)
text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
similarity = image_embeddings @ text_embeddings.T
retrieved = similarity.argmax(axis=1)
print("retrieved text indices:", retrieved.tolist())
print("top-1:", (retrieved == np.arange(5)).mean())
assert (retrieved == np.arange(5)).mean() >= 0.8''',
        "exercise": "设计一个 linear probe 与 full fine-tuning 的公平对比表，固定数据划分、增强、epoch 与评估指标。",
        "pitfalls": "预训练数据与下游测试泄漏；波段/分辨率不匹配；只比较单一微调策略；忽略预训练算力和数据许可。",
    },
    {
        "part": 9, "title": "Paper Reproduction Workflow",
        "doc": "docs/practices/09-paper-reproduction-workflow.md",
        "nb": "notebooks/practices/09-paper-reproduction-workflow.ipynb",
        "problem": "把论文方法转成可审计实验：主张拆解、环境锁定、数据划分、baseline、消融、重复实验与结论边界。",
        "pipeline": "claim → protocol → baseline → implementation → sanity checks → repeated runs → ablation → report",
        "concepts": [
            ("Claim matrix", "把论文的每个核心主张映射到数据、指标、对照和证据。"),
            ("Reproducibility", "固定代码版本、依赖、随机种子、硬件、数据 checksum 与划分。"),
            ("Ablation", "一次只改变一个因素，并保留强 baseline 与等预算比较。"),
            ("Uncertainty", "至少多随机种子报告均值±标准差，避免把偶然结果写成结论。"),
        ],
        "code": '''import numpy as np
baseline = np.array([0.714, 0.721, 0.718, 0.709, 0.724])
method = np.array([0.731, 0.728, 0.736, 0.729, 0.734])
delta = method - baseline
summary = {
    "baseline_mean": baseline.mean(), "method_mean": method.mean(),
    "mean_delta": delta.mean(), "delta_std": delta.std(ddof=1),
}
print({k: round(v, 4) for k, v in summary.items()})
assert summary["mean_delta"] > 0

claim_matrix = [
    {"claim": "module improves IoU", "evidence": "5 seeds + paired comparison"},
    {"claim": "gain is not extra compute", "evidence": "matched FLOPs/params baseline"},
]
for row in claim_matrix: print(row)''',
        "exercise": "选一篇目标论文，写三行 claim matrix：主张、必要对照、指标、失败时如何解释。",
        "pitfalls": "只跑一个 seed；复现时悄悄改变数据划分；弱 baseline；未记录失败实验；把未复现等同于论文错误。",
    },
]


def make_doc(spec: dict) -> str:
    concepts = "\n".join(f"| {name} | {desc} |" for name, desc in spec["concepts"])
    command, output_dir = REAL_EXPERIMENTS[spec["part"]]
    return textwrap.dedent(f"""\
    # Part {spec['part']}: {spec['title']}

    {spec['problem']}

    对应 Notebook：[运行本章](../../{spec['nb']})

    ---

    ## 1. 本章目标

    - 说清核心问题、输入输出 shape 与评估协议
    - 运行一个无需外部下载的最小实验
    - 把方法迁移到遥感场景并识别数据泄漏风险
    - 完成本章练习，为下一部分建立可复用实验习惯

    ## 2. 方法流程

    ```text
    {spec['pipeline']}
    ```

    ## 3. 核心概念

    | 概念 | 遥感直觉 |
    | --- | --- |
    {concepts}

    ## 4. 真实训练实验

    本章使用真实 PyTorch/CUDA 训练。长训练由脚本执行，避免 Notebook 重启导致结果丢失：

    ```powershell
    conda run --no-capture-output -n remote-sensing-dl python {command}
    ```

    本地结果写入 `{output_dir}`：`best.pt`、`last.pt`、`config.json`、`history.csv`、`metrics.json`、训练曲线和预测图。`data/` 与 `outputs/` 均由 Git 忽略，绝不提交权重或结果图。

    ## 5. 实验与评估建议

    1. 先让最小样本过拟合或通过确定性 sanity check。
    2. 固定 train/val/test，并优先按场景或空间区域划分。
    3. 同时报主要指标、类别指标、参数量、计算量与运行时间。
    4. 使用多个随机种子，并保存配置、日志、checkpoint 和代码版本。
    5. 可视化输入、标签与预测，特别检查边界、小目标和失败案例。

    ## 6. 常见坑

    {spec['pitfalls']}

    ## 7. 遥感迁移

    真实遥感任务还需核对空间分辨率、波段顺序、NoData、CRS、成像季节、传感器差异和跨区域泛化。公开 benchmark 的随机划分不能自动代表真实部署性能。

    ## 8. 作业与扩展

    {spec['exercise']}

    ## 9. 本章小结

    ```text
    {spec['pipeline']}
    ```

    学会模型名称不是终点；可靠的数据划分、指标实现、对照实验和失败分析共同决定结论是否可信。
    """)


def make_notebook(spec: dict):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.x"}
    concepts = "\n".join(f"- **{name}**：{desc}" for name, desc in spec["concepts"])
    command, output_dir = REAL_EXPERIMENTS[spec["part"]]
    short_command = command + " --epochs 1 --train-limit 32 --val-limit 16" if spec["part"] in (4, 5, 6) else command
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# Part {spec['part']}: {spec['title']}\n\n{spec['problem']}\n\n**先修要求**：完成前序章节，了解 NumPy、张量 shape 与训练/验证划分。"),
        nbf.v4.new_markdown_cell(f"## 学习目标与路线\n\n1. 明确问题与输入输出\n2. 理解核心结构\n3. 运行最小实验\n4. 检查指标和 shape\n5. 完成练习与遥感迁移\n\n`{spec['pipeline']}`"),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nROOT = Path.cwd().resolve()\nwhile not (ROOT / 'scripts').exists() and ROOT.parent != ROOT: ROOT = ROOT.parent\nprint('repository:', ROOT)"),
        nbf.v4.new_markdown_cell(f"## 核心概念\n\n{concepts}"),
        nbf.v4.new_markdown_cell("## 本地真实训练结果\n\n长训练由脚本执行；这里读取本机已经生成的指标和产物。GitHub 中不会包含 `outputs/`。"),
        nbf.v4.new_code_cell(f"output = ROOT / r'{output_dir}'\nmetrics_path = output / 'metrics.json'\nprint('local output:', output)\nprint(json.dumps(json.loads(metrics_path.read_text(encoding='utf-8')), indent=2) if metrics_path.exists() else '尚未运行：请执行下方命令。')"),
        nbf.v4.new_markdown_cell("## 可恢复的脚本训练\n\n默认不在 Notebook 内启动长训练。将 `RUN_SHORT_TRAIN` 改为 `True` 可执行一次小规模 smoke run；正式命令请在终端执行。"),
        nbf.v4.new_code_cell(f"RUN_SHORT_TRAIN = False\ncommand = r'''conda run --no-capture-output -n remote-sensing-dl python {short_command}'''\nprint(command)\nif RUN_SHORT_TRAIN:\n    import subprocess\n    subprocess.run(command, cwd=ROOT, shell=True, check=True)"),
        nbf.v4.new_markdown_cell("## 结果解释\n\n正式结果必须包含非空 checkpoint、完整 epoch 日志、验证指标与预测图；出现缺失时先检查数据配对、CUDA 与配置，而不要直接扩大模型。"),
        nbf.v4.new_markdown_cell(f"## 练习\n\n{spec['exercise']}"),
        nbf.v4.new_code_cell("# 答案脚手架\nexperiment = {'hypothesis': '', 'controlled_variables': [], 'metric': '', 'expected_failure': ''}\nprint(experiment)"),
        nbf.v4.new_markdown_cell(f"## 常见坑与扩展\n\n**常见坑**：{spec['pitfalls']}\n\n**扩展**：替换为真实遥感数据时，记录传感器、波段、分辨率、空间划分和随机种子。"),
        nbf.v4.new_markdown_cell("## 完成检查\n\n- [ ] 所有单元格从头运行\n- [ ] 指标实现通过 sanity check\n- [ ] 没有 train/val 空间泄漏\n- [ ] 配置和随机种子已记录\n- [ ] 至少分析一个失败案例"),
    ]
    return nb


def main():
    for spec in SPECS:
        doc_path, nb_path = ROOT / spec["doc"], ROOT / spec["nb"]
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(make_doc(spec), encoding="utf-8")
        nbf.write(make_notebook(spec), nb_path)
        print("generated", spec["doc"], "and", spec["nb"])


if __name__ == "__main__":
    main()
