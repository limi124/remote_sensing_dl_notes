# Part 2: CNN Image Classification

从 `Conv2d` 到 EuroSAT 遥感场景分类。

这一章把 Part 1 的训练闭环迁移到真实图像任务上。上一章输入是二维点：

```text
X: [B, 2]
y: [B]
```

这一章输入变成图像：

```text
image: [B, C, H, W]
label: [B]
```

目标不是一上来追求很高精度，而是把 CNN 的 shape、训练流程、数据增强、验证指标和遥感场景分类问题串起来。

---

## 1. 本章目标

学完后你应该能做到：

- 理解 `nn.Conv2d` 的输入输出 shape
- 说清楚 channel、kernel size、stride、padding 的作用
- 写一个最小 CNN 图像分类器
- 复用 Part 1 的 `train_one_epoch`、`evaluate`、`fit`
- 使用 `torchvision.datasets.EuroSAT` 读取遥感场景分类数据
- 区分 RGB 影像分类和多光谱影像分类
- 画训练曲线和混淆矩阵
- 做 overfit 小数据实验来检查训练流程

---

## 2. CNN 解决什么问题

MLP 可以处理表格特征或已经拉平成一维的向量，但图像有空间结构：

```text
上方像素和下方像素有关系
道路、河流、建筑物会形成局部纹理和形状
同一个目标可能出现在图像不同位置
```

CNN 的核心假设：

| 机制 | 作用 |
| --- | --- |
| 局部连接 | 先看小窗口，而不是一次看完整图 |
| 权重共享 | 同一个卷积核在整张图上滑动 |
| 多通道特征 | 每层提取多种纹理、边缘和语义线索 |
| 下采样 | 逐步扩大感受野，压缩空间尺寸 |

遥感场景分类中，模型要根据整张 patch 判断土地利用/覆盖类型，例如 `Forest`、`River`、`Residential`、`Industrial`。

---

## 3. 图像 Tensor Shape

PyTorch 图像模型默认使用：

```text
[B, C, H, W]
```

含义：

| 维度 | 含义 |
| --- | --- |
| `B` | batch size |
| `C` | channel 数，RGB 是 3，多光谱可能是 13 |
| `H` | height |
| `W` | width |

例如 EuroSAT RGB 图像一般进入模型后是：

```text
[B, 3, 64, 64]
```

如果使用 Sentinel-2 全 13 波段版本，输入可能是：

```text
[B, 13, 64, 64]
```

---

## 4. Conv2d 的 Shape 公式

`nn.Conv2d` 常见写法：

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```

输入：

```text
[B, 3, H, W]
```

输出：

```text
[B, 16, H_out, W_out]
```

空间尺寸计算：

```text
H_out = floor((H + 2 * padding - kernel_size) / stride) + 1
W_out = floor((W + 2 * padding - kernel_size) / stride) + 1
```

如果 `kernel_size=3, stride=1, padding=1`，空间尺寸不变：

```text
[B, 3, 64, 64] -> [B, 16, 64, 64]
```

如果接一个 `MaxPool2d(2)`：

```text
[B, 16, 64, 64] -> [B, 16, 32, 32]
```

---

## 5. 最小 CNN 分类器

```python
import torch
import torch.nn as nn


class TinyCNNClassifier(nn.Module):
    """一个最小 CNN 图像分类器。

    输入:
        x: [B, 3, H, W]
    输出:
        logits: [B, num_classes]
    """

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

为什么用 `AdaptiveAvgPool2d((1, 1))`：

```text
无论输入图像是 64x64 还是 128x128，最后都压成 [B, C, 1, 1]
```

这样分类头不需要手动计算 flatten 后的维度。

---

## 6. 打印中间 Shape

训练 CNN 前，先用假数据检查 shape：

```python
model = TinyCNNClassifier(in_channels=3, num_classes=10)
x = torch.randn(4, 3, 64, 64)

with torch.no_grad():
    y = x
    for layer in model.features:
        y = layer(y)
        print(layer.__class__.__name__, y.shape)

    y = model.pool(y)
    print("AdaptiveAvgPool2d", y.shape)
    y = torch.flatten(y, 1)
    print("flatten", y.shape)
    logits = model.classifier(y)
    print("logits", logits.shape)
```

你应该看到：

```text
logits: [4, 10]
```

分类任务中，`logits` 的第二维等于类别数。

---

## 7. EuroSAT 数据集

EuroSAT 是一个遥感场景分类数据集，基于 Sentinel-2 影像，包含 10 个土地利用/土地覆盖类别和 27000 张标注图像。原始数据有 13 个光谱波段，也常用 RGB 版本做入门实验。

常见类别：

```text
AnnualCrop
Forest
HerbaceousVegetation
Highway
Industrial
Pasture
PermanentCrop
Residential
River
SeaLake
```

资料入口：

- 官方 GitHub: https://github.com/phelber/EuroSAT
- 论文: https://arxiv.org/abs/1709.00029
- Zenodo: https://zenodo.org/records/7711810
- TorchGeo 文档: https://docs.torchgeo.org/en/latest/api/datasets/eurosat.html

本章使用 RGB 版本，因为它最接近普通图像分类流程，也更容易在普通电脑上跑通。

---

## 8. 读取 EuroSAT RGB

如果你的 `torchvision` 版本支持 `EuroSAT`，可以直接使用：

```python
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3444, 0.3809, 0.4082],
        std=[0.2037, 0.1366, 0.1148],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3444, 0.3809, 0.4082],
        std=[0.2037, 0.1366, 0.1148],
    ),
])

dataset = datasets.EuroSAT(
    root="data/eurosat",
    transform=train_transform,
    download=True,
)
```

如果下载失败，可以先手动下载并解压，或改用 Kaggle / Zenodo 镜像。教学时可以先用 `FakeData` 跑通代码，再切换到真实 EuroSAT。

---

## 9. 训练闭环

这一章的训练函数和 Part 1 基本一样，只是输入从 `[B, 2]` 变成 `[B, 3, H, W]`：

```python
import torch.nn.functional as F


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return {"loss": total_loss / total_count, "acc": total_correct / total_count}
```

分类任务的 loss 仍然是 `F.cross_entropy(logits, labels)`。

---

## 10. 小实验：过拟合一小批图像

在完整训练前，先取 32 或 64 张图像让模型 overfit：

```python
from torch.utils.data import DataLoader, Subset

small_dataset = Subset(dataset, list(range(32)))
small_loader = DataLoader(small_dataset, batch_size=16, shuffle=True)
```

如果模型不能在小数据上把训练 accuracy 拉到很高，优先检查：

```text
image shape
label dtype
loss
learning rate
模型输出类别数
数据增强是否过强
```

---

## 11. 混淆矩阵

整体 accuracy 不够解释模型错在哪里。遥感场景分类常见混淆包括：

```text
Industrial vs Residential
AnnualCrop vs PermanentCrop
River vs Highway
HerbaceousVegetation vs Pasture
```

可以统计混淆矩阵：

```python
def compute_confusion_matrix(model, dataloader, device, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            for target, pred in zip(labels.cpu(), preds.cpu()):
                matrix[target, pred] += 1
    return matrix
```

行是真实类别，列是预测类别。

---

## 12. 遥感迁移

| 普通图像分类 | 遥感场景分类 |
| --- | --- |
| 物体通常居中 | 场景由纹理和空间布局共同决定 |
| RGB 为主 | RGB、多光谱、SAR、多时相都可能出现 |
| 类别语义直观 | 类别边界可能依赖地理知识 |
| 数据增强多为翻转裁剪 | 要注意方向、尺度、季节、区域差异 |

EuroSAT 只是入门。真实遥感研究还会遇到：

- 多光谱波段如何输入模型
- 训练区和测试区空间相关导致虚高结果
- 跨区域泛化下降
- 类别定义和标注尺度不一致
- 云、阴影、季节变化造成域偏移

---

## 13. 常见坑

| 问题 | 可能原因 | 排查方法 |
| --- | --- | --- |
| `Expected 4D input` | 输入不是 `[B, C, H, W]` | 打印 batch shape |
| loss 不下降 | lr 不合适、标签错、增强太强 | overfit 小数据 |
| 类别数报错 | `num_classes` 和数据集类别不一致 | 打印 `dataset.classes` |
| 显存不足 | batch 太大、模型太大 | 减小 batch 或输入尺寸 |
| 验证精度异常高 | 训练/验证泄漏 | 固定 split，检查重复样本 |
| RGB 颜色奇怪 | normalize 后直接显示 | 反归一化再可视化 |

---

## 14. 作业

1. 打印每一层 CNN 的 shape，并解释为什么变化。
2. 把 `out_channels` 从 `16, 32, 64` 改成 `8, 16, 32`，比较速度和精度。
3. 去掉 `BatchNorm2d`，观察训练是否更慢或更不稳定。
4. 对比 `Adam(lr=1e-3)` 和 `SGD(lr=1e-2, momentum=0.9)`。
5. 训练 5 个 epoch，画 loss / accuracy 曲线。
6. 画混淆矩阵，写出最容易混淆的 3 对类别。
7. 思考：如果输入从 RGB 改成 Sentinel-2 13 波段，模型第一层要怎么改？

---

## 15. 本章小结

这一章你真正要带走的是：

```text
Conv2d 把图像从 [B, C, H, W] 变成多层 feature maps
CNN 分类器 = feature extractor + global pooling + linear classifier
EuroSAT 是把 CNN 训练闭环迁移到遥感图像上的第一站
```

下一章进入遥感数据管线时，重点会从模型转向数据：

```text
GeoTIFF
CRS
shapefile / label
tiling
augmentation
mask
```

