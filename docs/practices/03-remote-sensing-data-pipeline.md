# Part 3: Remote Sensing Data Pipeline

这一章把模型暂时放到一边，建立遥感深度学习中最容易出错、也最值得复用的数据流水线：

```text
GeoTIFF -> 地理参考 -> 矢量标注 -> mask -> 滑窗切片 -> 同步增强 -> Dataset
```

对应可运行 Notebook：

- [03-remote-sensing-data-pipeline.ipynb](../../notebooks/practices/03-remote-sensing-data-pipeline.ipynb)
- [实操脚本](../../scripts/part3_data_pipeline/README.md)

---

## 1. 本章目标

学完后你应该能：

- 用 Rasterio 读取多波段 GeoTIFF 和元数据
- 解释 CRS、仿射变换、像素坐标和地理坐标
- 将矢量标注按参考栅格对齐并栅格化为 mask
- 把大幅影像切成固定大小的训练 patch
- 对影像与 mask 做严格同步的空间增强
- 构造语义分割用的 PyTorch Dataset
- 识别 CRS 错位、NoData、边界 padding 和数据泄漏

---

## 2. 为什么普通图像读取不够

Pillow 读取 PNG/JPEG 后通常只剩像素值，而遥感影像还携带空间语义：

| 信息 | 作用 |
| --- | --- |
| `CRS` | 说明坐标属于哪个地理/投影坐标系统 |
| `transform` | 把 `(row, col)` 映射到真实世界坐标 |
| `bounds` | 影像覆盖的地理范围 |
| `nodata` | 无有效观测的像素值 |
| `dtype` | 反射率常是 `uint16`，不能默认按 0–255 处理 |
| `count` | 波段数，可能是 RGB、RGBNIR 或 13 波段 |

因此读取遥感影像时，应把数组与 profile 一起保留。

```python
import rasterio

with rasterio.open("scene.tif") as src:
    image = src.read()       # [C, H, W]
    profile = src.profile
    print(src.crs, src.transform, src.bounds, src.nodata)
```

---

## 3. CRS 与仿射变换

CRS 定义“坐标数字代表哪里”。例如：

```text
EPSG:4326  经度/纬度，单位通常是度
EPSG:32650 WGS 84 / UTM zone 50N，单位是米
```

仿射变换负责像素网格与地图坐标的转换：

```text
x = a * col + b * row + c
y = d * col + e * row + f
```

Rasterio 中不要手写公式，直接使用：

```python
x, y = src.xy(row, col)
row, col = src.index(x, y)
```

关键规则：影像与矢量标注进入叠加、裁剪或栅格化前，必须处于同一 CRS。

---

## 4. 多波段与数值缩放

Rasterio 返回的默认 shape 是 `[C, H, W]`，正好与 PyTorch 单张图像的 channel-first 约定一致。Sentinel-2 常把表观反射率按 10000 缩放保存：

```python
image = image.astype("float32") / 10000.0
```

但 10000 不是所有数据集的通用常数。真实项目必须先查产品说明和统计分布，并妥善处理 NoData。

---

## 5. 矢量标注栅格化

语义分割需要每个像素一个整数类别：

```text
mask: [H, W]
0 = background
1 = water
2 = built_up
```

矢量 polygon 要使用参考影像的 `height`、`width` 和 `transform` 烧录：

```python
from rasterio.features import rasterize

mask = rasterize(
    [(geometry, class_id), ...],
    out_shape=(src.height, src.width),
    transform=src.transform,
    fill=0,
    dtype="uint8",
)
```

常见错误是只让数组尺寸相同，却没有让 CRS 和 transform 相同；结果看似能训练，标签实际整体偏移。

---

## 6. 大幅影像滑窗切片

卫星影像通常远大于 GPU 能直接处理的尺寸，需要切成 patch：

```text
tile_size = 256
stride = 256  -> 无重叠
stride = 128  -> 50% 重叠
```

切片时必须保证影像和 mask 使用完全相同的 window。边缘不足一个 tile 时有三种策略：丢弃、padding、允许可变尺寸；批训练通常选择 padding 或丢弃。

重叠切片会增加样本量，但相邻 patch 高度相关。不能先切片再随机划分，否则同一区域可能同时进入训练集和验证集。

---

## 7. 同步数据增强

颜色抖动只作用于影像；翻转、旋转、裁剪等空间变换必须同时作用于影像和 mask：

```python
image = image[:, :, ::-1]
mask = mask[:, ::-1]
```

mask 只能使用 nearest-neighbor 插值，否则整数类别会被插值成不存在的数值。

---

## 8. PyTorch Dataset 输出约定

语义分割常用：

```text
image: float32 [C, H, W]
mask:  int64   [H, W]
```

`CrossEntropyLoss` 期待模型输出 `[B, K, H, W]`，target 为 `[B, H, W]` 的长整型类别索引，不需要把 mask 转成 one-hot。

本章的 `SegmentationTileDataset` 在安装 PyTorch 时返回 tensor；未安装时返回同 shape 的 NumPy 数组，便于先验证地理处理流水线。

---

## 9. 质量检查

训练前至少检查：

1. 可视化 RGB 合成图与半透明 mask 叠加。
2. 打印 image/mask shape、dtype、最小值和最大值。
3. 检查 mask 唯一值是否只包含合法类别。
4. 随机抽查边界 tile 与 NoData 区域。
5. 按场景、行政区或空间网格划分 train/val/test。
6. 记录波段顺序、缩放方式、CRS、tile size 和 stride。

---

## 10. 常见坑

| 现象 | 原因 | 修复 |
| --- | --- | --- |
| mask 整体偏移 | CRS/transform 不一致 | 重投影后按参考栅格 rasterize |
| RGB 颜色异常 | 波段顺序或拉伸错误 | 核对 band description，做百分位拉伸 |
| loss 报 target dtype | mask 是 float | 转为 `torch.long` |
| mask 出现小数类别 | 使用了双线性插值 | mask 改用 nearest |
| 验证分数虚高 | 相邻 tile 泄漏 | 按原始 scene/空间块划分 |
| 边界 tile shape 不同 | 未处理余数 | padding 或丢弃 |
| 内存爆炸 | 一次读完整大图 | 使用 Rasterio window 按需读取 |

---

## 11. 作业与扩展

1. 将示例 `tile_size` 改为 64、`stride` 改为 32，计算 tile 数量。
2. 给 mask 增加第三类 polygon，并检查唯一值。
3. 将合成数据替换为一幅真实 GeoTIFF，打印完整 profile。
4. 实现跳过“有效像素比例低于 10%”的边界 patch。
5. 按空间四象限划分训练/验证集，并解释它比随机 tile 划分更可信的原因。
6. 扩展 Dataset，返回 tile 的 transform 与 bounds，便于预测结果拼回地图。

---

## 12. 本章小结

```text
数组 shape 对，不等于地理位置对
影像和标签必须共享 CRS、transform 与 window
空间增强必须同步，mask 插值必须用 nearest
验证集应空间独立，不能让相邻 patch 泄漏
```

完成这条数据流水线后，Part 4 的 U-Net 才有可靠输入。
