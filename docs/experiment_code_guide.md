# 实验代号说明

本文档汇总当前项目中使用过的实验代号。实验大致分为输入消融、
双阶段探索、双分支模型和新数据实验四部分。

## 1. 基础输入实验

| 代号 | 输入 | 模型 | 目的 |
|---|---|---|---|
| I0 | Intensity | 普通 UNet | 强度单通道基础基线 |
| ID | Intensity + Depth | 普通 UNet | 检验原始深度是否直接提供有效信息 |
| IDE | Intensity + Depth + Local Depth Edge | 普通 UNet | 检验原始Depth与Local Edge是否互补 |
| A | Intensity + Depth + Raw Depth Edge | 普通 UNet | 检验原始深度及全局深度边缘 |
| B | Intensity + Local Depth Edge | 普通 UNet | 检验局部归一化深度边缘 |

`I0`和`ID`主要用于新的1390帧数据实验。`A`和`B`来自早期输入消融。

## 2. C组：Local Depth Edge与Stage1先验

| 代号 | 输入 | 模型 | 是否使用Stage1先验 |
|---|---|---|---|
| C1 | Intensity + Local Depth Edge | 普通 UNet | 否 |
| C1L | Intensity + Local Depth Edge | 轻量UNet | 与D3进行参数量更接近的公平比较 |
| C2 | Intensity + Local Depth Edge + Stage1 Probability + ROI Mask | 普通 UNet | 是 |
| C3 | C2输入 + ROI内Local Depth Edge | 普通 UNet | 是 |

### C1

C1是当前最重要的简单基线。它不再采用真正的两阶段训练，只把
Intensity和Local Depth Edge作为两个通道输入普通UNet。

Local Depth Edge的构造过程为：

```text
原始Depth
→ 过滤无效区域
→ 高斯平滑
→ Sobel梯度
→ 在有效深度区域内进行鲁棒归一化
```

### C2与C3

C2和C3用于检验Stage1的probability和ROI能否作为先验帮助最终分割。
实验发现模型容易直接复制Stage1预测，出现先验捷径，因此没有成为最终主线。

## 3. D组：双分支模型

D组都使用：

```text
Intensity分支 + Local Depth Edge分支 + UNet式Decoder
```

两个模态先独立编码，再在不同尺度融合。

| 代号 | 融合方式 | 目的 |
|---|---|---|
| D1 | 五个尺度直接相加 | 检验独立双编码器是否优于输入通道拼接 |
| D2 | 前四尺度直接相加，仅瓶颈层使用Gate | 检验深层单点门控 |
| D3 | 五个尺度均使用空间Gate | 检验多尺度自适应融合 |

### D1

融合公式：

```python
fused = intensity_feature + projected_edge_feature
```

Edge特征经过`1×1 Conv`投影到与Intensity特征相同的通道数，然后直接相加。

### D2

前四层与D1相同，仅在最深的瓶颈层学习Gate：

```python
fused = intensity_feature + gate * projected_edge_feature
```

早期结果表明，只控制瓶颈层不足以稳定提高性能。

### D3

五个编码尺度分别学习空间Gate：

```python
fused_s = intensity_feature_s + gate_s * projected_edge_feature_s
```

每个`gate_s`都是逐像素权重。D3可以在边缘可信的位置使用Depth Edge，
在噪声或无效区域降低其影响。

D3是目前主要的复杂模型，也是新数据增加后最值得重新验证的模型。

## 4. E组：D3后续结构消融

| 代号 | 结构 | 目的 |
|---|---|---|
| E1 | 仅前三尺度使用Gate，后两尺度完全禁用Edge | 检验深层Edge是否可以删除 |
| E2 | D3 + Boundary Loss 0.1 | 检验轻量边界监督 |
| E3 | D3 + Boundary Loss 0.2 | 检验更强边界监督 |

旧数据结果显示：

- E1没有超过D3，说明深层Edge应被弱化，但不适合彻底删除。
- E2波动明显，泛化不稳定。
- E3提高Recall，但没有提高平均IoU。

因此新数据主线暂不继续运行E1、E2和E3。

## 5. Run12融合实验

Run12使用两个已经训练好的专家：

```text
专家1：Intensity-only Stage1
专家2：C1
```

### Fixed Fusion

```python
final_prob = (1 - alpha) * stage1_prob + alpha * c1_prob
```

`alpha`是验证集选择的固定权重。

### Pixel Gate-Net

Gate-Net根据两个专家的概率、不确定性和ROI，逐像素产生融合权重：

```python
final_prob = (1 - gate) * stage1_prob + gate * c1_prob
```

旧数据中Gate-Net实际可用于训练的样本较少，因此没有超过固定融合。
若未来需要重试，应使用OOF/cross-fitting预测训练Gate。

## 6. 双阶段与Stage2相关名称

| 名称 | 含义 |
|---|---|
| Stage1-only Strong | 只使用Stage1完成最终分割，并单独为最终分割调参 |
| ROI Stage2 | Stage1先定位ROI，Stage2在裁剪ROI中重新分割 |
| Full-scale Refine | 不裁剪图像，Stage2在完整分辨率上修正Stage1 |
| Residual Refine | Stage2输出`delta logits`，与Stage1 logits相加 |
| Fixed ROI | 使用固定尺寸ROI，避免缩放和还原造成几何损失 |
| Oracle ROI | 使用GT构造ROI，用于估计ROI流程的理论上限 |

早期实验发现ROI resize/restore会引入明显的几何损失，后续Full-scale和
Fixed ROI虽然解决了部分问题，但整体效果没有稳定超过强单阶段基线。

## 7. 新数据正式实验

新数据位于：

```text
D:\myComputer\pointsCloud\data\identification\data\new_data\merged
```

实验结果默认位于：

```text
data/new_data_run0612/run1
```

新数据阶段当前可运行的代号为：

| 代号 | 输入与结构 | 建议用途 |
|---|---|---|
| I0 | Intensity-only普通UNet | 必须保留的基础基线 |
| ID | Intensity + Raw Depth普通UNet | 重新检验原始Depth |
| C1 | Intensity + Local Depth Edge普通UNet | 主要简单方法 |
| D1 | 双分支直接融合 | 双编码器结构消融 |
| D3 | 双分支多尺度Gate | 主要复杂模型 |

推荐先在Fold 0筛选所有模型：

```powershell
python scripts/run_new_data_experiments.py --experiments all --folds 0
```

然后对值得保留的模型运行完整五折：

```powershell
python scripts/run_new_data_experiments.py --experiments core
```

其中`core`代表：

```text
I0 + C1 + D1 + D3
```

如果Fold 0中的ID表现较好，正式五折应显式加入ID：

```powershell
python scripts/run_new_data_experiments.py --experiments I0 ID C1 D1 D3
```

## 8. 推荐的论文消融顺序

新数据最终可以按照以下逻辑组织：

```text
I0
↓ 加入原始Depth
ID

I0
↓ 构造Local Depth Edge
C1
↓ 分离Intensity与Edge编码器
D1
↓ 加入多尺度空间门控
D3
```

对应回答四个问题：

1. Intensity单独能达到什么效果？
2. 原始Depth能否被网络直接利用？
3. Local Depth Edge是否比原始Depth更有效？
4. 双分支和多尺度Gate是否能进一步提高Local Depth Edge的利用效率？
