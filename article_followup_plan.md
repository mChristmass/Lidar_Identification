# 文章后续构思

## 1. 当前可以确定的结论

目前实验已经说明，单光子激光雷达中的 depth 信息是有效的，但原始 depth edge 容易受到无效黑边和强伪边缘影响。直接增加 depth/edge 通道可以将 Mean IoU 从 0.8058 提升到 0.8244，经过有效区域约束和 99 分位归一化得到的 local depth edge 可以进一步提升到约 0.8342-0.8364。

双阶段框架暂时没有表现出稳定优势。ROI resize 会造成明显几何损失；全图 residual refinement 基本与 Stage1 持平；直接输入 Stage1 prob 又容易形成捷径学习；固定尺寸 ROI 虽然解决了 resize 问题，但仍低于全图 local-edge 模型。

因此，文章不宜把“双阶段一定优于单阶段”作为核心结论。更可靠的主线是：分析单光子激光雷达远距离弱目标中的黑边伪响应问题，并提出针对该成像特点的深度边缘增强和多模态融合方法。

## 2. 短期实验：验证两个模型是否互补

先冻结两个已经训练好的模型：

- intensity-only Stage1；
- intensity + local depth edge 的 C1 模型。

第一步做固定权重后融合：

```text
final_prob = (1 - alpha) * stage1_prob + alpha * c1_prob
```

每折只在验证集搜索 alpha，再在 test 上评估。如果固定融合不能超过 C1，说明 intensity-only 模型提供的信息基本已被 C1 吸收，继续设计复杂 Gate-Net 的意义较小。

如果固定融合有效，再训练逐像素 Gate-Net：

```text
gate = GateNet(stage1_prob, stage1_uncertainty,
               c1_prob, c1_uncertainty, roi_mask)

final_prob = (1 - gate) * stage1_prob + gate * c1_prob
```

两个专家保持冻结，Gate-Net 只负责判断每个像素更相信哪个模型。Gate-Net 初始化时偏向当前更强的 C1，并对 Stage1 prior 输入使用 dropout，减少完全复制 Stage1 的风险。

## 3. 更推荐的主方法：单阶段双分支融合

如果后融合没有稳定提升，后续重点转向轻量双分支单阶段模型：

```text
Intensity branch      -> 外观与目标响应特征
Local-edge branch     -> 深度几何与边界特征
                     -> 多尺度门控融合
                     -> segmentation decoder
```

与简单通道拼接相比，两个分支可以分别学习强度特征和边缘特征，避免浅层卷积直接混合不同分布的数据。在编码器的多个尺度上计算融合权重：

```text
gate = sigmoid(Conv([intensity_feature, edge_feature]))
fused = intensity_feature + gate * edge_feature
```

这样网络可以在目标边缘位置提高几何特征权重，在黑边、噪声和无效区域降低 edge 分支的影响。

还可以增加 boundary supervision：

```text
total_loss = segmentation_loss + lambda * boundary_loss
```

边界标签直接由 GT mask 生成，用于明确监督网络利用 local depth edge，而不是仅将其作为普通输入通道。

## 4. 文章实验组织

建议最终实验包括：

1. 传统阈值、连通域和边缘方法；
2. intensity-only U-Net；
3. intensity + raw depth / raw depth edge；
4. intensity + local depth edge；
5. 双分支多尺度融合网络；
6. local edge 构造步骤消融；
7. feature fusion 和 boundary loss 消融；
8. 不同距离、信噪比和目标尺寸的困难子集分析；
9. 参数量、推理时间和稳定性比较。

文章核心贡献可以概括为：

> 针对单光子激光雷达远距离弱目标中无效黑边伪响应强、真实目标边缘容易被淹没的问题，提出有效区域约束的局部深度边缘构造方法，并通过强度-几何双分支自适应融合提高目标识别与分割稳定性。

