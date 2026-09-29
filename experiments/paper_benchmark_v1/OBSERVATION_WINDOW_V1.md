# 八小时：公开/私有共同筛选观测结构描述

## 研究问题和边界

从处理后的Intensity、Depth和有效掩码提取观测支持及局部结构描述，是否有助于信息表达/融合？
这些描述是原输入的确定性变换，不是新传感器信息，不是已校准可靠性或光子后验。
不先证明各分量独立有效；先尝试完整候选，分别观察两个域的效果与副作用。
公开数据与私有数据同等参与方向筛选，不根据私有结果决定是否跑公开，也不反过来。

## 实验矩阵

|组|设计|
|---|---|
|B0|原CMX，匹配20轮继续训练预算|
|L|每个融合尺度后加入普通局部卷积残差，作为额外局部处理对照|
|E|将8维观测描述投影后，直接残差注入每个尺度的融合特征|
|C|8维观测描述与双模态特征的通道均值/最大值共同产生条件权重，分别调制两种模态进入FFM的特征|

E是显式描述的特征级直接注入，不是修改预训练首层成额外输入通道。
C的倍率范围为0.75–1.25，可增强也可抑制；不是按1-U直接关断，也不修复原图。
新增末层均零初始化，初始输出严格等于各自原模型。各组都只用分割CE，无修复、蒸馏或辅助损失。
L/E/C并非严格等参数或等FLOPs，结果记录新增参数量。本轮是完整方案探索，不能单凭胜负宣称机制已被证明。

- private：CMX-B1，每个种子从phase1对应best.pth开始；自然数据，训练970帧、dev137帧，128×128。
- nyuv2：CMX-B2，每个种子从phase1 clean对应best.pth开始；训练715帧、dev80帧，480×640；mixed v2增强。
- 每组固定20轮末轮评估。私有和公开用不同已有强基线，不能将跨域差异全部归因于数据。
- 各域内部四组的初始化、样本顺序、翻转/退化序列、基础LR与轮数相同。
- AdamW：base1e-5，新增参数1e-3，weight decay .01，poly .9，clip1，AMP。
- private batch8/accum1；public batch2/accum4；workers0、pin_memory=False。

seed42必跑：四组×两域=8组。按方法交替运行公开/私有，先跑双方B0。
seed777仅由耗时决定：剩余分钟≥1.25×首种子实际/保存耗时+10，才启动双方完整矩阵。
不按成绩筛掉任何方法。最大16组、320个训练epoch；公开最多176次条件评估，私有8组dev评估。
此前公开20轮约40–50分钟，私有128图预计较快：本轮完整两种子估计6–8小时；
若首轮实测偏慢，则完成双方seed42后结束。8小时是软预算，不是强制到点杀训练。
分母16在前期表示最多可能组数，完成状态另记实际计划。两种子不是三种子或跨采集组显著性证明。

## 描述与检查

8维：[中心valid、3/7邻域支持率、强度3/7邻域绝对残差、有效深度3/7邻域绝对残差、有效深度5邻域标准差]。
在现有归一化输入上计算，使用有效掩码加权深度统计；无有效点时相关深度描述为0，另保留支持为0。
边界使用复制padding，FP32局部矩避免AMP数值问题。没有3D平面拟合，不需要相机标定。
没有假设方差大就是不可靠，没有各帧独立拉伸描述数值。
启动先导出每域6个等间距训练样本的输入/描述灰度图及均值，位于input_audit。
这是有限的计算/表达检查，不是可靠性验证；仅用训练样本，不据此选模型。

## 指标与解释

公开：每个固定checkpoint评估clean+7类×3档退化，保存全图、自然空洞、新增空洞、有效深度、边界混淆矩阵。
强度是灰度代理，退化是v2固定压力测试，不是SPL物理仿真。
私有：前景IoU/Dice/Precision/Recall、边界F1、背景帧误报率，以及逐图TP/FP/FN。
附加8连通GT区域IoU>=.5的一对一匹配召回、小区域面积<=80像素的召回及分母。
这是连通区域指标，不是真实实例召回，也不等于历史人工Tiny类别；空分母为null。

comparison.json分别报告两域、各方法的均值/样本标准差及同种子相对B0的差值。
公开额外展示heavy photon/joint；私有背景误报差值越低越好。
不合并两个域的绝对分数，不自动宣称成功，不使用此前repair的严格门槛裁剪探索方向。
单域有收益时作为待解释的域依赖线索；双域一致收益才优先进入后续复现/消融。
L同样有效时不能把收益归于观测描述；E优于C时保留简单结构。
双方最终测试集均不用于本轮，所有结论仍是开发集探索。

## 同步清单

同步整个 `experiments/paper_benchmark_v1/` 以及 `scripts/run_observation_window_v1.sh`。
保留现有 `experiments/cmx_initial_transfer/official/` 和环境依赖。
服务器需已有（无需重复上传已有文件）：

- `data/public_semseg/nyuv2/processed/` 的train/dev清单及对应intensity/depth/label图像。
- `data/new_data/merged/{intensity.npy,depth.npy,label/}` 与 `paper_split_v1/{train_indices.npy,dev_indices.npy}`。
- `data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1/seed_42/best.pth` 和 `seed_777/best.pth`。
- `data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth` 和 `seed_777/best.pth`。
- `data/paper_benchmark_runs/repair_window_v1/protocol.json`：沿用数据冻结指纹，不读取repair权重。

不需要新增数据集、S2权重、原始光子直方图或3D标定。本轮不使用NJU。
预检核对来源参数、冻结清单/私有数据，记录所有使用的训练/dev图像与源权重、源码哈希。
权重和数据哈希阶段GPU暂时空闲属正常，状态为preflight。

## 启动

```bash
mkdir -p data/paper_benchmark_runs/observation_window_v1
nohup bash scripts/run_observation_window_v1.sh 0 \
  > data/paper_benchmark_runs/observation_window_v1/launcher.log 2>&1 &
echo $!
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/observation_window_v1 --watch --interval 60
```

训练状态含域/方法/种子/epoch/batch，OBS_GROUP_COMPLETE每组总结，OBS_SUMMARY给已完成种子均值与差值。
异常退出保存failure.json，不继续运行后续组；每epoch保存last.pth，可用同命令续跑。
已有输出的输入或代码指纹变化时拒绝混用。正常/捕获异常退出将锁改名存档，不删除文件。
失败记录在下一次启动时也改名存档。强杀留下RUNNING.lock时，先确认记录中的进程已停止，
再手动将锁重命名保留，不能并行启动相同输出目录。
已写入的second_seed_decision保持不变；因时间未获准的第二种子不会在重启时自动改判。
续跑重新获得本次启动的8小时额度，但估计使用已保存组耗时，不因复用结果误以为训练很快。

结束后同步回完整 `data/paper_benchmark_runs/observation_window_v1/`。
