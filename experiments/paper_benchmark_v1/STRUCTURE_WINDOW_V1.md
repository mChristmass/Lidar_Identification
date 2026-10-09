# 八小时双域结构探索（2026-10-09）

## 为什么开始新窗口

observation_window_v1 的 seed42 已完成八组、88次公开条件评价、4次私有评价，209.4分钟。
相对 B0：L 的公开退化平均 +0.0603pp / 私有前景IoU +0.1096pp；
E +0.0199pp / +0.0058pp；C -0.0086pp / -0.0086pp。没有明显的描述特异性收益。
旧 E/C 私有 checkpoint 只读检查确认新增学习率1e-3、优化器2440步；
零初始化末层权重已经非零。这不能证明功能贡献，却排除了这些末层一直保持零值的解释。
本轮不重复旧 E/C 第二种子，不逐项拆八维描述；先用完整结构候选探索。

## 方法与边界

|编号|内容|
|---|---|
|B0|原 CMX，匹配20轮继续训练；不是 MiT-B0 骨干|
|L|沿用普通局部卷积残差对照，融合后处理|
|S|八维观测描述经32通道卷积/GroupNorm多尺度分支编码；与双路语义特征结合，在每层FRM之前残差更新两路；更新传播到后续编码层|
|A|每层两组查询在3×3强度/深度特征邻域中进行跨模态聚合；描述参与查询/键计算，池化有效支持率作为深度候选先验；聚合残差在FRM之前更新两路并向后传递|

A 对新增聚合路径的无支持深度候选置零权重，仍保留原CMX通路；强度候选始终可用。
支持率不是置信概率；深度特征已经包含编码上下文，因此不宣称有效掩码能精确标记全部可靠特征。
不做输入修复、不用教师/蒸馏、不加辅助损失、不需要光子直方图或相机标定。
新增输出末层零初始化，初始函数等于源CMX。随后残差可自由学习，不限于0.75–1.25缩放。
这些是待验证方案，不是已确立的新颖性结论；S/A/L参数和FLOPs不等，不是严格机制消融。

## 数据、训练、评价

- private：CMX-B1，phase1对应种子best.pth；970 train /137 dev，128×128，自然输入。
- NYUv2：CMX-B2，phase1 clean对应种子best.pth；715 train /80 dev，480×640，mixed v2。
- 每组20轮，末轮固定评价；CE，AdamW base1e-5/新增1e-3，wd .01，poly .9，clip1，AMP。
- private batch8；NYUv2 batch2/accum4；workers0/pin_memory=False。
- 同域同种子各方法使用相同源权重、样本顺序和退化序列。跨域骨干不同，不归因于单一数据因素。
- 公开每组 clean+七类三档=22条件，保留全图/自然空洞/新增空洞/有效深度/边界混淆矩阵。
- 私有保存前景IoU/Dice/P/R、边界F1、背景帧误报和连通区域匹配，tiny指面积<=80而非真实实例类别。
- 双方最终测试集不访问；不把两个域分数合并，不挑开发集最佳epoch。

## 时间安排

seed42必跑：B0、L、S、A，每方法先公开后私有，共8组。
按上一轮实测加结构开销，第一轮约3.5–4.5h，八小时预计12–16组，实际由服务器速度决定。
第二种子seed777按固定顺序B0、S、A、L，逐个方法的双域成对准入，不按成绩选择。
准入需剩余>=1.10×该方法首轮双域实测耗时×已观测第二种子减速因子+5分钟。
没有第二种子的B0双域对照就不运行第二种子候选。允许最后只完成部分方法第二种子，比较时只做同种子配对。
这避免旧规则整轮差1分钟就闲置4.5h的问题，也不保证凑齐16组。
八小时是软预算，不强杀正在运行的组；seed42必跑块和已准入双域对可能超时。
schedule.json永久记录每次准入/跳过原因；重启不根据成绩重新选择，也不重新准入已跳过的方法。
续跑是新的启动时间窗口，已完成训练/评价校验后复用；耗时估计使用保存的历史耗时而非缓存读取速度。

## 启动检查与模块检查

启动先跑合成CPU单元测试；核对冻结数据、train/dev标签、源权重与源码哈希。
随后服务器用真实尺寸的随机张量分别检查S/A、B1/B2的AMP反传、Adam状态显存和FP32推理；不读取真实样本进行这一检查。
任何检查或训练失败立即停止，写failure.json，不能把状态文件时间戳当作进程仍存活的证据。

每个实验自动保存：
- history.json：训练损失、新增参数梯度最大值、AMP跳步计数。
- parameter_audit.json：新增参数相对确定性初始化的变化，不是相对断点的变化。
- module_probe.json：每域6个固定训练样本；公开clean/mixed两种输入，mixed可能抽到clean。
  记录各尺度残差/主特征RMS比例，A的深度聚合权重/熵，以及同一已训练网络开关模块的logit与预测变化。
  这是少量样本的作用检查，不是完整开发集消融或严格因果分解，不据此选样本/成绩。
- result.json、curves/*.json、私有per_frame.json及根目录comparison.json：正式开发集结果。

## 同步

同步完整 experiments/paper_benchmark_v1/ 和 scripts/run_structure_window_v1.sh。
保留现有 experiments/cmx_initial_transfer/official/ 及已有项目依赖。
服务器已有下列数据/权重则不用重新上传：

- data/new_data/merged/{intensity.npy,depth.npy,label/} 和 paper_split_v1/{train_indices.npy,dev_indices.npy}
- data/public_semseg/nyuv2/processed/ 的train/dev清单及对应图像、深度、标注
- data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1/seed_{42,777}/best.pth
- data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_{42,777}/best.pth
- data/paper_benchmark_runs/repair_window_v1/protocol.json（仅用于已有冻结数据指纹）

不需要旧E/C或S2权重，不需要新数据或预训练下载，不需要同步最终测试数据。
本轮使用新目录，不覆盖旧结果；不要同时恢复旧实验并修改其冻结源码。

```bash
mkdir -p data/paper_benchmark_runs/structure_window_v1
nohup bash scripts/run_structure_window_v1.sh 0 8 \
  >> data/paper_benchmark_runs/structure_window_v1/launcher.log 2>&1 &
echo $!
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/structure_window_v1 --watch --interval 60
```

STRUCTURE_GROUP_COMPLETE打印每组结果，STRUCTURE_SUMMARY打印已完成种子的均值与同种子差值；
STRUCTURE_BUDGET给第二种子双域准入决定。groups_possible=16是上限，不是承诺训练数量。
状态含当前组、epoch/batch、评价条件、已完成组数、消耗/剩余预算。结束时groups_planned为实际完成数。
同命令可恢复每轮末断点，完成组校验checkpoint后复用；正常结束/捕获异常会把锁和旧失败记录改名保留。
强杀后若锁仍在，先确认对应进程已停止，再改名存档，不能并行启动同一目录。
运行结束同步回整个 data/paper_benchmark_runs/structure_window_v1/，尤其JSON、日志和checkpoint。
