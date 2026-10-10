# 八小时：结构描述对照与无 clean 重建监督的任务校正

## 本轮目标

上一轮 structure_window_v1 完成16组、耗时423.15分钟。两种子公开退化平均增益：
S +0.469pp，A +0.484pp；S重度联合+2.147pp，两个种子clean均未下降。
但私有S/A前景IoU仅+0.072/+0.130pp，普通卷积L也有+0.119pp。
不能把公开收益直接归因于显式描述，或声称已在真实域形成明显优势。

本轮并行回答：S的收益是否超过同位置、近似容量的原始输入分支？
强度校正不用干净强度重建监督，能否通过真实分割标注学出收益？
不再细拆描述分量，不叠加S和修复，不测试第三个全新结构。

## 实验编号

|编号|含义|新增参数（B1/B2）|
|---|---|---:|
|B0|原CMX、匹配20轮续训；不是MiT-B0骨干|0|
|S_RAW|与S相同32通道金字塔、交换结构、逐层FRM前注入位置；分支输入直接为归一化I/D/M，不显式计算局部描述|282560|
|S|沿用八维观测结构描述金字塔，更新双模态并传递到后续编码层|284000|
|R_TASK|3→16→16→1卷积残差强度前端，和分割网络联合优化；无旧修复权重|2913|

S_RAW和S只在分支输入及第一层3/8输入通道上不同，参数差1440，约占S新增参数0.51%。
共同参数在同种子初始化一致，末层均为零，起点函数等于源CMX。
S_RAW仍有中心有效掩码，不能称为“完全不使用观测信息”；它不使用支持率、局部残差、深度标准差等显式统计。
若S_RAW与S相当，优先考虑额外分支/注入位置而非手工描述解释；这不是严格等FLOPs消融。

## R_TASK如何不依赖clean

I_corrected=clip(I+delta(I,D,M),0,1)。D、M不修改。
训练目标：CE(segmentation,y) + 0.01 × mean(delta²)。
正则惩罚裁剪前残差，防止通过饱和裁剪隐藏过大的修正；0.01为预先固定的探索权重，不做开发集搜索。
没有原始图重建MSE、clean教师、配对KL、噪声标签、置信度标签或图像级修复开关。
两域均重新初始化前端、联合训练，不进行公开修复器到私有模型的零样本接入。
已有分割标注仍被使用，因此不是无监督方法。输出是任务相关校正，不能据分割提升声称物理去噪或恢复真实无噪声强度。
公开数据仍使用灰度原观测构造mixed输入，clean条件表示没有追加退化，不是真实SPL干净观测。

## 训练与公平性

- private：CMX-B1，phase1对应seed的best.pth；970 train /137 dev，128×128，自然数据、确定性翻转。
- public：NYUv2 CMX-B2，phase1 clean对应seed的best.pth；715 train /80 dev，480×640，mixed v2。
- 四组同域同种子源权重、顺序、翻转/退化序列、轮数一致；固定20轮末轮评价。
- AdamW base1e-5 /新增1e-3，weight_decay .01，poly .9，clip1，AMP。
- private batch8；public batch2/accum4；workers0，pin_memory=False。
- B0/S_RAW/S仅CE；R_TASK另有明确记录的残差正则。R_TASK是完整方案探索，不与S宣称单因素比较。
- B0和S在新目录重新运行，便于本轮同预算对照；重复seed42/777不算新增独立种子证据。
- 每个公开模型评价clean+7类×3档=22条件，含区域混淆矩阵；私有记录前景、边界、误报、连通区域检出和逐图计数。
- 只访问训练/开发数据，不访问任何最终测试集；不将两个域分数混成一个排名。

## 八小时运行顺序

seed42必跑：B0、S_RAW、S、R_TASK，每方法先公开后私有，共8组。
预计首轮3.5–4h，两种子约7–8h；按上轮实测估计，具体由服务器速度决定。
seed777按固定包调度：
1. B0的公开+私有；
2. S和S_RAW共四组作为一整包，避免只补主方法不补同种子对照；
3. R_TASK的公开+私有。
每包要求剩余>=1.10×首种子对应组实测耗时×已观测第二种子减速因子+5分钟。
仅用耗时，不用成绩；时间不足可跳过该包后尝试更短的后续包，但无seed777基线不跑任何seed777候选。
最多16组、176次公开条件评价、8次私有评价。8h是软预算，不强杀已开始的组或已准入包。
schedule.json保存准入决定；续跑不自动重新准入已跳过包，每次启动重新计时、估计沿用保存的实际耗时。

## 验证和自动输出

启动先执行合成CPU测试，再核对数据、train/dev标注、源权重和代码SHA256。
服务器对S_RAW/S/R_TASK的真实尺寸随机张量做AMP反传、Adam状态分配和FP32推理检查。
此为硬件检查，不是用真实数据训练；本地不启动真实训练。

每组保存history.json（CE、原始残差L2、总目标、梯度、AMP跳步）、parameter_audit.json、module_probe.json。
R_TASK探针记录实际修改MAE/最大值/裁剪比例，以及同一已训练模型关闭前端后的预测变化。
S/S_RAW探针记录各层残差相对主特征的比例。探针只用固定训练样本，不是完整开发集因果消融。
comparison.json同时提供各方法相对B0的同种子差值，以及S_minus_S_RAW的严格配对种子差值。
每组完成有TASK_CONTROL_GROUP_COMPLETE；TASK_CONTROL_SUMMARY汇报已完成种子的均值与差值。
状态文件有当前组、epoch/batch、评价条件、已完成数、预算消耗/剩余；16是最大组数。

## 同步与启动

同步整个experiments/paper_benchmark_v1/及scripts/run_task_control_window_v1.sh。
保留CMX official目录和现有项目环境。服务器需要已有：

- data/new_data/merged/{intensity.npy,depth.npy,label/} 与 paper_split_v1/{train_indices.npy,dev_indices.npy}
- data/public_semseg/nyuv2/processed/ 的 train/dev 清单和对应强度、深度、标签
- data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1/seed_{42,777}/best.pth
- data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_{42,777}/best.pth
- data/paper_benchmark_runs/repair_window_v1/protocol.json（只读冻结数据指纹）

不需要新的clean强度图、旧修复器权重、structure_window_v1权重、新数据集或预训练下载。
使用新目录，不覆盖旧结果，不要同时继续旧实验并修改它的冻结代码。

```bash
mkdir -p data/paper_benchmark_runs/task_control_window_v1
nohup bash scripts/run_task_control_window_v1.sh 0 8 \
  >> data/paper_benchmark_runs/task_control_window_v1/launcher.log 2>&1 &
echo $!
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/task_control_window_v1 --watch --interval 60
```

错误立即保存failure.json并停止；每epoch原子保存last.pth，同命令可恢复；已完成组校验权重后复用。
正常或捕获异常结束将锁改名保留，旧failure也改名保留，不删除用户文件。
强杀后若RUNNING.lock仍在，必须确认对应进程已停止再改名，不允许同目录并发启动。
结束后同步整个data/paper_benchmark_runs/task_control_window_v1/，包括日志、JSON与checkpoint。
