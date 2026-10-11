# Paired window v1：8–10小时双域训练机制筛选

## 实验编号

|论文工作编号|代码/目录名|训练内容|
|---|---|---|
|E0|B0|原CMX，参考视图CE|
|E1|S_RAW|原始I/D/M结构分支，参考视图CE|
|E2|PAIR|S_RAW，参考与扰动视图各0.5倍CE|
|E3|CONS|E2 + 语义一致性|
|E4|MASK|E2 + 隐藏观测位置的融合特征预测|
|E5|BOTH|E2 + 上述两种约束|

每方法都跑NYUv2和private，seed42，共12组，不是五折交叉验证。
全部使用冻结训练/dev划分，不访问最终测试集。
公开每组clean+7族×3级退化，共22条件；整轮132次公开评价、6次私有评价。
优先20轮：启动时先对全部方法做真实尺寸的合成GPU计时和显存检查，
依据10小时软预算选择所有组统一20/16/12轮，然后冻结，不能根据成绩改轮数。
若连统一12轮也无法满足估计预算，正式训练前报错停止。
合成计时包含前反传、Adam状态、FP32推理，另加I/O和25%余量；
CONS/BOTH的合成检查强制所有非忽略像素参与一致性，以免随机网络低置信度漏测计算/显存。
真实训练绝不强制该支持区域，仍执行正确性和置信度筛选。
计时并非服务器端端到端耗时保证，不强杀任务；续跑沿用已冻结轮数和相同预算参数。

## 数据与目标

参考视图沿用原流程：private自然输入和确定性翻转，NYUv2 mixed v2（含现有退化）。
因此“参考”不等于“clean真值”，不引入新教师权重或真实干净强度标签。

E2–E5的第二视图完全相同：在同一参考观测上增加高斯强度扰动和块状深度隐藏。
private：归一化强度噪声std=.01，8像素块、隐藏概率.10。
NYUv2：std=.03，16像素块、隐藏概率.25。
概率表示块被选中的概率，实际隐藏量以每epoch hidden_fraction记录，不承诺逐图精确比例。
仅隐藏参考视图中M=1的位置，同时将D/M置0；已有无效位置不充当重建真值。
坐标与标签不变，不重新归一化D，不按模型成绩改增强；这是代理增强而非SPL物理噪声模拟。
E0/E1不额外看第二视图；E2专门控制额外视图、监督次数、BN更新等组合效应，
不能把E2相对E1的收益完全解释为某个单一因素。

一致性：
- 参考logits停止梯度；不是固定/EMA教师，不需额外前向。
- 仅真实标签非255、参考argmax正确且置信度>=.5的位置使用KL。
- 每个参与类别先平均，再对类别平均，避免私有背景像素数量压倒前景。
- 参考前向保持训练模式，包含相同模型的训练态随机性；不是无噪声权威答案。
- 参考和扰动两路都有真实标签CE，不完全依赖自训练伪标签。

特征补全：
- 目标为参考视图前两个融合尺度的停止梯度特征，不是强度图或深度图。
- 扰动特征通过两个训练期卷积预测头，计算通道余弦距离。
- 只在人工隐藏且非忽略标签区域监督；降采样时按有效监督面积加权。
- 天然空洞没有对应补全真值；能否迁移到天然空洞须看dev区域评价。
- 预测头只参与训练，部署推理仍为S_RAW，不运行辅助头。
- 这是掩码特征自蒸馏/辅助任务，不能先验宣称物理重建或方法新颖性。

损失：
E0/E1: CE(ref)。
E2: .5 CE(ref)+.5 CE(secondary)。
E3/E4/E5在E2上增加ramp*(.2 KL+.1 feature_loss)，仅启用对应项。
ramp=min(1,epoch/5)。不同项尺度记录在history，未做超参数网格。
参考与扰动顺序反传后仅做一次优化器更新，避免同时保留两份完整反传图。

## 训练公平性与诊断

同域所有组从对应phase1 seed42源权重开始；不继续训练上轮context权重。
private B1，970train/137dev，128×128；NYUv2 B2，715train/80dev，480×640。
所有S_RAW分支初始化相同，MASK/BOTH预测头初始化相同，初始化不改变其他随机状态。
AdamW：base1e-5、adapters1e-3、wd.01、poly.9、clip1。
private batch8/accum1；public batch2/accum4；AMP；workers0、pin_memory=False。
每组固定末轮评价，不按dev挑epoch。不同方法的参数/计算量不完全相同。

每epoch保存：
总loss、参考/扰动CE、一致性/补全损失、置信度筛选覆盖率、隐藏像素比例、
新增参数梯度最大值、AMP跳步次数。参数最终更新量和固定训练样本S_RAW开关探针另存。
探针不是全dev消融；辅助头训练期有效性由梯度、参数更新和loss检查。
公开保存自然空洞/新增空洞/有效区域/边界等混淆矩阵；
私有保存前景IoU、边界、误报、连通区域和小区域检出及分母。
comparison.json包含CONS−PAIR、MASK−PAIR、BOTH−PAIR和描述性交互差值。
单种子std=null，不声称显著性；不得混用旧10/20轮结果当本轮匹配对照。

## 同步到服务器

推荐同步整个 experiments/paper_benchmark_v1/ 以及 scripts/run_paired_window_v1.sh。
本轮新增5文件：
- experiments/paper_benchmark_v1/paired_model.py
- experiments/paper_benchmark_v1/run_paired_window.py
- experiments/paper_benchmark_v1/test_paired_window.py
- experiments/paper_benchmark_v1/PAIRED_WINDOW_V1.md
- scripts/run_paired_window_v1.sh

若此前context版本已完整同步，只需这5个新增文件。
测试复用test_context_window.py工具，不能只上传runner和model。

沿用已有数据与依赖，不需要新增数据：
- data/new_data/merged/{intensity.npy,depth.npy,label/}
- data/new_data/merged/paper_split_v1/{train_indices.npy,dev_indices.npy}
- data/public_semseg/nyuv2/processed/ 的train/dev清单及图像/深度/标注
- data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1/seed_42/best.pth
- data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth
- data/paper_benchmark_runs/repair_window_v1/protocol.json（仅用于冻结数据指纹）
- experiments/cmx_initial_transfer/official/ 及原训练环境

无需context结果、seed777、clean强度真值或NJU数据。
完整代码/训练与dev数据/源权重/轮数计划冻结哈希，变化时拒绝混入已有结果。
代码同步应在启动前完成，训练中不要再更新该目录。

## 启动与进度

服务器项目根目录、原conda环境：

~~~bash
mkdir -p data/paper_benchmark_runs/paired_window_v1
nohup bash scripts/run_paired_window_v1.sh 0 10 \
  >> data/paper_benchmark_runs/paired_window_v1/launcher.log 2>&1 &
echo $!
~~~

第二个参数为预算小时；只有8小时就将10改为8。续跑保持原参数。
首次启动自动先跑合成CPU测试，再做GPU校准、数据预检、12组训练与评价。
每组完成打印PAIRED_GROUP_COMPLETE，随后PAIRED_SUMMARY；全部完成打印PAIRED_COMPLETE。
每轮有简短日志，状态文件在batch间刷新。

~~~bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/paired_window_v1 --watch --interval 60
~~~

状态包括runtime_calibration/preflight/training/evaluation/complete/failed，
以及当前组、epoch/batch、完成组数、软预算剩余。preflight阶段GPU空闲正常。
启动估计和实际轮数在schedule.json及日志PAIRED_SCHEDULE查看。
本流程不是五折；不输出虚构的五折平均。

同命令可从epoch末继续；完成组校验后复用，失败立即记录并停止。
正常退出保留改名后的锁；旧failure重启时改名保留，不删除文件。
强杀后如残留RUNNING.lock，应先核实该进程已停止，再手动改名归档；
不要同时向同一个结果目录运行两个训练任务。

结束后同步回整个data/paper_benchmark_runs/paired_window_v1/，
至少保留schedule/protocol/complete/comparison/status、device_smoke、launcher.log，
各组history/result/parameter_audit/module_probe、公开curves与私有per_frame。
checkpoint建议留在服务器供后续分析/续跑；有空间可一起同步。
