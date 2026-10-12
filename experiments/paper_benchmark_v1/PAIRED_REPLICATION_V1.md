# 6小时：PAIR/CONS三种子复现

本轮只复现，不引入新模块，不更改损失、增强、优化器、模型或训练轮数。
保留seed42发现阶段，新增seed777/2025；不能把发现阶段当成独立确认结果。

## 矩阵与时间
PAIR、CONS × NYUv2、private × seed777、seed2025，共8个新训练组，全部20轮。
公开每组22条件，共88次新公开评价；私有4次新评价。
复用原paired_window_v1中这两个方法的4组seed42 JSON结果，合并三种子均值、
样本标准差和逐种子CONS−PAIR差值；未完成配对的种子不会提前加入平均。
两个域分开，单种子std=null；三种子也不自动构成显著性证明。

上一轮四组PAIR/CONS双域合计约180分钟。因此两个新增种子约360分钟，
不是此前误估的“一个种子6小时”。保留固定完整矩阵，不按成绩裁剪。
6小时是软预算，预检/服务器波动可能略超；不强杀，不降为16轮。
如果必须在6小时硬截止前停止，可在epoch边界后自行中断，同命令续跑；
不要并发启动，也不要把部分实验当完整三种子报告。

## 不变的协议
private：phase1 CMX_B1对应种子best.pth，970train/137dev，128×128。
NYUv2：phase1 clean CMX_B2对应种子best.pth，715train/80dev，480×640，mixed v2。
PAIR=.5参考CE+.5扰动CE；CONS增加与上一轮完全相同的正确/置信像素类别平衡KL。
不使用clean强度真值，不增加新的数据集，不访问最终测试集。
workers0、pin_memory=False，AMP；每epoch checkpoint，固定末轮评价。
直接复用paired_model.py和原训练目标；训练主体有与原runner的一致性回归测试。
公开评价CONS与同种子PAIR的退化审计一致；不需要新增B0组。

复用seed42前校验：原protocol为20轮paired_v1，final_test_access=false；
所有原数据/源权重指纹、原Python依赖代码哈希、4组history、
公开44个curve与对应结果/checkpoint标识一致。
不需要读取上轮训练后的last.pth；不能缺少原JSON或改动原训练代码。
新的复现脚本只新增文件，不修改原paired脚本/模型。
整个复现也冻结输入、代码、源权重、参考JSON，续跑不允许混入变更。

## 同步
若服务器已有上一轮代码，新增4文件即可：
- experiments/paper_benchmark_v1/run_paired_replication.py
- experiments/paper_benchmark_v1/test_paired_replication.py
- experiments/paper_benchmark_v1/PAIRED_REPLICATION_V1.md
- scripts/run_paired_replication_v1.sh

依赖已有完整experiments/paper_benchmark_v1/和CMX official代码，测试也复用旧测试辅助。
不要在训练启动后修改代码。原代码哈希不匹配时停止查明差异，不关闭校验。

已有数据保持原样。必须确认以下四个新训练源权重在服务器：
- data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1/seed_777/best.pth
- data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1/seed_2025/best.pth
- data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_777/best.pth
- data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_2025/best.pth

还需保留同位置seed42源权重（仅校验历史来源），repair_window_v1/protocol.json，
以及paired_window_v1/protocol.json、complete.json、PAIR/CONS在两域seed42的result.json、
history.json和公开curves/*.json。无需把已有文件重复上传，也不用下载新预训练权重。

## 启动
~~~bash
mkdir -p data/paper_benchmark_runs/paired_replication_v1
nohup bash scripts/run_paired_replication_v1.sh 0 \
  >> data/paper_benchmark_runs/paired_replication_v1/launcher.log 2>&1 &
echo $!
~~~

~~~bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/paired_replication_v1 --watch --interval 60
~~~

顺序：seed777的PAIR公开/私有、CONS公开/私有，然后seed2025相同顺序。
每组完成打印REPLICATION_GROUP_COMPLETE，随后更新三种子汇总（仅纳入已完成配对）。
状态含当前组、轮次/batch、已完成新组数0–8和预算剩余。最终REPLICATION_COMPLETE。
这是种子复现，不是五折；不输出五折平均。

异常立即记录failure.json并停止；同命令epoch末续跑，已完成组校验后复用。
锁和旧失败记录改名保留，不删除文件。强杀残留锁须先确认进程停止再改名归档。
结束同步回整个data/paper_benchmark_runs/paired_replication_v1/；
至少保留protocol/schedule/complete/comparison/status、launcher.log及各组JSON/curves。
训练checkpoint可留服务器用于续跑，不必反复搬回本地。
