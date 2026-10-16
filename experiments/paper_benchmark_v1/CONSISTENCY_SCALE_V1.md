# 8–10小时：PIXEL复现与一致性标量尺度对照

## 目的与停止边界
最后一次针对“类别分配还是损失尺度”的集中验证，不继续LOCAL/MASK，不做权重网格。
PIXEL补seed777/2025；新SCALE跑seed42/777/2025，两域都跑，全部固定20轮。
共10个新组：5个公开、5个私有，公开每组22条件，合计110次公开评价。
原PAIR/CONS三种子与PIXEL seed42只读复用，完成后四方法均具三种子结果。
计划约8小时，预算10小时为软预算；不强杀、不按分数删组、不缩短轮数。
根据结果决定是否保留一致性为辅助策略，不先验宣称形成论文核心创新。

## SCALE定义
用同一模型、同一参考/扰动视图，在原CONS同样的正确/置信支持区域计算：
L_class = 每类平均KL再对参与类别平均；
L_pixel = 所有支持像素平均KL；
r = stop_gradient(L_pixel / L_class)；
L_scale = r * L_class。
参考概率、比值分子和分母都停止梯度。
注意：若不停止比值梯度，这个表达式会直接退化成像素平均KL，无法作为对照。
L_class<=1e-8时辅助项置零；仅为数值安全，日志记录匹配误差。

这保证当前模型、当前batch上的辅助损失标量通常等于该模型自己的像素平均KL，
但梯度方向仍为类别等权梯度乘以r。
它不保证不同训练轨迹间的损失相等，不匹配梯度范数，也不能完全隔离所有“强度”因素。
这是动态标量归一化对照，不是已证实的新方法。
r没有依据dev指标调整；原外部系数.2与5轮ramp保持不变。

PIXEL直接调用原consistency_refine_model.py中的训练目标，不重写其实现。
两组均用原S_RAW/CONS网络，不增加推理参数和路径。
同域同种子源权重、双视图增强、CE=.5ref+.5secondary、顺序反传、优化器与AMP完全沿用。
private B1/970train/137dev/128²；NYUv2 B2/715train/80dev/480×640。
不需要clean强度真值，不读取最终测试集。

## 记录与汇总
SCALE新增balanced_kl、pixel_kl、scale_ratio、scale_match_abs_error、effective_cons_weight，
保留CE、实际约束覆盖、前景支持、参数更新/梯度、AMP跳步和区域评价。
比较PIXEL对PAIR/CONS，SCALE对PAIR/CONS/PIXEL；每对比较只使用双方都已完成的种子。
comparison.json保存各组原始指标、逐种子差值和配对差值均值/样本标准差。
单种子std=null；seed42用于过方向选择，不能冒充独立确认；三种子不等于显著性证明。

## 同步
已有此前完整代码时，仅新增5文件：
- experiments/paper_benchmark_v1/consistency_scale_model.py
- experiments/paper_benchmark_v1/run_consistency_scale.py
- experiments/paper_benchmark_v1/test_consistency_scale.py
- experiments/paper_benchmark_v1/CONSISTENCY_SCALE_V1.md
- scripts/run_consistency_scale_v1.sh

依赖原完整paper_benchmark_v1/和CMX official代码，尤其paired_model.py、
consistency_refine_model.py、run_paired_replication.py和测试辅助文件。
旧代码不改动；启动前核对所有历史代码指纹，运行中不要同步更新代码。

数据沿用原数据。源权重为phase1的private CMX_B1与公开clean CMX_B2，
均需seed42/777/2025的best.pth；前几轮服务器已有，不必重复上传。
保留以下目录的协议/完成记录与所需组的result/history/公开curves：
- paired_window_v1：PAIR/CONS seed42双域；
- paired_replication_v1：PAIR/CONS seed777/2025双域；
- consistency_refine_v1：PIXEL seed42双域。
这些目录都在data/paper_benchmark_runs/；另保留repair_window_v1/protocol.json。
不需要历史训练last.pth，仅需要原phase1源权重；无需新增数据集。

复用前核验历史协议20轮、未访问final、完整轮次、源数据/权重/代码哈希，
以及曲线checkpoint标识和分数一致；新协议冻结所有引用的历史JSON。

## 启动
~~~bash
mkdir -p data/paper_benchmark_runs/consistency_scale_v1
nohup bash scripts/run_consistency_scale_v1.sh 0 \
  >> data/paper_benchmark_runs/consistency_scale_v1/launcher.log 2>&1 &
echo $!
~~~

~~~bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/consistency_scale_v1 --watch --interval 60
~~~

先SCALE seed42双域，再seed777的PIXEL/SCALE双域，最后seed2025同样执行。
进度显示0–10个新组；每组完成SCALE_GROUP_COMPLETE，汇总SCALE_SUMMARY。
同命令epoch末续跑，完成组校验后复用。失败立即记录并停止，锁/旧失败改名保留。
强杀残留RUNNING.lock须先确认进程停止再改名归档，禁止同目录并发运行。
结束同步回data/paper_benchmark_runs/consistency_scale_v1/的全部JSON、curves、日志。
checkpoint可以留在服务器；本地仅运行合成CPU测试，不启动正式训练。
