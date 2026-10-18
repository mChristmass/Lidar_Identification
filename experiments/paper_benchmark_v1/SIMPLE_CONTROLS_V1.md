# 约12小时：简单平滑与普通卷积替代对照

## 目的和矩阵

暂不展开论文定位，不新增复杂模块。验证此前收益是否可由简单替代方案解释。

|编号|方案|主要匹配比较|
|---|---|---|
|FIX_SMOOTH|PAIR输入强度做固定轻度高斯近似平滑，无可训练修正模块|PAIR、R_RES|
|D_CONV|PAIR加普通I/D/M卷积残差分割支路，不输入主干语义概率|PAIR、F_DETAIL|
|SMOOTH_DETAIL|固定平滑后接原F_DETAIL，主干与细节支路均使用平滑输入|RD_CORRECTED、F_DETAIL、FIX_SMOOTH|

三个方法×seed42/777×NYUv2/private=12个新组，各20轮。
每个种子依表中方法顺序执行，每个方法先公开后私有。公开6组×22条件=132次评价，私有6次dev评价。
预计约9–11小时，预留至12小时；12小时是软预算，不会强杀训练。服务器负载变化时可能超出。
不根据结果删组，不选最佳epoch，不为凑时长增加参数搜索。本轮不跑2025新训练。

## 对照精确定义

- 固定核为[[1,2,1],[2,4,2],[1,2,1]]/16，replicate边界；输出I'=0.5I+0.5filter(I)。
  核和混合比例预先固定，不从dev选择。保留D/M原值，所有训练视图与评价输入均使用同一处理。
  这是一个具体的轻度平滑基线，失败不代表所有传统滤波都无效。
- D_CONV是三层3×3 Conv/GroupNorm/GELU（第一层stride2），末尾零初始化1×1分类输出，
  上采样后残差加到主干logits。只输入I/D/M，不读取主干概率，也不使用真实标签作输入。
  宽度在4的倍数4..128内按参数数量最接近F_DETAIL确定，完全不读取效果。
  private宽度16：5218 vs F_DETAIL 5506参数；公开宽度24：12232 vs 11624参数，差约5.2%。
  是近似容量对照，不是严格等参数或等FLOPs；结构、语义反馈和随机初始化路径也不完全相同。
- SMOOTH_DETAIL完全复用历史F_DETAIL实现和初始化，不改写其解码过程；只在模型入口固定平滑。
- FIX_SMOOTH没有可训练candidate参数，candidate_grad_max为0是预期；CMX与S_RAW仍正常训练。
  控制组全部仅用双视图CE，无可训练修正L2，无KL、clean重建或教师。
- 保留所有历史实现不动，不加载历史last.pth继续训练；均从同域同种子phase1源权重开始。

## 协议和判读

private970train/137dev、128²、CMX_B1；NYUv2 715train/80dev、480×640、CMX_B2。
公开mixed v2，私有自然输入/翻转；双视图及CE=.5reference+.5secondary完全沿用PAIR。
AdamW base1e-5/新增1e-3、wd.01、poly.9、clip1、AMP；private batch8，公开batch2/accum4。
workers0、pin_memory=False；固定20轮末轮评价；最终测试集不访问。
公开依然是灰度RGB-D代理与人工退化，不等同于真实SPL物理噪声。

先看各对照对PAIR，再看对既有单模块/组合的配对差值；只用42/777对应结果，不能混用三种子平均。
若简单平滑接近R_RES，应收窄“可学习修正不可替代”的主张；若D_CONV接近F_DETAIL，
不能把收益独归因于语义引导。若控制组落后，也不自动证明新颖性或排除所有简单替代方案。
所有结果、失败情况如实保留；本轮为两种子对照，不是显著性证明。

## 同步5个新文件

1. experiments/paper_benchmark_v1/simple_control_model.py
2. experiments/paper_benchmark_v1/run_simple_controls.py
3. experiments/paper_benchmark_v1/test_simple_controls.py
4. experiments/paper_benchmark_v1/SIMPLE_CONTROLS_V1.md
5. scripts/run_simple_controls_v1.sh

已有此前完整代码时无需同步其他程序。无需新增数据、权重、第三方包；不要同步Mamba废弃草稿。
沿用private/NYUv2数据和phase1两域seed42/777/2025的best.pth。
2025源权重只用于完整历史协议指纹校验，本轮不训练2025。
服务器保留data/paper_benchmark_runs下这些历史记录：

- repair_window_v1/protocol.json；
- paired_window_v1的protocol/complete，PAIR和CONS seed42两域result/history/公开curves；
- paired_replication_v1的protocol/complete，PAIR seed777/2025两域result/history/公开curves；
- broad_repair_v1的protocol/complete，R_RES/F_DETAIL seed42/777两域result/history/公开curves；
- repair_detail_v1的protocol/complete，RD_CORRECTED seed42/777两域result/history/公开curves。

历史last.pth不需要，源phase1 best.pth需要。校验后冻结本轮数据、代码与历史引用指纹。

## 启动和进度

服务器项目根目录、原训练环境：

```bash
mkdir -p data/paper_benchmark_runs/simple_controls_v1
nohup bash scripts/run_simple_controls_v1.sh 0 \
  >> data/paper_benchmark_runs/simple_controls_v1/launcher.log 2>&1 &
echo $!
```

启动脚本先合成CPU测试，再预检查并正式训练。可选只做预检查（会冻结protocol，不训练）：

```bash
python -u -m experiments.paper_benchmark_v1.run_simple_controls --preflight-only
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/simple_controls_v1 --watch --interval 60
```

显示0–12组、epoch/batch及耗时。日志每组完成CONTROL_GROUP_COMPLETE，更新CONTROL_SUMMARY，
全部结束CONTROL_COMPLETE；每组第1/5/10/15/20轮简短输出。两种子报告均值/样本标准差。
metadata记录实际参数量，history记录梯度、平滑幅度及残差；参数更新审计与训练样本开关诊断保留。
失败即停并写failure.json。同命令可epoch末续跑，已有完成组校验后复用，不要同目录并发。
强杀残留锁先确认旧进程停止再改名归档，不删除文件。运行中不要更新代码或数据。

完成后回传data/paper_benchmark_runs/simple_controls_v1的全部JSON、curves和launcher.log。
last.pth可留服务器。本地只做合成CPU验证，未验证服务器CUDA峰值显存或实际耗时。
