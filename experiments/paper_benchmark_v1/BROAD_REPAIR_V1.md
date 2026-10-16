# 10小时：无clean真值的强度处理与细节支路筛选

## 实验矩阵

恢复原计划，不使用Mamba，不再调整一致性KL。以既有PAIR为固定训练对照。

|编号|相对PAIR的新增内容|种子|
|---|---|---|
|R_RES|3层卷积预测强度残差，零输出初始化，裁剪至[0,1]；分割监督+原始残差L2|42、777|
|R_FILTER|按位置混合原图、3/7/15窗口均值，权重由I/D/M预测；凸组合、近恒等初始化|42、777|
|F_DETAIL|半分辨率16通道I/D/M细节支路，结合模型自身语义概率，残差修正输出logits|42、777|

每个方法、种子同时跑private与NYUv2，共12个新组，各20轮，使用末轮模型。
公开6组各评估22条件：clean + 7退化族×3强度，共132次；私有6次dev评价。
顺序：seed42全部三个方向（每个方向先公开后私有），再seed777同样顺序。
不是五折实验；每个域/方法完成两种子后报告均值、样本标准差及对PAIR的配对差值。
历史PAIR seed42两域约89.3分钟，六对约8.93小时；新增模块成本未在服务器实测。
预计约9–10小时，可能超时。10小时为软预算，不强杀、不按dev表现删除组或缩短训练。

## 解释边界

三个候选均在S_RAW/PAIR基础上新增结构；R_RES不是旧R_TASK结果的简单复用。
R_RES/R_FILTER仅根据语义标签优化输入处理，不是已验证的物理去噪或真实强度重建。
R_FILTER会保留少量初始化平滑（identity softmax bias=6），并非严格恒等。
F_DETAIL使用本模型预测概率，不使用真实标签作为推理输入；所有新增模块推理时保留。
这些是候选筛选，不先验宣称创新成立。参数量/计算量不相同，后续有有效方向才补容量对照。
不比较不同种子集合的平均值；两种子不等于统计显著性，seed42参与了历史探索。

## 固定训练协议

- private：970 train/137 dev，128×128，phase1 CMX_B1同种子best.pth。
- NYUv2：715 train/80 dev，480×640，灰度强度代理+原始深度/有效性，phase1 clean CMX_B2同种子best.pth。
- 公开训练沿用mixed v2（含clean），私有自然数据/翻转；第二视图沿用PAIR的强度噪声及有效深度块遮挡。
- CE=.5参考视图+.5第二视图；R_RES/R_FILTER另加.01倍两视图平均修正平方惩罚。
- 不使用clean重建真值、KL、一致性教师或额外修复权重；不读取最终测试集。
- AdamW base1e-5/新增1e-3、wd .01、poly .9、clip1、AMP；private batch8，公开batch2累积4。
- workers=0、pin_memory=False。epoch末保存模型/优化器/AMP状态，逐epoch确定性重置数据随机种子。
- 记录候选梯度、参数更新、修正幅度/恒等权重/细节残差；训练样本上单独关闭候选支路检查影响，保留S_RAW。

## 同步清单

已有前几轮完整代码的服务器，仅新增以下5个文件，保留相对目录：

1. experiments/paper_benchmark_v1/broad_window_model.py
2. experiments/paper_benchmark_v1/run_broad_repair.py
3. experiments/paper_benchmark_v1/test_broad_repair.py
4. experiments/paper_benchmark_v1/BROAD_REPAIR_V1.md
5. scripts/run_broad_repair_v1.sh

无需新增数据、预训练权重或安装依赖。不要同步mamba_pilot.py、run_broad_mamba.py两个废弃草稿。
两个草稿在本地保留，已从本轮代码指纹和训练依赖排除。
历史paper_benchmark_v1代码及cmx_initial_transfer/official代码须保持原样；只同步上述5文件即可。

服务器需保留：
- data/new_data/merged及paper_split_v1；data/public_semseg/nyuv2/processed。
- phase1_v1/private/tune/CMX_B1/seed_{42,777,2025}/best.pth。
- phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_{42,777,2025}/best.pth。
  2025仅用于验证历史完整协议的源权重指纹，本轮不训练2025。
- repair_window_v1/protocol.json。
- paired_window_v1的protocol/complete、PAIR和CONS seed42两域result/history/公开curves。
- paired_replication_v1的protocol/complete、PAIR seed777两域result/history/公开curves。
  上述目录均在data/paper_benchmark_runs；历史训练last.pth不需要。

## 启动与续跑

服务器项目根目录、原训练环境中执行：

```bash
mkdir -p data/paper_benchmark_runs/broad_repair_v1
nohup bash scripts/run_broad_repair_v1.sh 0 \
  >> data/paper_benchmark_runs/broad_repair_v1/launcher.log 2>&1 &
echo $!
```

脚本先执行合成CPU测试，再检查数据、源权重与历史代码/结果指纹；任何失败立即退出。
可选先检查但不训练：

```bash
python -u -m experiments.paper_benchmark_v1.run_broad_repair --preflight-only
```

此检查会冻结新目录的protocol.json。正式训练用同一份代码和参数，勿在中途更新代码。

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/broad_repair_v1 --watch --interval 60
```

status.json显示组数0–12、epoch、batch、耗时及软预算剩余时间；不是精确GPU剩余时间估计。
launcher.log在第1/5/10/15/20轮打印简短进度；每组完成打印BROAD_GROUP_COMPLETE，
BROAD_SUMMARY报告已完成种子集合的绝对均值/标准差与配对差值；全部结束BROAD_COMPLETE。
单种子std=null，不是NaN。遇失败写failure.json并停下，不后台跳过失败组。
相同命令可从epoch末续跑并复用已完成组。不要同时对同一目录启动多个进程。
若强杀留下RUNNING.lock，先确认旧进程已停止，再将锁改名保留，勿直接删除或并发重启。

## 回传

完成后同步data/paper_benchmark_runs/broad_repair_v1下的全部JSON、curves及launcher.log。
last.pth可留服务器。重点比较公开clean/退化平均/强噪声，以及私有目标IoU/边界/误报/小目标。
本地验证仅为合成CPU测试，不能替代服务器CUDA、显存和实际耗时验证。
