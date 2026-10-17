# 16小时窗口：强度修正与细节支路的匹配组合验证

## 固定实验矩阵

|编号|主干输入|细节支路输入|新增训练|
|---|---|---|---|
|RD_CORRECTED|修正I、原D/M|修正I、原D/M|42/777/2025，各公开+私有|
|RD_RAW|修正I、原D/M|原始观测I/D/M|42/777/2025，各公开+私有|
|R_RES|沿用上一轮单模块原实现|无|补2025，公开+私有|
|F_DETAIL|沿用上一轮单模块原实现|原始观测I/D/M|补2025，公开+私有|

共16个新训练组、每组20轮。公开8组×22条件=176次评价，私有8次dev评价。
顺序：42的两组合双域 → 777的两组合双域 → 2025单模块双域 → 2025两组合双域。
每个方法均先NYUv2后private。没有成绩门槛，不挑选性删除组或中途改变种子。
复用PAIR三种子双域及R_RES/F_DETAIL的42/777双域；补齐后四种候选都有三种子匹配对照。
校验器还读取PAIR历史同批CONS seed42记录，但CONS不作为本轮主要比较。

历史PAIR每对两域约89分钟，8对约11.9小时；组合模块增加开销，计划约12–14小时。
16小时为预留窗口/软预算显示，不是强杀时间；硬件负载变化时不能保证16小时内完成。
不为了凑满预算增加无关实验。本地CPU合成测试不能验证服务器CUDA耗时和显存。

## 方法及公平性

两个组合的差别仅为细节支路接收修正强度还是原始观测强度；参数量、初始化与训练目标相同。
两者的细节支路均读取修正输入主干产生的语义概率（不detach）。
RD_RAW并未切断修正模块的所有细节损失梯度，只改变细节卷积分支的图像输入。
原始观测不等于clean真值；不使用clean重建、教师、一致性KL或物理噪声真值。
修正分支与细节分支分别严格匹配历史R_RES/F_DETAIL的初始化，不顺序消耗彼此随机流。
主干/S_RAW初始化不变；单模块2025直接调用原BroadCMX及原backward_objective，不重写实现。
所有新增训练从phase1同域同种子源权重开始，不继续训练上一轮last.pth。
组合推理时保留两个模块；不能把推理参数或计算开销忽略。

## 保持不变

- private：970 train/137 dev，128×128；NYUv2：715 train/80 dev，480×640，灰度代理I/D/M。
- private源CMX_B1、公开源clean CMX_B2，均使用phase1对应seed的best.pth。
- 公开mixed v2，私有自然输入/翻转；第二视图沿用PAIR扰动；全部固定20轮末轮评价。
- loss=.5CE_reference+.5CE_secondary+.01×两视图平均原始修正平方。
- AdamW base1e-5/新增1e-3，wd .01，poly .9，clip1，AMP；private batch8，公开batch2/accum4。
- workers=0、pin_memory=False；epoch末保存优化器/AMP/模型，确定性数据种子，支持续跑。
- 不访问最终测试集，公开22条件预算不变；代理退化不宣称为SPL真实物理仿真。

## 汇总与判定

每个组合分别对PAIR、R_RES、F_DETAIL做相同种子的配对差值，另比较RD_RAW−RD_CORRECTED。
报告绝对均值/样本标准差、差值均值/标准差以及实际参与种子；不混合不同种子集合。
先看公开clean/全退化平均/强噪声，再看私有目标IoU/边界/误报/小目标。
组合超过PAIR但未超过对应单模块，不等于证明叠加有效；两组合无差别也必须如实报告。
小样本三种子不是显著性或新颖性证明；42/777已用于选方向，2025也不是独立外部数据验证。
训练记录修正幅度、细节残差、候选梯度与参数更新；固定训练样本分别关掉修正/细节作诊断。
开关诊断不是重新训练消融，不能据此单独作因果论证。

## 同步（已有上一轮完整代码时仅新增5文件）

1. experiments/paper_benchmark_v1/repair_detail_model.py
2. experiments/paper_benchmark_v1/run_repair_detail.py
3. experiments/paper_benchmark_v1/test_repair_detail.py
4. experiments/paper_benchmark_v1/REPAIR_DETAIL_V1.md
5. scripts/run_repair_detail_v1.sh

不用同步Mamba草稿；没有改动任何历史训练代码。无新增数据、权重或第三方依赖。
服务器保留原始数据、phase1两域seed42/777/2025源权重，以及以下历史记录：

- repair_window_v1/protocol.json。
- paired_window_v1：protocol/complete，PAIR和CONS seed42两域result/history/公开curves。
- paired_replication_v1：protocol/complete，PAIR seed777/2025两域result/history/公开curves。
- broad_repair_v1：protocol/complete，R_RES和F_DETAIL seed42/777两域result/history/公开curves。

目录均位于data/paper_benchmark_runs。历史last.pth不需要，phase1源best.pth需要。
所有引用先检查数据/源权重/代码指纹、完整轮次和公开曲线一致性，再冻结本轮协议。

## 启动

服务器项目根目录、原训练环境执行：

```bash
mkdir -p data/paper_benchmark_runs/repair_detail_v1
nohup bash scripts/run_repair_detail_v1.sh 0 \
  >> data/paper_benchmark_runs/repair_detail_v1/launcher.log 2>&1 &
echo $!
```

脚本先合成CPU测试，之后校验并训练。可选先单独检查，不训练：

```bash
python -u -m experiments.paper_benchmark_v1.run_repair_detail --preflight-only
```

检查会冻结协议；续跑必须使用同一代码、参数和数据。状态查看：

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/repair_detail_v1 --watch --interval 60
```

显示0–16组、当前epoch/batch及耗时；软预算剩余不是精确ETA。
日志第1/5/10/15/20轮简短输出，每组结束RD_GROUP_COMPLETE，汇总RD_SUMMARY，全部结束RD_COMPLETE。
失败立即记录failure.json并停止；同一命令从epoch末续跑，已完成组校验后复用。
禁止同目录并发；强杀残留RUNNING.lock先确认旧进程停止，再改名归档，勿删除文件。

结束后同步data/paper_benchmark_runs/repair_detail_v1下所有JSON、curves和launcher.log。
last.pth可留服务器。本轮不替你在本地启动正式实验。
