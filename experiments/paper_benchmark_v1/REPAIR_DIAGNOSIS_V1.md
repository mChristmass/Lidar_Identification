# 一小时窗口：S2 权重拆分诊断

不训练，不修改已有权重，不访问最终测试集。预计20–60分钟（沿用原GPU）；
这是工作量估计，不是满60分钟强制中断。全部使用seed42，不能用来宣称跨种子显著性。

公开NYUv2冻结80张dev，四种组合：

|组合|分割网络参数及缓冲区|S2前端|
|---|---|---|
|R0_off|原mixed R0|关闭|
|R0_on|原mixed R0|开启|
|S2_off|S2训练后的分割网络|关闭|
|S2_on|S2训练后的分割网络|开启|

各评估clean、photon_proxy heavy、joint heavy、depth_noise heavy，共16次。
不替换输入为零；关闭是直接绕过修复前端，原始强度进入骨干。
R0_off和S2_on必须在mIoU误差1e-5以内复现旧曲线，否则立即报错。
记录全图、自然空洞、新增空洞、有效深度和边界区域指标。

私有冻结dev另评估原B2及接入S2前端的B2，共2次，并核对上一轮前景IoU。
记录137张图逐帧门值、门值分位数、修复幅度、目标/背景/无效深度区域修复幅度。
无区域像素时输出null，不以0冒充测量值。没有真实噪声标签，因此门值不代表已校准的真实噪声强度。

总计18次分割评估，额外前端统计不运行分割网络。默认workers0、batch1、eval、FP32。
输入指纹继承repair_window_v1并检查本次源码/参考曲线，按条件保存结果，可断点续跑。
预检需要读取现有大权重计算SHA256，开始时GPU无占用不一定是故障。

## 如何解释

- S2_off相对R0_off的变化：分割网络参数/缓冲区变化的综合影响，不能独立归因到BN。
- R0_on相对R0_off：修复前端接入原网络的影响。
- S2_on相对S2_off：修复前端在共同训练网络上的影响。
- 两种前端增益之差：权重组合的交互，可能含共同适配；不等同于训练过程的因果证明。
- 若关闭前端后仍掉点而R0_on有收益，优先考虑冻结分割网络训练前端。
- 若仅S2_on有收益，先检查共同适配，不直接认定冻结骨干可行。
- 私有门大幅开启而误报增多只能提供线索，仍需样本核查，不能据此宣称修复了物理噪声。

## 同步与启动

同步 `experiments/paper_benchmark_v1/` 和 `scripts/run_repair_diagnosis_v1.sh`。
已有官方CMX代码和运行环境保持不变。所需数据均沿用上一轮，无新增数据集：

- `data/public_semseg/nyuv2/processed/` 的train/dev清单及dev图像。
- `data/new_data/merged/` 的intensity.npy、depth.npy、dev标签、冻结train/dev索引。
- `data/paper_benchmark_runs/mixed_control_v1/mixed/last.pth` 及curves。
- `data/paper_benchmark_runs/repair_window_v1/protocol.json`、private_dev_transfer.json，
  以及 `S2_seed42/` 内last.pth、result.json、curves。
- `data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B2/seed_42/best.pth`。

项目根目录、原训练环境：

```bash
mkdir -p data/paper_benchmark_runs/repair_diagnosis_v1
nohup bash scripts/run_repair_diagnosis_v1.sh 0 \
  > data/paper_benchmark_runs/repair_diagnosis_v1/launcher.log 2>&1 &
echo $!
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/repair_diagnosis_v1 --watch --interval 60
```

每项完成打印DIAG_COMPLETE 1/18等进度，最后打印DIAG_CONTRAST。
结果含public_summary.json、private_S2.json（逐帧审计）、summary.json、complete.json。
失败写failure.json并退出；意外kill留下锁时，确认锁中进程已停止才能删除RUNNING.lock。
同步回完整 `data/paper_benchmark_runs/repair_diagnosis_v1/` 即可。
