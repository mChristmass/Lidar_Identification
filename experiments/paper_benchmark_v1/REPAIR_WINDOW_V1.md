# 八小时实验窗口：干净输入保真与选择性强度修复

## 目的与固定协议

上一轮 D1/D2 在重强度噪声下有收益，但干净输入约损失 0.3–0.4 个百分点，
D2 相对 D1 的增益很小。本轮先验证修复是否跨种子有效，再处理保真问题。
这是开发集筛选，不是论文最终测试，也不预先认定模块有创新性或普遍有效。

所有新训练从对应种子的 phase1 clean CMX_B2 权重出发，固定20轮，不根据
开发集挑选轮次。保持原 mixed 增强、batch2、累计4步、workers0、pin_memory=False、
base LR=1e-5、新增参数 LR=1e-3、AdamW decay=.01、poly power=.9、clip1、AMP。
评估 clean 加7种退化×3档，共22条件。photon_proxy 是合成代理，不是校准的 SPL 物理仿真。

## 一次启动执行的顺序

必跑（预计约2.5–3小时，包含评估和诊断）：

1. D1 seed777：原恢复前端+一致性训练，补齐上一轮缺失对照。
2. S1 seed42：D1结构不变；无强度退化样本的恢复监督权重从 .05 提高到1。
3. S2 seed42：在S1上增加图像级噪声门，输入为强度以及3/7邻域高通幅值。
   门只从观测强度预测，不在推理时使用真实噪声标签。训练用合成增强标签监督。
4. S1/S2 在 clean、photon light/medium/heavy 上记录门均值和修复幅度。
5. 私有 dev 上比较固定 B2 与分别接入公开集 D1/S1/S2 seed42前端的模型。
   只搬运 restorer/noise_gate 权重，保留私有二分类骨干与分类头，不训练私有模型。
   公开灰度和私有强度的归一化、纹理和噪声域不同；迁移阴性不等同于私有训练必然无效。

损失：CE + .2×KL + 5×修复MSE；S2再加 .1×gateBCE。
有强度噪声样本的修复项为门控前、门控后MSE平均；无强度噪声样本为门控后identity MSE。
gateBCE正样本权重1、负样本.2；S1等同于恒开门。前端末层零初始化，初始分割与原模型一致。
S2门偏置初始化为-2。没有延续 D2 的跨模态融合门，以隔离修复的作用。

筛选规则同时要求：相对同种子R0，clean损失不超过.15个百分点；
21退化均值提高至少.25个百分点；photon与joint共6条件均值提高至少.8个百分点。
符合者选21退化均值更高的一项，平局按S1优先。阈值是预先声明的筛选规则，不是显著性检验。

若达标且本次启动剩余时间≥270分钟，依次运行：

6. 胜出S seed777。
7. R0 seed2025。
8. D1 seed2025。
9. 胜出S seed2025。

因此最多新增7组训练、154次公开条件评估、4次私有dev评估；诊断不算训练组。
复用 R0 seed42/777 与 D1 seed42，成功跑完后形成 R0/D1/胜出S 的三种子对照。
每个新的确认组开始前要求剩余≥65分钟，否则记录skipped，不把未完成三种子当成完整结论。
未达门槛在必跑结束后提前停止；不为凑满8小时重复无效实验。
依据上一轮5组242.7分钟，完整流程预计约6–8小时，但这是软预算，不是8小时强制杀进程。
每次重新启动重新获得8小时运行额度；已完成组复用，未完成组按epoch checkpoint续跑。
筛选决定写入decision.json后保持不变；若因预算跳过了确认组，可用相同命令续跑。
screen_only不会在重启时自动改判为confirm。

## 同步清单

同步整个 `experiments/paper_benchmark_v1/` 代码目录及 `scripts/run_repair_window_v1.sh`。
继续保留 `experiments/cmx_initial_transfer/official/` 及其已有依赖。不要覆盖或删除旧结果。
服务器上须已存在（已有则无需再次上传）：

- `data/public_semseg/nyuv2/processed/`：冻结715 train、80 dev的清单及对应图像。
- `data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_{42,777,2025}/best.pth`。
- `data/paper_benchmark_runs/mixed_control_v1/protocol.json` 与 `mixed/{last.pth,result.json,curves/}`。
- `data/paper_benchmark_runs/noise_window_v1/protocol.json` 与 `R0_seed777/`、`D1_seed42/` 的 last.pth、result.json、curves。
- `data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B2/seed_42/best.pth`。
- `data/new_data/merged/{intensity.npy,depth.npy,label/}` 与 `paper_split_v1/{train_indices.npy,dev_indices.npy}`。

无需新增原始数据、无需NJU、无需公开或私有 final_test。预检冻结源码、源权重、清单、
旧参考曲线及私有数组/dev标签的SHA256。旧R0曲线只有初始化指纹，因此还与上一轮保存的曲线指纹核对，
不能声称旧文件自身具有原来未记录的最终权重指纹。

## 启动与查看

在仓库根目录、原训练环境下：

```bash
mkdir -p data/paper_benchmark_runs/repair_window_v1
nohup bash scripts/run_repair_window_v1.sh 0 \
  > data/paper_benchmark_runs/repair_window_v1/launcher.log 2>&1 &
echo $!
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/repair_window_v1 --watch --interval 60
```

日志包含 GROUP_COMPLETE、REPAIR_PROGRESS、REPAIR_GATE、PRIVATE_DEV、REPAIR_COMPLETE。
前期分母7代表最多可能训练组数，筛选后complete中的groups_planned标明实际计划。
comparison.json中持续保存每组结果与按实际完成种子统计的均值/样本标准差；
不同方法种子数不同时不要直接把汇总当作配对显著性结论。
失败时记录failure.json并退出，watch会停止；不会继续重复显示失败前进度。
异常kill可能留下RUNNING.lock：先确认其中PID/主机对应进程已停止，再手动删除该锁后续跑。
源码或冻结输入发生变化时要求另建输出目录，不允许静默混用旧checkpoint。

同步回本轮完整 `repair_window_v1/` 目录进行分析。不要把结果、权重或私有数据提交Git。
