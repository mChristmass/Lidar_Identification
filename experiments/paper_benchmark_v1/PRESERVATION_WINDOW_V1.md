# 2–3小时：固定教师保真与冻结分割网络

三组全部从同一个已训练mixed R0 seed42 epoch20出发，再训练20轮。
不从S2权重出发，不搬运已有修复器；新前端沿用S2结构和零修复初始化。
只用NYUv2冻结train/dev，不访问最终测试集，不进行私有训练。

|组|含义|分割网络|额外约束|
|---|---|---|---|
|P0，目录R0_seed42|匹配额外轮数的R0继续训练|正常训练|原mixed CE|
|P1|固定R0教师保真|正常训练|无强度噪声样本对固定R0输出的KL|
|P2|冻结网络重新学前端|参数与缓冲区均冻结|原S2损失|

P1/P2原损失为CE+.2配对KL+5修复MSE+.1门BCE；P1再加1.0保真KL，温度2。
保真样本包括clean和仅深度退化，教师看到与学生相同的观测输入，不使用噪声标签推理。
教师始终eval且无梯度。P2虽冻结分割网络，仍允许梯度经过分割网络传到前端；不能对学生整体no_grad。
每次切换train模式都重新固定骨干/解码器eval，并在训练完成后逐项检查其参数及缓冲区未变。

batch2、accum4、workers0、pin_memory=False、AMP；AdamW，base LR1e-5、前端1e-3，
weight_decay .01，poly .9，clip1。统一seed42、20轮mixed增强和固定末轮评估，不挑最佳dev轮次。
三组各22条件（clean+7类退化×3档），共66次条件评估。

主要收益与本轮P0比较，避免把额外训练收益计入模块；同时保存相对旧R0的收益。
筛选仍为clean损失≤.15pp、21退化平均收益≥.25pp、6强度/联合条件平均收益≥.8pp。
P0自身的gain字段以旧R0为参照；P1/P2的gain字段以本轮P0为参照，comparison_role明确标识。
这是机制筛选，不是固定教师项的完整消融；若P1有效，后续仍需同起点S2无固定教师对照。
冻结P2和P1更新参数数量不同，轮数相同不代表计算量相同；不宣称完全相同FLOPs。
预计2–3小时，P1额外教师推理可能增加耗时；不强制到点杀进程，不自动扩种子。

## 同步

同步整个 `experiments/paper_benchmark_v1/` 和 `scripts/run_preservation_window_v1.sh`。
沿用服务器已有官方CMX代码及环境。所需数据/权重：

- `data/public_semseg/nyuv2/processed/` 冻结train/dev清单与图像。
- `data/paper_benchmark_runs/mixed_control_v1/mixed/` 内last.pth、result.json、curves。
- `data/paper_benchmark_runs/repair_window_v1/protocol.json` 用于校验已诊断R0与清单指纹。

不需要重新上传私有数据，也不需要S2权重。代码、参考曲线、初始化权重和清单均冻结指纹；
续跑如果发生变化要求新目录。断点按epoch保存，失败写failure.json并停止。
正常完成后重新执行只校验并复用结果。强制终止后确认锁中进程已停止，才可删除RUNNING.lock。

## 启动

```bash
mkdir -p data/paper_benchmark_runs/preservation_window_v1
nohup bash scripts/run_preservation_window_v1.sh 0 \
  > data/paper_benchmark_runs/preservation_window_v1/launcher.log 2>&1 &
echo $!
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/preservation_window_v1 --watch --interval 60
```

每组完成打印GROUP_COMPLETE，末尾PRESERVATION_SUMMARY打印三组比较。
status.json包含group_index/groups_possible与epoch；comparison.json保存完整对照和筛选结论。
本轮只有一个种子，不输出虚假的多种子标准差。同步回完整preservation_window_v1目录分析。
