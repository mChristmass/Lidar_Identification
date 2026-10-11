# 2–3小时：保留原始表示的多尺度上下文校正

## 目标与编号

上一轮说明：S_RAW可重现S的大部分公开收益；R_TASK无clean重建监督也有抗噪线索，
但整体收益很小且clean下降。本轮不再调整像素修复损失，而探索特征级上下文校正。

|编号|方案|
|---|---|
|B0|原CMX，10轮同预算对照|
|S_RAW|原始I/D/M结构分支，沿用既有实现，10轮|
|F_CTX|原CMX + 多尺度上下文特征残差校正，原强度图不改写|
|SF_CTX|S_RAW + 同一个上下文校正分支|

F_CTX将四层融合特征各投影至32通道，按粗到细顺序双线性上采样传递上下文；
每层采用深度可分离局部处理、GroupNorm/GELU、零初始化输出投影，形成残差加回原融合特征。
之后进入原解码器。原特征有直接加法路径，但训练仍会更新骨干，不保证天然免除clean退化。
SF_CTX同时在编码阶段执行S_RAW、在融合特征进入解码器前执行上下文校正。
上下文分支在F_CTX/SF_CTX间使用同种子的相同初始化；不改变已有S_RAW及源模型初始化。
全部仅用分割CE，不用clean重建、修复权重、教师、额外损失或八维显式描述。
这是完整候选的短程探索，不宣称已建立新颖性或物理去噪能力；参数量和计算量不完全相同。

## 为什么本轮统一10轮

四方法×两个数据集×seed42=8组，每组固定10轮末轮评价，共80个训练epoch。
公开每组仍评估完整22条件，合计88次；私有4次开发集评价。
预计2–2.5h，预留至3h；数据读取/GPU速度会影响实际耗时。
本轮不跑seed777、不根据成绩删组、不复用历史20轮结果充当10轮对照。
学习率按10轮完整poly衰减，不等于20轮方案的前10轮；不可直接比较绝对分数后宣称退步/进步。
3小时仅作为软预算显示，不强杀训练；全部八组为固定设计，机器明显变慢时可能超过3h。
这是筛选而非最终论文多种子验证，单种子标准差为null。

## 不变的协议

- private：CMX-B1，phase1 seed42 best.pth；970 train/137 dev，128×128，自然输入与确定性翻转。
- NYUv2：CMX-B2，phase1 clean seed42 best.pth；715 train/80 dev，480×640，mixed v2。
- 所有组从同域同种子的相同phase1源权重开始；新模块重新初始化，不继续训练旧实验checkpoint。
- AdamW base1e-5/新增1e-3，wd .01，poly .9，clip1，AMP；private batch8，公开batch2/accum4。
- workers0、pin_memory=False。双方最终测试集不访问，公开退化仍是代理压力测试而非SPL物理仿真。
- 冻结数据/清单/源权重/代码指纹；原输入、翻转/退化序列、epoch数、评价条件在方法间一致。
- 不混合两个域的绝对指标：公开mIoU及区域混淆矩阵；私有前景IoU/边界/误报/连通区域检出。

## 一次运行自动完成

1. 合成CPU测试与数据指纹检查。
2. 服务器随机张量的真实尺寸AMP反传、Adam状态显存与FP32推理检查。
3. 按B0、S_RAW、F_CTX、SF_CTX顺序，每方法先公开后私有，完成训练和全套开发集评价。
4. 保存梯度/参数更新、特征残差RMS，以及固定训练样本的模块开关影响。
   SF_CTX额外记录只关闭上下文分支、保留S_RAW时的预测变化，避免两分支贡献混在一起。
5. comparison.json输出F_CTX−B0、SF_CTX−S_RAW，并记录
   SF_CTX−S_RAW−F_CTX+B0作为描述性组合差值；不能单凭单种子差值证明显著互补或因果机制。

探针仍只是少量固定训练样本，不是完整开发集消融；输入审查沿用既有工具，
其中显示的描述图不表示本轮模型使用八维描述。
CONTEXT_GROUP_COMPLETE每组汇报，CONTEXT_SUMMARY汇总当前已完成结果。
状态显示当前组、epoch/batch、评价条件、完成组数和软预算剩余。

## 同步清单

同步整个experiments/paper_benchmark_v1/及scripts/run_context_window_v1.sh。
保留已有CMX official代码及环境。本轮仅需要seed42源权重：

- data/new_data/merged/{intensity.npy,depth.npy,label/} 和paper_split_v1/{train_indices.npy,dev_indices.npy}
- data/public_semseg/nyuv2/processed/ 的train/dev清单及对应强度、深度、标注
- data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1/seed_42/best.pth
- data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth
- data/paper_benchmark_runs/repair_window_v1/protocol.json（仅作冻结数据指纹）

无需新数据、clean强度真值、旧修复器或上轮训练权重。新结果单独保存，不覆盖旧实验。

```bash
mkdir -p data/paper_benchmark_runs/context_window_v1
nohup bash scripts/run_context_window_v1.sh 0 \
  >> data/paper_benchmark_runs/context_window_v1/launcher.log 2>&1 &
echo $!
```

```bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/context_window_v1 --watch --interval 60
```

同命令可从epoch末断点续跑；已完成组校验后复用。失败立即停止并记录failure.json。
正常/捕获异常退出后锁改名保留，重启时旧失败记录改名保留，不删除用户文件。
强杀残留RUNNING.lock时先确认进程已停止，再改名存档，不能并发运行同一目录。
同步回整个data/paper_benchmark_runs/context_window_v1/，保留JSON、日志与checkpoint。
