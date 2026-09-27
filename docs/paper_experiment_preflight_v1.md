# 论文实验前置协议 v1（2026-09-22）

## 已冻结的决策

1. 私有数据按采集组隔离：train 970、dev 137、final_test 283。开发期间只允许读取 train/dev。
2. 退化参数仅由私有 train 统计得到。深度无效率的轻/中/重目标为
   13.65% / 18.95% / 24.56%，最大空洞面积目标约为
   6.89% / 10.97% / 16.99%。
3. 公开集采用 NYUv2 40 类标准 795/654 划分。外观输入是 RGB 转灰度的单波段代理；
   深度使用 `rawDepths`，禁止用补洞后的 `depths` 代替。
4. 公开集人工退化是“与私有训练集统计对齐的可重复压力测试”，不是物理精确的单光子仿真。
5. 所有修改输入形式的公开集实验都必须重新训练基线。原论文 RGB-D SOTA 数字只作背景，
   不能和灰度/退化输入的结果直接作胜负结论。

## 正式实验之前的通过条件

- `prepare_private_paper_split.py verify` 通过。
- 私有退化画像含来源哈希，且只读取 train。
- NYUv2 解包审计确认 795/654、480×640、40 类与 raw depth 空洞。
- 退化算子确定性测试和轻/中/重单调性测试通过。
- T8、CMX-B0/B1/B2 在目标尺寸完成前向/反向服务器 smoke test。
- 首次服务器 smoke test 固定 `workers=0`，不启用 pin-memory；稳定后才单独测试提速设置。
- 在实验登记表写明模型、种子、分辨率、训练轮数、学习率和选择规则。

## 阶段顺序

阶段 1 只回答“现代框架是否比 T8 更有上限”：T8、CMX-B0/B1/B2 在私有自然数据、
NYUv2 clean 与 joint-medium 上用三个种子比较。阶段 2 固定 T8 和选中的 CMX，建立完整
退化曲线；同一个 clean 或 mixed-corruption 训练检查点必须评估全部退化条件，禁止针对每个
测试退化分别重训模型。阶段 3 才根据失效模式设计和加入新模块。最后冻结全部设置，再解锁 private
final_test 与 NYUv2 official test。

精确矩阵见 `experiments/paper_benchmark_v1/experiment_registry_v1.json`。
