# 新数据补充实验

## 新增模型

- `C1L`：轻量C1，UNet基础通道数为38，参数量接近D3。
- `IDE`：Intensity + Raw Depth + Local Depth Edge，检验Depth与Edge互补性。

建议输出到：

```text
data/new_data_run0612/run3
```

完整五折：

```powershell
python scripts/run_new_data_experiments.py `
  --experiments C1L IDE `
  --folds 0 1 2 3 4 `
  --runs-dir data/new_data_run0612/run3
```

## 自动测试报告

从本次修改开始，每个模型、每个fold测试完成后都会保存：

```text
test_predictions/
test_visualizations/
test_per_image_metrics.csv
test_detailed_summary.json
```

每张测试图的可视化包含：

```text
Intensity | Depth | Local Depth Edge
GT        | Prediction | Error Overlay
```

Error Overlay颜色：

- 绿色：预测正确的目标像素TP。
- 红色：错误预测为目标的像素FP。
- 蓝色：被漏掉的目标像素FN。

旧run2模型如果需要补生成可视化，可以使用后续提供的checkpoint报告脚本，
无需重新训练。

```powershell
python scripts/report_new_data_checkpoints.py `
  --runs-dir data/new_data_run0612/run2 `
  --experiments I0 ID C1 D1 D3
```

聚合逐帧指标：

```powershell
python scripts/aggregate_new_data_test_reports.py `
  --runs-dir data/new_data_run0612/run2
```
