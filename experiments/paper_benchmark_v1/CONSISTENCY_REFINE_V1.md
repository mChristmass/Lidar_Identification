# 3小时：一致性作用区域与聚合方式筛选

## 固定矩阵
- LOCAL：原CONS的类别等权KL，只在额外扰动区域施加。
- PIXEL：原CONS相同的正确/置信支持区域，将类别等权改为像素等权。
- 每种方法都跑NYUv2与private、seed42、20轮，共4个新组。
- 复用paired_window_v1的seed42 PAIR和CONS各双域结果，不重训旧对照。
- 公开每组22条件，共44次新评价；私有2次新评价。
- 固定末轮，不因成绩改变轮数或删组，不与三种子平均直接作方法配对比较。

LOCAL区域定义：人工隐藏的原有效深度位置，或第二视图强度与参考视图的实际差值>=2倍该域增强噪声std。
两者取并集，再与原CONS的标签非255、参考预测正确、置信度>=.5条件相交。
这是已知训练增强区域，不是物理可靠性估计，不参与推理。
强度加噪仍发生在整图；阈值区域并不覆盖所有受噪声影响的像素。
该阈值是本次预设探索参数，未以开发集调参，也不保证有效。
LOCAL保留各参与类别等权，PIXEL保留原空间支持，只改变像素/类别聚合。
两候选均不增加模型参数，部署模型与S_RAW/CONS相同，不引入clean强度真值或新教师。
PAIRED原增强、参考/扰动0.5CE、KL系数.2、5轮ramp、优化器、数据顺序、源权重均保持不变。
新增日志记录eligible_coverage、实际约束coverage、region_fraction以及私有前景支持统计。
这是单种子方向筛选，不是已经获得三种子验证的新方法。

## 时间与运行
上一轮CONS在两域合计约90分钟，两个候选约3小时，预检/机器波动可能略超。
3小时为软预算，不强杀、不缩短20轮；需要硬截止时可中断后同命令epoch末续跑。
不再重复GPU合成计时，网络和输入尺寸未改变；本地只跑合成CPU测试。
公开保留自然空洞/新增空洞/边界等区域评价；私有保留前景/边界/误报/连通小区域指标。
双方最终测试集均不访问。

## 同步
若已有上一轮完整代码，仅新增以下5文件：
- experiments/paper_benchmark_v1/consistency_refine_model.py
- experiments/paper_benchmark_v1/run_consistency_refine.py
- experiments/paper_benchmark_v1/test_consistency_refine.py
- experiments/paper_benchmark_v1/CONSISTENCY_REFINE_V1.md
- scripts/run_consistency_refine_v1.sh

依赖已有paired_model.py、run_paired_replication.py及原完整paper_benchmark_v1/、CMX official代码。
历史代码保持原样：复用对照前会核对原paired_window_v1代码/数据/源权重指纹和结果JSON。
不要仅复制runner；不要在训练运行时更新目录。

无需新增数据或权重。保留：
- 私有970train/137dev原数据和划分、NYUv2 715train/80dev清单与图像。
- phase1/private/tune/CMX_B1/seed_42/best.pth。
- phase1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth。
- repair_window_v1/protocol.json。
- paired_window_v1/protocol.json、complete.json。
- paired_window_v1中PAIR/CONS双域seed42的result.json、history.json，以及公开curves。
上述phase1/repair/paired路径均位于data/paper_benchmark_runs/。
不需要777/2025权重，也不需要paired_replication_v1结果目录或上轮训练后的last.pth。

## 启动
~~~bash
mkdir -p data/paper_benchmark_runs/consistency_refine_v1
nohup bash scripts/run_consistency_refine_v1.sh 0 \
  >> data/paper_benchmark_runs/consistency_refine_v1/launcher.log 2>&1 &
echo $!
~~~

~~~bash
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/consistency_refine_v1 --watch --interval 60
~~~

每组完成REFINE_GROUP_COMPLETE，随后分别对PAIR/CONS输出REFINE_SUMMARY。
状态显示已完成新组数0–4、当前数据集/方法/epoch/batch。全部结束REFINE_COMPLETE。
同命令续跑；旧锁/失败记录改名保留，不删除文件。
强杀后残留RUNNING.lock须先确认进程停止再改名归档，禁止同目录并发启动。
结束同步回data/paper_benchmark_runs/consistency_refine_v1/，至少所有JSON、curves、日志。
checkpoint可以留服务器供续跑和后续评价。
