# GlobalCell：新增第15类协同超边

本目录已按用户要求直接更新为第15类协同超边版本。**保留同一张全局异构超图中的14类知识/表达关系，新增 `Synergy_Hyperedge {Drug A, Drug B, Cell}`，不另建HERMES分支。** 变更前源码已归档，旧结果、checkpoint和停止记录保留。

```text
Drug：原HERMES分子图编码256 ──┐
Cell：原HERMES表达MLP256 ─────┼→ 一张全局异构超图
其他实体：固定CODER768→256 ──┘   原14类关系 + 当前训练折正协同三元超边
                                      ↓
                                 HHNN α / β
                                      ↓
                        真实共享实体处的HYPER式关系交互
                                      ↓
                         Drug A / Drug B / Cell 条件化γ
                                      ↓
                         自身256 + 门控 × 投影(知识256)
                                      ↓
                          原式HERMES预测头双向平均 → 协同概率
```

## 第15类边如何构建

- 入口 `global_cell/synergy.py:add_training_synergy` **只接受当前折的fit行**，不接受valid/test数据。先核验原始训练行SHA，再选择标签为1的训练行。
- 每个正训练样本构成一条三元超边，成员是全局DrugA、DrugB、Cell节点，角色位置为 `[0,1,2]`。Cell ID已经包含Drug数量偏移，不重复偏移。药物按ID排序，保持与当前预测端一致的交换对称处理。
- 权重为1，对应原HERMES协同边的单位权重；不改动原14类关系的成员、权重、表达值或来源。
- 与作者实现一致，重复的正训练观测保留为独立超边，另外记录唯一三元组数和重复数。药物交换后的重复也在审计中披露。
- 各折在加载自身fit之后、创建模型之前构图。基础图和CODER输入只读复用，**不把任何一折的协同图保存为所有折共享的缓存**。
- α聚合三个成员；β保留不同关系和位置通道。协同边可在真实共享的Drug节点与DTI等关系组合，也可在共享Cell节点与表达关系组合。γ仍读取查询相关的关系/组合token。

## 明确保留与限制

本次只增加协同关系，不增加原Refiner、辅助重构损失、残差分支、RL、路径选择或其他新模块。**恢复协同超边不等于逐字恢复原HERMES Refiner**：实际传播仍然是当前HHNN α/β和关系交互；是否恢复协同过滤效果需要后续匹配实验验证。

Drug/Cell编码器、CODER投影、注意力、融合层和预测头全部从头初始化、可训练；固定CODER向量只是离线输入。训练入口保留seed0、原CV1五折、3000轮、BCE与原学习率配方；本轮**不提交/启动正式五折，也不恢复旧版本**。`--stage check --device cpu`仅在完整图上对128行监督做两步更新，用于正确性检查，不作为性能结论。

原CV1划分存在药物交换后相同查询跨fit/valid出现。审计只读取三列实体ID统计重叠，不读取valid标签选边，不改历史划分。后续同折比较应披露该协议限制，不能称为查询完全隔离的评估。

## 文件与检查

- `global_cell/synergy.py`：训练折协同边构建与来源审计。
- `global_cell/cli.py`：加载当前折后加入第15类边，记录 `SYNERGY_AUDIT.json`。后续训练必须使用新的输出子目录和本次源码清单，不能复用旧14关系版本的结果目录、入口标记或冻结清单。
- `global_cell/network.py`：原统一图模型；增加协同关系专属梯度诊断。
- `tests/test_synergy.py`：仅正训练样本构边、折隔离、原关系不变、错误SHA/偏移拒绝、协同通道Drug/Cell梯度与传播。
- `scripts/audit_synergy.py`：两个数据集全部十折构图审计；不训练。
- `artifacts/synergy15/SYNERGY_FIVE_FOLD_AUDIT.json`：实际每折边数、训练行SHA和既有查询重叠。

所有新增文件、日志、检查checkpoint和临时目录均位于 `/ifs2`。基础图、划分、CODER和作者封存源码沿用本目录原有文件，不写入旧缓存。未来如获准GPU训练，仍须走现有统一队列和全用户每卡最多两个进程约束；本轮不使用旧并发例外。

## 已完成的验证

迁回前快照的8项机制测试、两数据集共十折的构图审计均已通过。ALMANAC fold0新增6,515条协同边，ONEIL fold0新增1,201条。迁回前快照的两数据集fold0完整全局图CPU检查均通过：128行监督、两次更新、协同关系各参数实际变化、全部组件非零梯度、药物交换/Batch一致性与严格checkpoint重载。此处是正确性检查，不是性能评估。

汇总见 `artifacts/synergy15/VALIDATION_SUMMARY.json`；完整来源和检查记录在 `artifacts/synergy15/cpu_check_{ALMANAC,ONEIL}`。原14关系源码已备份至 `artifacts/before_synergy15_inplace_*`，历史训练保持停止；未新增任何GPU或正式五折任务。

迁回原目录后，保留了原目录已有的 `attention.py` 修改（表达FiLM、归一化和角色内γ读取），没有用归档快照覆盖。原目录14项测试与作者导入/入口路径检查通过，记录见 `artifacts/synergy15/INPLACE_VERIFIED.json`。上述两步CPU结果属于迁回前源码快照，不能当作当前已修改注意力版本的完整前后向验证或效果结果。

## 当前正式训练：用户19:38修改版

用户要求运行当前代码，随后说明已自行修改。新一轮完整保留当前attention.py/network.py：表达边使用有界表达值及其全局标准化值作为双输入FiLM；γ采用对称三元上下文、自身Query和可学习对称角色路由偏置；Drug/Cell分别用初始0.05门控残差接入，原预测头双向平均。未加载旧checkpoint，所有下游参数仍从头可训练。图含原14类关系和当前折正fit协同超边。此前“只加协同关系”和迁入前CPU检查描述的是旧快照，不能代替当前版本验证。

两数据集各seed0、原CV1五折、3000完整轮、内部验证epoch1000–3000选轮，BCE/Adam/截断指数调度不变。两套当前源码真实GPU两步检查通过后，统一队列运行正式fold0；核验前两次完整fit更新后放行其余折。新任务GPU2/0/1按全用户每卡2进程调度，旧实验继续；不借用旧GPU3例外。

- 最新入口与状态：`CURRENT_RUN.json`。
- 新结果：`runs/global_cell_synergy15_cv5_seed0_20261117/{ALMANAC,ONEIL}/fold0..4`，`STATUS.json`、`RESULTS.md`。
- 本次完整源码与输入SHA：`artifacts/synergy15_cv5_seed0_20261117/frozen`。
- GPU检查：同调度目录`gpu_checks/{ALMANAC,ONEIL}`；CPU测试日志`unit_tests.log`。
- 原14关系结果和STOPPED.json继续保存；与本次新结果分开。外层/封存测试不参与。

## 2026-11-18：仅重跑当前ALMANAC fold0

用户要求先单独运行包含协同角色 `[0,0,2]` 与20%协同边消息Dropout的当前代码。attention.py/network.py/synergy.py未改写；从头初始化、全下游训练，固定CODER输入、3000轮和原优化配方保持。仅训练ALMANAC seed0/fold0，封存测试不参与。

当前实现先计算完整α/β权重，再对协同边消息施加训练期20%整边随机遮罩和1/0.8缩放；eval不采样。4项当前机制测试通过，GPU完整图两步检查通过后才放行一个正式fit。

本轮入口 `scripts/launch_synergy002_fold0.py`，仅通过原统一队列调度；CPU报告 `scripts/report_synergy002_fold0.py` 不调度GPU、不推广其他折。输出 `runs/almanac_fold0_synergy002_drop20_seed0_20261118`；源码、输入、计划与GPU检查在 `artifacts/almanac_fold0_synergy002_drop20_seed0_20261118`。旧六折结果保留，旧剩余四折搁置；新旧版本分开报告。
