# 双流 Transformer 修复与知识筛选对照（2026-11-20）

本轮在原工程内新增独立修复入口，保留正在运行的 Transformer 五折、旧源码和历史结果。Cell 表达矩阵、Cell 编码器、CODER/Uni-Mol 缓存、两药对称读出和原始数据划分保持。全部下游参数从头训练，不冻结，不加载旧协同 checkpoint。

## 依据与修复范围

ALMANAC fold1 在 1680–1687 轮出现训练 BCE `0.0724 → 5.5227`，验证 AUPRC `0.5090 → 0.1188`。这是明确的训练失稳，但没有当时的梯度和 Adam 状态，无法认定具体触发层或二阶矩机制。

对旧已选 checkpoint 的 CPU 复现还发现，GO/通路语义投影权重 RMS 已低至约 `1e-14–1e-18`；这些节点真实入射，没有构图遗漏。较好 checkpoint 的 Transformer 增量 RMS 约为输入的 6%–11%，不能把残差过大认定为已证实的崩落根因。证据在 `artifacts/prediction_transformer_diagnosis_20261120/ALMANAC_fold{0,1}.json`。固定 checkpoint 关闭分支属于依赖检查，不等于独立训练消融。

本次稳定性修复是一套固定的保护配方，同时改变优化器、步长与幅度约束，不能从其整体结果归因到单一改动。知识筛选另做同容量对照，不在旧五折中途混改。

## 稳定性保护

- 学习率统一为 `1e-4`，知识分支不再使用两倍学习率。
- 使用 AdamW、`eps=1e-6`、解耦 weight decay `1e-4`。原传播 gate 和零输出矩阵仍不衰减。此改动旨在避免耦合 L2 经 Adam 放大后持续压缩弱监督语义权重；是否改善需要实测。
- 每次整折更新前进行全局 L2 梯度裁剪，上限 `1.0`。非有限 loss/梯度拒绝更新，明确记录失败，不跳过问题批次。
- 每个样本、每个实体 token 的知识残差范数最多为原模态 token 的 **25%**；Transformer 残差最多为同一原模态 token 的 **10%**。两者合计最多 **35%**，并保留初始增量严格为零。用于定标的模态范数停止梯度；两个分支和模态参数本身仍训练。
- 上限只约束残差，不保证模态权重不变、验证性能保底或从此消除所有崩落。25%/10% 是预先固定的保守试验值，不是调参得出的最优值。
- 每轮保存各模块真实梯度 L2、总梯度、裁剪比例。每 100 轮记录残差上限、语义投影 RMS 和传播状态。选中 checkpoint 包含优化器及调度器状态，供后续定位。

## 两组匹配的 ALMANAC seed0/fold0 试验

| 模式 | 稳定性配方 | 知识输入 |
|---|---|---|
| `full` | 相同 | 原全局异构图 |
| `selected` | 相同 | 在全图传播前做下述固定筛选 |

两组保留完全相同的参数名称、数量、初值、模态输入、Cell 边、训练预算和选轮规则。先按完整图构造原模块，再仅替换事实入射缓冲区；缺少证据的原关系 token 保持原参数但 mask 为 false，不补造关系。固定节点编号和全部 CODER 行不变。筛选后超边成员减少及孤立外部节点会在审计中披露。

知识筛选规则：

1. 保留全部原始二元 DTI（target/enzyme/carrier/transporter）及原权重，以 `drug_target` 蛋白为机制锚点。
2. 保留真实靶点–靶点 PPI；中间蛋白须连接至少两个不同靶点，并获至少两个不同药物支持。以跨药物支持数除以蛋白度数平方根排名，每靶点最多 16 个中间蛋白，保留真实靶点–中间蛋白边。不将连接推断成已验证协同机制。
3. GO/通路原集合至多 200 个蛋白；筛后至少保留两个真实蛋白成员，且直接靶点成员获至少两个药物支持。保留原功能实体与成员来源，并记录原始/保留成员数。不声称进行了 GO 语义去重。
4. Disease–Protein 仅保留靶点/桥梁蛋白对应的真实事实；Drug–Disease 各关系和权重保持。
5. 全部 Cell 表达观察、表达值、routes，以及当前 fit 正样本 `Synergy_Hyperedge` 原样保留。筛选排名不读取协同标签、验证/测试标签或性能。

这是**有机制约束的全局知识筛选**，不是针对每个 DrugPair×Cell 的局部检索、RL 或人工机制真值。原 HHNN α/β、关系交互、条件 γ 和末端 Transformer 均保留。筛选缩减噪声候选，也可能删掉有用的较长机制路径，需由匹配内部验证判断。

## 训练、检查与结果

本轮只安排上述两次 ALMANAC fold0 fit，不自动扩展新五折、ONEIL 或 training seed。每组先通过完整真实图 CPU 三步检查，再通过完整真实图 GPU / 1024 监督查询三步检查；检查中的平衡小样本指标不属于效果结果。

正式训练为原划分、完整 3000 轮、无早停、整折 BCE 每轮一次更新，chunk1024 仅限制查询显存。`selected.pt` 仍按 epoch1000–3000 的内部 AUROC 选择，用于与历史同口径比较；额外从首次评估起保存 `best_aupr.pt` 作为诊断，二者不能混记为同一选轮规则。封存测试不读取。

新入口：`dual_branch/repair_cli.py`。运行示例（仅展示；GPU 任务由统一队列启动）：

```bash
python -m dual_branch.repair_cli --device cpu --dataset ALMANAC --fold 0 \
  --stage check --check-rows 64 --knowledge-mode selected \
  --result-directory /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/transformer_repair_20261120/example_cpu_check
```

结果目录：`/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/runs/transformer_repair_fold0_seed0_20261120/ALMANAC/{full,selected}/fold0`。

调度/证据：`artifacts/transformer_repair_20261120`；状态指针：`REPAIR_RUN.json`，旧 `CURRENT_RUN.json` 保留。GPU 仍仅使用原共享 `queue.py` 和 IFS2 `auxiliary_jobs.json`，所有目录合计每卡最多两个本用户 GPU 进程；未筛选正式对照采用更保守的一进程准入。筛选版通过1024查询GPU检查后，实测allocated峰值2151MiB，保留8192MiB放置预算，恢复默认的两进程准入以利用现有第二个槽位；不放宽全局上限，详见 `SELECTED_FIT_ADMISSION.json`。新检查/fit 优先于旧待运行任务，旧正在运行任务不停。日志、checkpoint、缓存和临时文件全部实际写入 IFS2。

首次 CPU 检查在首步后的进度写入处发现重复的 `optimizer_updates` 字段，已修正为 `updates_this_epoch`，旧失败目录和源码保留。以 `cpu_{full,selected}_retry1/CHECKS.json` 为当前正确性凭据。是否开训和是否有提升必须读取实际进度与完整结果，不能从实现完成或短检查通过推断。
