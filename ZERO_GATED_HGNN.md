# 三层零门控真实超图传播（ALMANAC seed0/fold0）

当前单层 `[0,0,2]` + 20% 协同消息 Dropout 已完成3000轮。
内部验证AUROC选定第2986轮：AUROC0.832360、AUPRC0.502258。
原HERMES同折为0.853738、0.543928，因此满足用户此前“仍不足时再进行第三步”的条件。
本轮仅补一个ALMANAC seed0/fold0，不扩展ONEIL、五折、种子或参数网格。

## 结构

原输入/图/关系角色/损失均沿用；新增真实αβ传播轮次和逐层零门控：

```
H0：全体Drug、Cell、PrimeKG节点的初始表示
for l in [1, 2, 3]:
    U_l = AlphaBeta_l(H_previous, actual_hyperedges)  # 关系/角色槽位
    G_l = tanh(g_l) * U_l                           # g_l=0
    slot_l = H_previous[slot_nodes] + G_l
    H_next = H_previous + scatter_mean(G_l, slot_nodes)
final_slot → 原RelationInteraction → 原条件γ → 原门控读出 → 原HERMES式头
```

每层都有独立αβ参数及256维零门控。节点反馈平均仅限该节点真实入射关系/角色槽位，
不是全局均值池化，不构建相似图。最终关系/角色槽位保留给原HYPER式交互，未新增Attention。
所有实际中间蛋白、功能锚点和Drug/Cell都参加下一轮更新，不能把同一轮重复执行称为多跳。
Synergy Dropout仍在各αβ轮的β之后施加，各轮独立采样；原其它关系不变。

每层g=0时节点增量严格为0，且恒等支路没有LayerNorm、投影或激活。
无入射关系的节点增量严格为0。消息分支正常初始化，避免门控和分支同时置零导致无法学习。
首步先由门控获得监督梯度；首步传播参数的监督梯度为0，门控更新后传播参数才获得非零梯度。
原Adam带weight_decay，因此零监督梯度不表示这些参数首步完全不发生任何变化。

这不是作者HypergraphConv源码的逐字复现，是用户指定的现有αβ结构上的门控迭代。
原末端读出gate保持0.05，未同时改动；新传播门控才是0。
零门控保证各传播层初始化恒等，不保证整个新模型初始预测等于旧单层模型：
新结构的初始槽位包含节点恒等支路，旧结构槽位直接是单轮聚合输出。
不宣称门控只会在验证有利时打开，门控仅用训练梯度学习，验证只选checkpoint。

## 训练与证据

原单层global_cell源码与历史结果保留且SHA核验；实现使用同工程models子目录下的独立子类。
新trainer是原cli隔离副本，仅适配模型、三步检查、零门控首步梯度规则和诊断记录。
原3000轮、1000–3000内部验证AUROC选轮、Adam/BCE/调度/query_chunk1024/整折一次更新不变。
全部下游参数从头可训练，固定CODER只作输入，不加载旧checkpoint，不读取封存测试。

机制测试覆盖逐层初始RMS=0、共享参数初值一致、真实上一层节点反馈、外部节点与孤立节点、
门控先学习再传播、Query分块/药物交换、严格checkpoint恢复。
真实完整ALMANAC图检查先使用128条监督做3步检查，不把其指标当正式结果。
零门控时本就不应该有远端传播，因此图干预检查临时将门控开到0.05验证可达性，
然后恢复精确零值再开始3步优化；初始零增量审查独立保存。

CPU完整图检查通过后，按统一auxiliary_jobs.json排一个GPU检查和一个正式fit。
三轮保留计算图显存更大，GPU检查预留26,000MiB、训练32,000MiB；空闲不足则排队，
不绕过每卡全用户最多2GPU进程的限制，不缩图、减batch或擅自改变配方。

结果 `runs/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118`；
源码/输入冻结和检查 `artifacts/zero_gated_hgnn3_ALMANAC_fold0_seed0_20261118`。
报告器只用CPU读取状态，不增加实验。SimilarityRefiner继续撤销，旧路线不恢复。

## 最新授权：两个数据集各五折

2026-11-18用户明确“直接开始五折”，随后确认 ALMANAC、ONEIL 各五折，
覆盖此前只允许ALMANAC单折的范围限制。本次固定两数据集各seed0、fold0–4，
总计10个正式fit，每折3000轮，1000–3000内部AUROC选轮；不新增seed或参数网格。

`scripts/train_zero_gated_hgnn_cv5.py`放开数据集/折限制，模型、损失和正式训练配方保持。
`models/zero_gated_checkpoint.py`仅给αβ和RelationInteraction加激活重计算以降低显存，
不增加模型参数或改变公式；重计算保留随机状态，αβ内部没有运行统计层。
含真实Synergy Dropout的3步同初值对照验证损失、全部梯度、参数更新、随机流和验证预测一致。
GPU检查改为1024条监督、1024条验证及query_chunk1024，匹配正式训练的查询峰值；
检查本身的分数不作性能结论，正式fit不载入检查checkpoint。

10折原始训练/验证数组SHA与原HERMES划分逐项一致，每折仅用自身正训练标签构协同边。
ALMANAC复用已通过的完整图CPU机制检查；ONEIL新增真实完整图3步CPU检查并通过。
旧ALMANAC单折的GPU任务尚未启动，已移出活动队列归档，以免重复fold0；所有CPU记录保留。

两套GPU检查均通过后放行两个fold0；各数据集fold0两次完整fit更新及
“首步gate有梯度、传播监督梯度为0，第二步传播梯度非零”核验后，才放行其余4折。
GPU检查预留11000MiB；正式fit先保守预留32000MiB，CPU观察器根据实际GPU检查峰值，
按`ceil512(1.15*peak+1024)`（最低6144MiB）校准本套件放置预算；
仅改变共享队列的显存估计，不改变模型、训练预算或每卡全用户最多2进程的约束。
观察器不直接启动GPU进程，仍由原统一queue.py调度，所有新输出、日志、临时目录和缓存IFS2。

新结果：`runs/zero_gated_hgnn3_cv5_seed0_20261118/{ALMANAC,ONEIL}/fold0–4`。
新冻结、划分、CPU/GPU检查与调度：`artifacts/zero_gated_hgnn3_cv5_seed0_20261118`。
CPU观察器`scripts/report_zero_gated_cv5.py`核验每折3000次更新、源码/划分/选定checkpoint SHA；
各数据集完整五折通过后才汇总内部验证均值与总体标准差，封存测试始终不读取。
