# 样本条件化三路门控：双数据集 seed0 五折

用户在讨论门控融合后要求“先运行一下五折”。本轮采用刚讨论的门控版本，
ALMANAC、ONEIL 各原CV1五折，共10个从头训练fit，默认training seed0，不新增其他seed或网格。
保留此前拼接实现及全部历史结果；拼接修正版只有CPU正确性检查，未自动增加其10次正式对照。

## 相对机制修正版，仅替换最终融合

固定各分支：BRICS2D/UniMol3D片段拼接、原Cell表达MLP、全节点CODER、统一8关系图、
3轮αβ、真实关系组合、单药上下文/最多一个PPI中间蛋白、K10路径GRU、条件γ、作者三层Refiner。
所有原输入/缓存不变，不用旧checkpoint初始化。CODER/UniMol为固定预训练输入；下游全部可训练。

对于Drug A、Drug B、Cell各自的三路输入m/s/k：

```text
模态 m → LN → 模态对齐 ── × w_m ─┐
协同 s → LN → 协同对齐 ── × w_s ─┼→ 加权和256 → Dropout.2/LN
知识 k → LN → 知识对齐 ── × w_k ─┘                ↓
             ↑                        原三token Transformer → 原头
 DrugPair×Cell Query + 三路特征 + 可用性mask
             ↓
  每样本、每实体 masked-softmax 三路权重
```

对齐为独立无bias Linear256→256/GELU，门控MLP隐藏64。Query使用两药模态向量之和、绝对差、Cell，
保证药物交换对称。两药共享门控/对齐，Cell使用独立参数。完全无可用KG候选时知识权重严格0；
这不等于“无两药路径”——仍可存在真实单药或Cell相关知识。权重由监督学习，不是校准置信度。
每100轮输出权重均值、范围、熵及极端权重比例，检查是否只依赖单一路径；不额外添加门控损失。
这是借鉴GMU思想的三源softmax适配，不声称作者原GMU完整复现。

## 训练和选点

3000完整轮、无早停、原BCE；每轮整折梯度累积一次更新、query_chunk1024。
AdamW全部lr1e-4/eps1e-6/wd1e-4，传播gate decay0，全局梯度L2裁剪1。
`selected.pt` 从首次评估按内部验证AUPRC保存，另存 `selected_auc.pt`（全程AUROC）与
`legacy_selected_auc.pt`（1000轮以后AUROC）。前3轮、1000前每5轮、之后每轮评估。
不读外层/封存测试、不用其选参，不将新AP选点与原AUROC主结果当严格同口径消融。

正式入口 `dual_branch/gated_cli.py`；队列桥接 `train_hermes_gated_mechanism_cv5`。
原 `mechanism_cli.py` 仍构造拼接版本，不用于本次门控五折。
结果：`runs/gated_mechanism_cv5_seed0_20261122/{ALMANAC,ONEIL}/fold0..4`。
证据/日志：`artifacts/gated_mechanism_cv5_seed0_20261122`。

## 检查与调度

先4项门控单元、两数据完整图CPU各64监督/64验证/3次更新；然后通过原统一队列做两数据各
完整图/1024监督/最多128验证/3步GPU检查。核验首步全部组件/门控/对齐梯度、缺失源权重0、
Cell条件、药物交换、真实关系组合和路径可达、Refiner及checkpoint重载。CUDA scatter允许
Refiner重复前向1e-5数值误差，CPU仍要求0。检查不是性能实验。
每数据自身GPU检查通过后，根据实测allocated/进程峰值和显式余量校准显存放置预算，才放行fold0；
fold0前三次完整fit更新和梯度/初始化核验通过后放行其余4折，全部由同一个queue.py启动。
所有用户目录合计每卡最多2个训练进程，无旧并发例外；不足即排队，所有新结果/cache/tmp/log/checkpoint在IFS2。
旧`--output`仅为兼容入口，显式保留IFS2 `--runtime-directory`，不另启GPU调度器，不自动恢复旧失败路线。

原机制修正版的事实边界仍有效：DTI覆盖68/87、27/38；桥梁128预算截断121/30对；
共享αβ包含fit-Synergy，知识/协同是读取通道分离，不是统计独立；Refiner为CODER+Synergy适配，
没有恢复完整原HERMES输入/辅助目标；所有路径/成员注释均不等于已验证的协同因果机制。
