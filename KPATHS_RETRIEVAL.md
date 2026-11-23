# PrimeKG 的 K-Paths 式离线路径检索

本次完成检索实现与真实图审计，尚未接入预测模型，也没有新增 GPU 训练。当前修复版和原五折继续按各自冻结版本运行。全部新增代码、索引和报告位于当前工程与 IFS2，不恢复已停止的 PathBatch。

## 参考论文与适配范围

[K-Paths（KDD 2025）](https://arxiv.org/html/2502.13344v3) 采用无需训练的路径检索：用 Yen 算法找无环短路径，再按有序关系序列去重。论文实验默认最多 10 条、最长 3 跳，研究药物相互作用/药物重定位，并非 Drug–Drug–Cell 协同预测。其 GNN 可以读取检索子图，不要求先运行 LLM。

[作者代码](https://github.com/rsinghlab/K-Paths) 的本地审计版本为 `bf25d24eac6939574542d8994957b340ea1f5ab8`，来源文件沿用旧工程内只读封存的参考副本，未恢复其训练。`artifacts/kpaths_reference_review_20261120/CODE_CHECKS.json` 保存作者函数的三个小图核对结果。论文 Table 9 的 G–participates–PW 在三个基准均未被保留，不能据此声称它的结果已验证我们需要的 Reactome 路径。

当前实现有明确的任务适配，不能称为作者实验的完整复现：

| 项目 | reference 模式 | adapted 模式 |
| --- | --- | --- |
| 基本检索 | Yen 无环短路径，最多 3 跳 | 相同检索，加受约束的 4 跳功能路径 |
| 同一有序关系序列 | 最多保留 1 条 | 最多保留 2 个不同实体实例 |
| 最终输出 | 最多 10 条 | 最多 10 条，按证据类别轮转读取候选 |
| 功能路径 | 不额外补充 | Drug–Target–GO/Reactome–Target–Drug |
| 排序 | 跳数优先 | 类别内按跳数及中间节点度数惩罚排序 |

四跳补充只用真实 `drug_target` 和功能成员事实，原始功能集合超过 200 个蛋白的锚点不进入该补充步骤。这个阈值是预先指定的工程约束，尚未经协同性能验证。adapted 的基础三跳搜索最多收集 40 条关系去重后的候选；GO/Reactome 四跳候选堆各保留 64 条，再参与最终选择。保留更多实体实例可以避免不同蛋白/通路被相同关系模板完全折叠，但不保证这些实例均与当前 Cell 有关。

## 输入、方向和边界

索引从 `HERMES_PrimeKG_HHNN/data/raw/{nodes.csv,edges.csv,drug_mapping.json}` 读取原始 PrimeKG。允许的 PPI、GO/Reactome、Disease–Protein 使用完整原始事实池，不再限制为旧全局图内的蛋白集合；药物相关边仅保留两个数据集的映射药物。流式读取 8,100,498 行后，允许关系去重得到 **745,886 条事实**，这不是 PrimeKG 全部关系的边数。

每条输出路径保留实体 ID、关系序列、端点角色、原始事实 ID 和 CSV 行号（不含表头，从 0 开始）。连通性用简单图检索，并行的 target/enzyme 等事实单独保存和展开，避免被覆盖。反向遍历显式标记 `^-1`；PPI 按对称互作处理，不赋予因果调控方向。药物端点按 PrimeKG ID 排序，交换输入药物得到同一候选集合。

检索不读协同样本或标签、表达矩阵、embedding、checkpoint 或封存测试。不允许协同三元边、直接 Drug–Drug 边或第三种药物构成检索捷径。不存在连接路径时，单侧邻居单独写入 `fallback`，不会计为连接路径；两个端点映射成同一实体时不伪造路径。

每次基础搜索最多考察 128 条实体路径，时间预算为 2 秒。时间仅在 Yen 每次产生候选之后检查，并非能中断单次搜索的硬超时。补充四跳另有 2 秒软预算。触及枚举/时间预算会写入 `censored=true`；这种结果不能宣称是穷尽的全局 Top-K。度数惩罚、成员数阈值及同模板上限也都属于启发式，不代表生物学质量已被证明。

## 2026-11-20 真实图审计

从每个数据集的全部可能无序药物对中独立随机抽取 **16 对**（抽样随机种子 0），不读取实际协同样本或标签。因此下表是这 32 对的结果，不是完整数据集覆盖率。

| 抽样指标 | ALMANAC | ONEIL |
| --- | ---: | ---: |
| 药物映射（全体药物） | 85/87 | 32/38 |
| reference 找到连接的药物对 | 14/16 | 10/16 |
| adapted 找到连接的药物对 | 14/16 | 11/16 |
| adapted 有双端靶点支持候选的药物对 | 5/16 | 5/16 |
| adapted 有 GO/Reactome 路径的药物对 | 4/16 | 5/16 |
| adapted 返回的四跳路径条数 | 13 | 16 |
| adapted 发生搜索截断的药物对 | 0/16 | 1/16 |

“双端靶点支持”只表示两端都来自原始 `drug_target` 事实、且路径不经过疾病，并不表示已证实的协同机制。映射使用已有严格映射文件，没有在本轮猜药物别名或新增映射。

本机已构建内存索引后的单对平均检索时间：reference 约 ALMANAC 20 ms / ONEIL 7 ms；adapted 约 19 ms / 27 ms。计时很短、有缓存和样本差异，不能据此比较两个算法谁更快；原始图读取、索引构建和落盘另计，整个抽样审计约 33 秒。ONEIL 的 ERLOTINIB–ETOPOSIDE 在 adapted 基础检索触及 128 条实体路径预算，已明确标记截断。

**值得警惕的是：短路径仍容易被疾病背景和代谢酶占据。** adapted 的 ALMANAC 103 条路径中，76 条经过疾病节点；ONEIL 77 条中为 39 条。另有 11/20 条来自其他蛋白关联，例如共享代谢酶，不能与双端药物靶点混为一种证据。

真实例子：

- `Erlotinib → liver disease ← Dasatinib`：两边都是禁忌证事实，反映疾病背景，不能解释肿瘤细胞中的协同杀伤。
- `Gefitinib → CYP2D6 ← Hydroxyurea`：两边都是 enzyme 关系，有药理背景意义，但不是共同治疗靶点证据。
- `Mitotane → ESR1 → Reactome 功能集合 ← KIT ← Sunitinib`：四条原始事实都存在，可以作为功能联系候选，仍需 Cell 条件化选择及性能验证。
- `Erlotinib → EGFR → neuron projection morphogenesis ← WEE1 ← MK-1775`：即使两端都是 target 且 GO 成员关系真实，也不能据此认定其对当前细胞系有作用。

`QUALITY.json` 将路径分成 `target_anchored_candidate`、`other_protein_association_context`、`disease_association_context`，并保存具名例子。所有输出路径和 fallback 涉及的 **452 个不同原始事实行**已重新读取原始 CSV 独立核验。6 项单元检查与真实路径的长度、去重和来源检查通过；这些是正确性证据，不是模型效果证据。

## 与当前模型连接的原则

下一步模型适配应保留全局 Drug–Drug–Cell 协同超边和 Cell 表达关系，用药物对检索得到的路径作为外部候选证据，再由当前 Cell 条件化读取。主机制候选与疾病/代谢背景应分开输入和统计，避免“两个药都关联常见疾病”占据主机制预算。空证据用显式掩码，不造假路径。

当前代码还没有完成这项模型适配，也没有新的路径版训练结果。不能把路径候选直接合并后称为真实多元相互作用超边；每条链首先是多条二元事实支持的关系组合证据，功能集合的多成员结构须继续保留原始成员来源。后续应以相同训练预算验证候选筛选是否带来增量，不把“有路径”当作“有用”。

## 文件与复现

- `dual_branch/kpaths_retrieval.py`：并行事实索引、reference/adapted 检索。
- `tests/test_kpaths_retrieval.py`：并行关系、方向、四跳、实体变体、交换、非法捷径与截断检查。
- `scripts/audit_kpaths_retrieval.py`：CPU、无标签的原始 PrimeKG 抽样审计。
- `scripts/report_kpaths_quality.py`：证据类别统计与原始行独立复核。
- `artifacts/kpaths_reference_review_20261120/raw_primekg_audit/{INDEX,AUDIT,QUALITY}.json`：原始 SHA、统计与核验。

在当前工程目录运行，`--out` 必须为新的 IFS2 输出目录，不覆盖既有审计：

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009/tmp \
/ifs2/home/liushuhong/anaconda3/envs/py38/bin/python -B -u -m scripts.audit_kpaths_retrieval \
  --pairs 16 --candidate-budget 128 --seconds-budget 2 \
  --out /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/kpaths_new_audit

PYTHONDONTWRITEBYTECODE=1 \
/ifs2/home/liushuhong/anaconda3/envs/py38/bin/python -B -u -m scripts.report_kpaths_quality \
  --audit /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/kpaths_new_audit
```
