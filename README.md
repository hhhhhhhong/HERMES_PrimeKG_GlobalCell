# HERMES_PrimeKG_GlobalCell

最新代码：**2026-11-23 主干修复版**。当前范围为 ALMANAC seed0/fold0 的无知识／真实知识两组探索，尚未确认性能提升。此前版本与五折入口继续保留，不能混用各版本的训练结果。

## 当前信息流

```text
Drug：BRICS 2D + UniMol 3D ─┐
Cell：原 HERMES 表达编码 ──┴→ 模态表示 → 作者 Refiner → 协同表示
                                              ↑
                                  当前 fit 正例 Synergy 超图

CODER → PrimeKG 统一异构图 → 三轮 α/β → 真实共享实体关系交互
                                      + Cell 条件化路径编码
                                              ↓
                          模态与协同共同构造 Query → 条件化 γ
                                              ↓
               模态 / 协同 / 知识 → 来源门控融合 → 三个实体 token
                                              ↓
               Drug 共享、Cell 独立适配器 → 单层 Transformer
                                              ↓
                                    对称协同预测头
```

修复版让分子与表达表示直接经过协同 Refiner；CODER 只用于知识分支。固定训练图取消逐 batch 目标边屏蔽，每轮共享一次图传播与一次优化器更新，Refiner 的 BatchNorm 每轮更新一次。保留 20% 协同超边 Dropout。`none` 不计算知识分支并屏蔽知识来源，`real` 使用现有真实知识。

两组安装参数及初值一致，但无知识组的知识参数不参与优化，因此不是相同有效容量的随机知识对照。当前模型复用作者 Refiner，并不等同于完整原 HERMES 复现。

## 训练与检查

- ALMANAC fold0、seed0、全参数从头初始化，不加载旧协同 checkpoint。
- AdamW，lr=1e-4、eps=1e-6，梯度裁剪1，原参数组衰减设置保留。
- 最多3000轮，至少300轮；100轮无验证 AUROC 改善后早停。第1/2/3轮及此后每5轮验证。
- 从首次评估开始保存最佳 `selected_auc.pt`，同时记录该点 AUPRC。
- CPU和GPU真实图正确性检查已通过；单元检查覆盖共享图梯度累计等价性和早停逻辑。正确性检查不代表性能提升。
- 训练图只使用 fit 标签；原划分中的重复查询限制仍然存在，不宣称严格未见组合泛化。封存测试不参与。

完整说明：[BACKBONE_REPAIR.md](BACKBONE_REPAIR.md)。

## 主要入口

| 文件 | 用途 |
|---|---|
| `dual_branch/backbone_repair.py` | 修复模型：模态输入 Refiner、固定图、知识开关 |
| `dual_branch/backbone_repair_training.py` | 每轮共享传播、梯度累计、AUROC早停 |
| `dual_branch/backbone_repair_cli.py` | ALMANAC fold0 两组检查与训练 |
| `scripts/enqueue_backbone_repair.py` | 服务器统一队列提交 |
| `dual_branch/entity_token_adapter.py` | Drug/Cell 输入适配器 |
| `dual_branch/hypothesis_evidence.py` | 协同条件 γ 与此前精确屏蔽版本 |
| `dual_branch/cell_conditioned_model.py` | Cell 条件化路径选择 |
| `global_cell/` | 构图、α/β、关系交互及基础注意力 |
| `tests/test_backbone_repair.py` | 梯度累计和早停检查 |

此前正式五折适配版见 [ENTITY_TOKEN_ADAPTER_CV5.md](ENTITY_TOKEN_ADAPTER_CV5.md)，它仍采用逐 batch 目标边屏蔽；不能把其结果归给本次修复版。其他阶段见 [HYPOTHESIS_EVIDENCE_CV5.md](HYPOTHESIS_EVIDENCE_CV5.md)、[GATED_MECHANISM_CV5.md](GATED_MECHANISM_CV5.md)、[DUAL_BRANCH_GEMINI_V2.md](DUAL_BRANCH_GEMINI_V2.md)。

## 数据、依赖与部署范围

本仓库是源码同步快照，包含代码、测试、方法文档和依赖源码；不包含数据集、PrimeKG原始数据、表达矩阵、实体向量、预训练权重、checkpoint、日志、运行状态或成绩文件。`ENVIRONMENT.json` 记录服务器环境。

部分脚本依赖原服务器绝对路径、审计manifest和已生成缓存，尚未提供跨机器一键复现。外部分子编码器源码位于 `integration/HERMES_UniMol_StructureResidual/`，迁移时需放置到预期的相邻目录或适配路径。统一队列副本位于 `integration/mcdsp/`，依赖服务器注册表，不能作为独立调度器直接运行。

服务器继续使用统一GPU队列；当前本会话授权每卡最多3个该用户进程，显存或槽位不足时排队。源码同步不改变训练进程。新输出均保存IFS2。

## 来源与许可

HERMES作者代码固定来源与哈希见 `vendor/hermes/PROVENANCE.json`；HYPER辅助函数来源见 `vendor/HYPER_PROVENANCE.json`，原许可保留。PrimeKG提供知识关系，表达派生连接不解释为已证实的生物机制。作者源码不作改写，未为全部第三方代码重新声明统一许可证。

`SOURCE_MANIFEST.json` 记录每个复制文件的来源和SHA256，完整覆盖修复版冻结的132份源码。历史实现保留用于追溯。
