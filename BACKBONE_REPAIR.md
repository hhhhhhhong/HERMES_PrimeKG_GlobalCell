# ALMANAC fold0：主干修复与知识增量对照

2026-11-23 用户要求修复训练慢、效果一般的问题。只开展 ALMANAC seed0/fold0 两组探索，不扩展新五折。此前运行和结果保留。

## 修复内容

模态编码（原有 BRICS 2D + UniMol 3D、原 HERMES Cell 表达编码）直接进入作者三层 Refiner。Refiner 只读取当前 fit 正样本构成的 Synergy 超图，保留 20% 整边 Dropout。CODER 只用于知识分支。协同特征仍参与 γ 的 Query，末端保留来源门控、Drug/Cell 适配器、单层 Transformer、对称预测头。

取消逐 batch 目标协同边屏蔽，使用固定 fit 图。每轮生成一次模态表示和图表，每个 query chunk 对分离的叶子累计梯度，再一次反传到原图及模态编码器；每轮一次 AdamW 更新。Refiner 每个 BN 每轮只更新一次。知识路径按 512 条分块编码并进行激活重计算，降低全候选池训练的激活内存。知识图前向的随机流隔离，避免它额外消耗模态/Refiner 的随机流。

## 两组

- `none`：不计算知识分支，知识来源显式 mask；保留完整参数结构和融合接口。
- `real`：使用原 PrimeKG 知识分支和 Cell 条件路径。

两组全部参数从头初始化，同 seed0、同完整初始参数 SHA、同数据划分和训练配方。保留相同已安装容量，但 none 的知识参数没有梯度，因此不是相同有效容量的随机知识对照。对照检验修复模型中加入整个知识分支的净效果，不能证明具体生物机制、不能把历史模型差异归于单个修复。

优化器 AdamW，lr=1e-4，eps=1e-6，原参数组 weight_decay=1e-4/gates=0，梯度裁剪1，指数调度0.9997。最多3000轮，至少300轮；连续100轮没有验证 AUROC 改善后早停。第1/2/3轮及随后每5轮验证，主 checkpoint 为 `selected_auc.pt`，同时记录该点 AUPRC。Patience 按训练轮数计数。所有模型和模态下游参数可训练，不加载旧协同 checkpoint。

## 协议与限制

构图只用 fit 标签；验证/测试标签不进入图。恢复原固定训练图协议，也保留原划分中的重复查询问题（历史 TARGET_OVERLAP_AUDIT.json）。不宣称严格未见组合泛化。封存测试不参与。本方案不是原 HERMES 完整复现，也不能保证提升。早停解决无效训练，不能提升既有验证峰值。

检查包括：真实全图、3次更新；共享图梯度与未分块损失等价；Refiner 到药物和 Cell 模态的独立梯度；BN 更新次数；无知识组知识梯度/权重为零；药物交换/读出分块一致；checkpoint 重载；两组初值一致。CPU检查通过后在统一队列提交两个2048监督行GPU检查，两者均通过才启动两组正式探索。

实现：`dual_branch/backbone_repair.py`、`backbone_repair_training.py`、`backbone_repair_cli.py`。
结果：`runs/backbone_repair_fold0_20261123/{none,real}`。
检查与冻结证据：`artifacts/backbone_repair_fold0_20261123`。

命令（CPU检查示例，GPU训练只能经统一 queue）：

```bash
CUDA_VISIBLE_DEVICES='' /ifs2/home/liushuhong/anaconda3/envs/py38/bin/python -B -m dual_branch.backbone_repair_cli --knowledge real --stage check --device cpu --result-directory /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/backbone_repair_fold0_20261123/new_cpu_check
```

只读查看 `progress.json` 和 `selected_auc.json`。检查指标不是性能结果；两组内部验证用于探索，不能解释成完整五折提升。
