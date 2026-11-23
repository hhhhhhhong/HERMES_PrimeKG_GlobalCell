# 多模态机制修正版（2026-11-22）

本轮对应先前代码审查的四项结构/训练局限，以及药物 2D/3D 简单相加。
所有实现仍在本工程；旧五折、旧模型和此前拼接版均保留。新模型全部下游从头可训练，
不加载旧协同 checkpoint；固定 CODER 与 Uni-Mol 缓存不微调，Cell 矩阵/MLP 不改。

## 修改后的实际结构

```text
统一图：DTI / PPI / BP / CC / MF / Reactome / Cell-expression / fit-Synergy
                              ↑
                     所有节点 CODER → 类型投影
                              │
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
三轮 α/β → 真实共享实体 RelationInteraction      作者原码三层 Refiner
        │                                      同一图的 fit-Synergy 子关系
        ├→ 全部根节点生物关系/组合                   │
        └→ 有序路径 GRU                             │
                   ↓                                │
          DrugPair × Cell 条件 γ                     │
                   ↓                                ↓
                 知识流                           协同流
                   │                                │
BRICS2D || coverage×UniMol3D → 拼接投影 → Drug A/B    │
Cell 表达 → 原 MLP                    → Cell         │
                   ↓                                │
                 模态流                             │
                   └──── 三路分别LN → 拼接768 → 投影256 ┘
                                      ↓
                        3-token / 4-head Transformer
                                      ↓
                     原 Consolidator 结构、双药对称预测
```

三路最终融合均通过可训练投影参与预测，没有知识残差注入、零输出初始化或 25%/10% 外部限幅。
Drug A/B 共享融合参数，Cell 使用独立融合参数。Transformer 和编码器内部的正常跳连保留。

## 逐项修正

1. **协同信号独立读取。** 直接调用封存作者 `HERMES.model.Refiner`，保持其三层
   HypergraphConv、门控残差和最终计算公式。它读取统一图中的当前 fit 正例 Synergy
   incidences，输出不经过知识 γ，独立进入最终融合。
   适配点为根节点 CODER 投影输入、仅 Synergy 子关系和训练期 20% 整边 Dropout；
   Dropout 真正删除 incidence 并重编号。它不是完整原 HERMES 的多关系/重构目标复现，
   也没有加载原预测头权重。Synergy 只由当前 fit 标签构造，valid/test 标签不造边。

2. **接通真实关系组合。** 读取 RelationInteraction 输出的所有根节点生物关系单通道
   及组合通道，包括有真实共享蛋白证明的 DTI→PPI、DTI→Pathway、表达→PPI 等。
   顺序/实体位置描述和原始路由证明保留。协同关系移到独立通道；外部证据 γ
   仍读取根节点相关渠道和当前药物对的 K 条路径，不跨样本混读。
   路径 GRU 仍编码 α/β 后节点；显式关系组合通过独立真实输出通道进入 γ。

3. **扩大有根据的连接与单药上下文。** 在旧候选上增加
   `Drug–Target–中间蛋白–Target–Drug` 四跳真实 PPI 路径，桥梁不必是药物直接靶点。
   每模板最多两个实体实例、最终 K≤10；每对最多检查按 PPI 度排序的 128 个共同桥梁，
   截断明确记录。保留全部真实直接 DTI，并给每个靶点每个功能家族读取最多四个
   按原集合规模排序的注释（原成员集合≤200）。无连接路径仍能读取单药真实关系，
   不把单边事实伪装成两药连接。GO/Reactome 超边只包含实际成员；单成员子集明确标为二元注释。

4. **保存早期峰值。** 新独立训练入口的 `selected.pt` 从第一次评估按内部验证 AUPRC 保存，
   `selected_auc.pt` 独立保存全程 AUROC，`legacy_selected_auc.pt` 单独保留 epoch≥1000
   AUROC 诊断。第1/2/3轮评估，1000轮前每5轮、之后每轮评估，仍固定3000轮无早停。
   选择规则变化必须与旧结果区分；不能将新 AP 选点与旧 AUROC 选点当作严格的单项消融。

5. **2D/3D 内部也使用拼接。** 复用原 BRICS 编码与固定 Uni-Mol 投影，
   每个片段用 `[2D128 || coverage×3D128] → 可学习128`，再按原片段汇聚/计数得到 Drug256。
   缺失 3D 在投影后严格乘零。原 BRICS 图传播内部残差不改。

## 数据准备与已知边界

准备入口 `scripts/prepare_mechanism_repair.py` 只读原始事实索引、药物身份和原无标签图，
不读协同样本/标签，也不改表达。缓存为 `data/mechanism_repair_20261122`。
原始事实索引 SHA 和每条新边/路径的 source row、端点和关系均留档。

| 数据集 | 可能药物对 | 旧有连接 | 新有连接 | 保留新桥梁路径 | 桥梁预算截断药物对 |
|---|---:|---:|---:|---:|---:|
| ALMANAC | 3741 | 1252 | 2094 | 4011 | 121 |
| ONEIL | 703 | 245 | 351 | 699 | 30 |

这些是所有可能无序药物对，不是按标签抽样的覆盖。实际 fit/valid 行覆盖另存 `COVERAGE.json`。
两数据分别仍只有68/87、27/38药物有真实 DTI；缺失事实没有补造，候选不保证穷尽所有机制。
新增外部节点为0，旧全部CODER数值精确复用，没有新下载或重编码。Cell表达边仍35310/20544。
图中功能集合/连接路径不是已验证的协同因果机制，覆盖增加也不代表预测指标增加。
Cell表达在模态与图中重复出现是既有设计，不应计为新增外部知识。

## 检查与正式入口

单元测试：`python -m unittest discover -s tests -p 'test_mechanism_*.py' -v`。
缓存完成后用 `scripts/freeze_mechanism_repair.py` 冻结新旧依赖及输入 SHA。

CPU 真实全图检查示例：

```bash
cd /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell
PYTHONDONTWRITEBYTECODE=1 /ifs2/home/liushuhong/anaconda3/envs/py38/bin/python \
  -m dual_branch.mechanism_cli --dataset ALMANAC --fold 0 --stage check --device cpu \
  --result-directory /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/mechanism_repair_20261122/cpu_ALMANAC \
  --expected-manifest /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/mechanism_repair_20261122/frozen/manifest.json
```

ONEIL 使用独立输出目录。每组仅64 fit/64内部验证、3次更新，检查首步各组件真实梯度、
作者Refiner前向一致、实际协同incidence干预、DTI–PPI/Pathway组合输出干预、路径隔离、
空知识、缺失3D、交换/分块和新实例严格checkpoint恢复。这些都是正确性检查，不报告性能提升。
正式入口相同模块的 `--stage train`，固定原seed0/划分/3000轮、原BCE整折每轮一次更新、
AdamW lr1e-4/eps1e-6/wd1e-4、传播gate decay0、全局梯度L2裁剪1；不添加辅助目标。
正式训练必须提供源码/输入冻结清单，GPU仍必须通过原统一队列，不直接另启调度器。
本轮未提交正式训练、五折或GPU检查，不读取封存测试、不自动推GitHub。
