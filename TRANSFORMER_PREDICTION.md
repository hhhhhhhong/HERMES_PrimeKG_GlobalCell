# 预测前的 Transformer 交互模块

## 接入位置

```text
现有知识/模态融合后的三个表示 [Drug A, Drug B, Cell]
    ↓ [batch, 3, 256]
共享 Drug 角色编码；独立 Cell 角色编码
    ↓
1 层 Transformer Encoder（4 heads，FFN 512，Dropout 0.2，Pre-LN）
    ↓
零初始化输出投影 + 原输入残差
    ↓
原 HERMES Consolidator；交换两药的两个读出取平均
    ↓
协同概率
```

Cell 表达矩阵、Cell MLP、PrimeKG 构图和知识筛选均未在这项改动中修改。
本模块也没有实现此前讨论的知识精选；后续应分别比较构图与预测交互的贡献。

## 使用

实现：`dual_branch/transformer_prediction.py`。

```python
from dual_branch.transformer_prediction import TransformerDualBranchModel

# data 沿用 dual_branch.data.load 的返回值。
model = TransformerDualBranchModel(data).to(device)
optimizer = torch.optim.Adam(model.optimizer_param_groups())
```

`forward(rows, table, base)`、`base_features()`、`knowledge_table()` 和
`components()` 接口兼容原模型，可以使用 `global_cell.network.train_epoch`
的整折梯度累积方式。独立入口 `dual_branch/transformer_cli.py` 复用原训练
循环，只修改模型导入与版本记录；既有学习率/调度/3000轮/选轮规则均保留。
入口支持 `--stage check` 或 `--stage train`，不应将它的结果混入原五折。
2026-11-20 用户随后明确要求直接运行五折。已通过同一统一队列提交
ALMANAC、ONEIL 各 seed0 五折（共10个正式fit）及两套GPU检查；检查通过后
自动放行正式训练，不能将注册或检查阶段称为正式fit已经开始。

所有下游参数从头可训练，不加载历史协同 checkpoint；新增 Transformer
是随机初始化的小型 Encoder，不需要下载另一个预训练模型。
原 `dual_branch/model.py`、`dual_branch/cli.py` 和现有五折保持原样。

## 正确性与边界

- 两药共享角色且无位置编码；评估时交换两药得到相同预测，允许浮点误差。
- 注意力仅作用于每条样本的三个 token，不能跨 batch 混合不同样本。
- 输出矩阵初值为零，初始残差严格为零，初始预测与当前双流模型一致。
  这不代表恢复了原完整版 HERMES，也不保证训练后残差幅度有上限。
- 第一更新时输出投影收到监督梯度，Transformer 内部监督梯度为零；
  输出投影更新后内部注意力开始收到梯度。这是预期的梯度时序。
- 训练期 Dropout 具有随机性；单独模块不保证不同随机掩码下逐位相等。
  完整模型继续沿用原先的药物 ID 排序和对称读出。
- 原优化器分组保持，Transformer body 使用原预测头学习率和衰减；
  新零输出投影使用预测头学习率、无衰减，所有可训练参数恰好覆盖一次。
- 正确性检查不能解释为 AUPRC 提升；正式五折依用户后续明确授权提交。

检查入口：

当前核验：5 项独立检查、ONEIL 完整图 CPU 三次更新均通过，新增 593,152 个参数。
原有参数初值与原五折 seed0/fold0 哈希一致；原三组优化器配置逐项不变，
初始预测误差为 0。详见 `artifacts/prediction_transformer_20261120/DELIVERY.json`。
正式入口可传入该目录的 `manifest.json` 作为 `--expected-manifest`；新增修改后
须重新核验并冻结版本，不能把后续知识筛选变化归入当前快照。

```bash
/ifs2/home/liushuhong/anaconda3/envs/py38/bin/python -m unittest discover -s tests -p 'test_transformer_prediction.py' -v
```

真实完整图、少量监督样本的 CPU 三步检查入口：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /ifs2/home/liushuhong/anaconda3/envs/py38/bin/python -m scripts.check_transformer_prediction --dataset ONEIL
```

它使用既有 fold0 的 64 条训练、32 条验证样本，只检查梯度与前向正确性，
不报告预测性能、不读取封存测试；检查记录位于
`artifacts/prediction_transformer_20261120/ONEIL/CPU_CHECK.json`。
重复运行前需选择新的检查输出版本，不覆盖已经完成的检查记录。

## 已授权的正式五折

- 结果：`runs/dual_branch_transformer_cv5_seed0_20261120`。
- 状态：该目录的 `STATUS.json` 与 `RESULTS.md`，以实际状态为准。
- 冻结源码、输入、计划与调度证据：`artifacts/dual_branch_transformer_cv5_seed0_20261120`。
- 两数据集各 seed0/fold0–4，3000完整轮，全部下游从头可训练，无旧checkpoint初始化。
- 仍使用原BCE、整折每轮一次更新、query_chunk1024、原指数调度，
  epoch1000–3000按内部验证AUROC选轮；不读取封存测试，不扩seed或网格。
- 当前PrimeKG图与Cell输入保持；此前文献讨论的知识精选尚未作为新改动加入。
- 各数据集先通过完整图/1024监督查询的GPU三步检查；正式fold0完成三次整折
  更新、实际优化器分组和Transformer梯度时序核验后，放行其余四折。
- `scripts/report_prediction_transformer_cv5.py` 只观察和记录，不启动GPU进程；
  所有训练由既有 `MCDSP-main/tools/paper_routes/queue.py` 统一调度，所有目录
  该用户GPU进程合计每卡最多2，无旧并发例外。
- 新结果、checkpoint、stdout日志、状态、缓存、临时文件全部在IFS2；旧IFS入口
  仅为共享队列兼容参数，实际可写runtime为 `/ifs2/home/liushuhong/work2/runtime/paper_routes_20261009`。
