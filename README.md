# HERMES_PrimeKG_GlobalCell

HERMES 药物/细胞编码与 PrimeKG 全局异构超图的实验代码。
当前正式方案为三层真实 α/β 传播，每层零初始化门控残差，保留关系交互和样本条件化 γ。

```text
Drug: HERMES 分子图编码器 ─┐
Cell: HERMES 表达 MLP ───┼→ 统一15关系异构超图
其他实体: CODER + 类型投影 ┘       ↓
                        3轮 α/β + 每层零门控残差
                                  ↓
                        真实共享实体上的关系交互
                                  ↓
                        DrugPair × Cell 条件化 γ
                                  ↓
                        门控残差读出 → HERMES式预测头
```

协同超边只使用当前训练折正例，Drug/Drug/Cell 角色为 `[0,0,2]`；
训练期在 β 后对协同边消息施加20% Dropout，验证时关闭。
新传播层门控初值为0；末端原读出门控仍为0.05。
激活重计算只降低显存占用，保持模型公式、参数和随机流。
没有 SimilarityRefiner 前向平滑，没有新增相似度重构损失。

## 当前训练配方

- ALMANAC、ONEIL 各 seed0 五折；每折3000轮。
- 全部下游参数从头训练，不冻结，不加载旧 checkpoint。
- 每轮一次完整训练折更新，query_chunk=1024。
- BCE、Adam、原学习率调度；第1000–3000轮按内部验证AUROC选模型。
- 外部 CODER 向量是固定输入；封存测试不参与本轮训练或选参。

## 主要文件

| 文件 | 用途 |
|---|---|
| `global_cell/attention.py` | α/β、真实共享实体关系交互、条件化 γ |
| `global_cell/synergy.py` | 当前训练折正协同三元超边 |
| `global_cell/network.py` | 单轮底座、门控读取和原式预测头 |
| `models/zero_gated_hgnn.py` | 三轮全节点迭代及零门控残差 |
| `models/zero_gated_checkpoint.py` | 等价的激活重计算 |
| `scripts/train_zero_gated_hgnn_cv5.py` | 当前正式训练入口 |
| `scripts/enqueue_zero_gated_cv5.py` | 本地统一队列提交 |
| `scripts/report_zero_gated_cv5.py` | 状态、完整更新核验与内部验证汇总 |
| `tests/` | 机制、对称性、门控梯度和重计算等价检查 |

详细方法见 [ZERO_GATED_HGNN.md](ZERO_GATED_HGNN.md)。
旧方案说明保存在 [docs/HISTORY.md](docs/HISTORY.md)，以当前方法文档为准。

## 数据与运行环境

本仓库是源码快照，不包含训练数据、PrimeKG 原始文件、编码向量、模型权重、
checkpoint、日志和实验结果。`models/` 在原服务器是目录链接，本快照存储实际Python源码。
`ENVIRONMENT.json` 记录当前服务器依赖版本；作者原依赖文件另存于 `vendor/hermes/requirements.txt`。

训练仍需准备原服务器的数据、审计/源码清单和路径配置。
部分准备脚本依赖相邻研究目录的已审计缓存，调度脚本依赖现有MCDSP统一队列。
相关队列入口的只读副本保存在 `integration/mcdsp/`，没有将整个MCDSP工程一并导出。
因此这次同步是当前代码备份，并非已经完成跨机器的独立一键复现适配。

## 来源

- HERMES：<https://github.com/Christina327/HERMES>，固定提交记录见 `vendor/hermes/PROVENANCE.json`。
- HYPER：<https://github.com/HxyScotthuang/HYPER>，辅助函数来源见 `vendor/HYPER_PROVENANCE.json`；保留作者MIT许可证。
- PrimeKG：使用经过审计的真实知识关系；不把潜在相似性解释为真实生物学事实。

原作者源码保持字节不变；本仓库没有为所有第三方代码重新声明统一许可证。
`SOURCE_MANIFEST.json` 记录本次复制的源码散列与来源，便于核对版本。
