# 联合多模态融合（2026-11-22）

旧 K-Paths 五折采用 `h_modal + bounded_delta(knowledge, h_modal, .25)`，
末端 Transformer 另输出最多 10% 的增量。新模型把知识作为联合预测的输入模态。
它不保证提高成绩，必须通过后续相同划分、训练预算与选点口径的实验验证。

## 实际结构

```text
全节点固定 CODER → 3 层 α/β → 全局 Cell/Synergy token + 有序路径 GRU
                                             ↓
                                    DrugPair × Cell 条件 γ
                                             ↓
                                     三个知识向量 [B,3,256]
                                             │
BRICS 2D + 固定 Uni-Mol 3D → Drug A/B 向量     │
原 Cell 表达 MLP          → Cell 向量         │
                   ↓                         ↓
          分别 LayerNorm → 同实体拼接 [B,3,512]
                                 ↓
         Linear(512,256) → GELU → Dropout(.2) → LayerNorm
         （两药物共享参数；Cell 使用自己的融合投影）
                                 ↓
          普通 Transformer：3 token / 4 head / FFN512 / 1 层
                                 ↓
            原 Consolidator 结构 + 双药交换对称读出 → 协同预测
```

对于实体 i：

`z_i = Project_role([LN(h_modal_i) || LN(h_knowledge_i)])`。

不存在 `h_modal + W_delta h_knowledge` 分支接入、零输出初始化或 25%/10% 外部残差限幅。
Transformer 内部保留标准 Attention/FFN 残差；Cell MLP 与药物编码器内部原有跳连也保留。
所有下游组件从第一次反向传播即可获得监督梯度。

## 本次变化边界

- 实现位于 `dual_branch/multimodal_fusion.py`，仍在原工程文件夹内。
- 原五折代码、checkpoint 与结果保留；本模型不载入其训练权重。
- 沿用 `kpaths_data.load` 与 `KPathsModel.knowledge_table`：8 类图关系、Cell 表达输入、
  训练折正例 Synergy、三层 α/β、GRU 路径候选和 γ 的证据读取范围均沿用旧版。
- 药物编码器内部仍以 `BRICS fragment + coverage × projected Uni-Mol fragment` 融合
  2D/3D。本次改的是 **知识流与结构/表达流之间** 的融合，并未将 2D、3D、知识拆为三条独立流。
- 原 HERMES Refiner 仍不在此模型中；生物关系组合读取范围和路径覆盖率也没有因此改善。
- `use_knowledge=False` 将知识输入置零后经过同一个融合层和 Transformer；
  它是同容量读取干预，不等于原 HERMES 预测。完全空证据与该干预预测相同。
- 固定 CODER / Uni-Mol 缓存不微调；两个编码器的下游投影、图传播和预测模块均可训练。

## 验证与训练状态

单元测试：

```bash
cd /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell
PYTHONDONTWRITEBYTECODE=1 /ifs2/home/liushuhong/anaconda3/envs/py38/bin/python \
  -m unittest discover -s tests -p test_multimodal_fusion.py -v
```

真实全图 CPU 检查入口：`python -m dual_branch.multimodal_checks --dataset ALMANAC
--result-directory /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/multimodal_fusion_20261122/cpu_ALMANAC`。
ONEIL 使用独立输出目录。每组仅 64 条 fit 监督、64 条内部验证、3 次更新，
审查第一步各组件真实梯度、交换/分块、空证据、路径隔离、真实 3D 输入与磁盘 checkpoint 重载。
报告位于上述目录的 `CHECKS.json`；这是正确性检查，不是性能结果。
4 项单元测试、ALMANAC 与 ONEIL 两套完整图 CPU 各 3 次更新现均通过。
全部组件第一步梯度非零、交换误差 0、独立实例严格重载误差 0；
64 份原五折训练源码 SHA 核验未变。检查器首次使用了 Python 3.9 才支持的路径 API，
在模型构造前失败，已改为兼容当前 Python 3.8 的路径判断；失败记录保留在
`artifacts/multimodal_fusion_20261122/failed_attempts/python38_path_api.json`。

本次未提交正式训练或新五折。模型实现了与现有 `stable_training.train_epoch` 相同的
`base_features / knowledge_table / forward / components / optimizer_param_groups` 接口，
真实检查直接调用该训练步骤。后续正式入口必须使用新模型构造、独立结果目录与正确配方记录；
不能直接运行仍指向旧残差模型的 `kpaths_cli.py` 或套用旧的残差恒等检查。
GPU 任务仍必须走原统一队列，全用户每卡最多两个进程；所有新增输出位于 IFS2。
