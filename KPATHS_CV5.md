# K-Paths 候选证据 + 全局 Cell / Synergy：正式五折

用户在离线检索审计后明确要求“五折”。本次在原工程接入模型，并提交 ALMANAC、ONEIL 各 seed0、原 CV1 五折，共 10 个全新训练任务。旧模型与实验继续运行，旧结果完整保留；没有恢复已停止的 PathBatch。

## 当前结构

1. 原始 PrimeKG 离线检索：对每个可能的药物对生成最多 10 条靶点支持路径，最长 3 跳，另补受约束的 `Drug–Target–GO/Reactome–Target–Drug` 四跳。
2. 这些路径的真实事实并集与原 Cell 表达关系组成统一全局超图，各折加载后只加入本折 fit 正样本的 `Synergy_Hyperedge {DrugA, DrugB, Cell}`。
3. 所有节点继续使用已修正分词的固定 CODER 输入，经三轮 α/β 门控传播；全局 Cell/协同渠道保留真实共享实体的关系交互。
4. 全局传播后的节点表征与有方向的关系 embedding 交错组成路径序列，用一个共享单层 GRU 编码。它是新增的轻量任务适配，不是 K-Paths 作者预测模型。
5. 原 DrugPair×Cell 条件化 γ 读取**该药物对的实际路径**及全局 Cell/协同渠道，输出三个实体的知识上下文。这里没有把全局 KG 类型表示当作该样本的路径。
6. 独立模态分支仍为 BRICS 2D + 固定 Uni-Mol 3D 缓存、原 Cell 表达 MLP。知识以零初始化输出残差接入三个模态 token，再经现有最终预测 Transformer、原 Consolidator 的对称双药读出。

路径按数据集药物 ID 的规范顺序编码，保留反向关系及真实端点角色，交换两药保持预测一致。相同药物对在不同 Cell 下共享静态候选，但 γ 的 Query 包含 Cell 表达编码，能够产生不同路径权重。

当前全局超图有 **8 类超边**：真实二元 Drug–Target、PPI，BP、CC、MF、Reactome 成员超边，Cell 表达边，以及 fit 正例协同三元超边。功能超边保留有原始成员事实支持的子集，不声称这些成员彼此具有已验证的多元相互作用。PPI 主路径当前是三跳 `Drug–Target–Target–Drug`，尚不包含任意多跳蛋白桥。

疾病、禁忌证和 enzyme/carrier/transporter 路径不进入本版主机制候选；上一轮完整原始图审计仍保留它们。这项取舍由先前疾病/代谢路径占比过高的审计推动，仍需五折验证效果。模型未恢复作者原 Refiner；保留协同超边不能称为与原完整 HERMES 等价。

## 离线数据与隔离

从原始允许事实中选取 target、PPI、GO、Reactome，检索池共 661,714 条去重事实，不受旧蛋白池限制。基础搜索最多 128 个实体候选、2 秒软预算；每个关系模板保留最多两个实体实例，四跳功能集合原始成员上限为 200。本次所有药物对均未触及搜索预算。

| 统计 | ALMANAC | ONEIL |
| --- | ---: | ---: |
| 药物数 | 87 | 38 |
| 严格映射到 PrimeKG 的药物 | 85 | 32 |
| 全部可能无序药物对 | 3,741 | 703 |
| 有符合靶点约束连接路径的药物对 | 1,252 | 245 |
| 缓存路径条数 | 4,402 | 917 |
| 路径事实并集 + 直接靶点事实 | 2,100 | 923 |
| Cell 表达边 | 35,310 | 20,544 |
| 全局节点（原节点保留） | 18,201 | 15,203 |

这些是所有可能药物对的候选统计，不能当作实际训练样本覆盖率。fold0 的实际 fit/valid 路径覆盖为 ALMANAC 21,396/54,354、5,327/13,589；ONEIL 4,465/13,643、1,139/3,411，其他折见输入审计。有路径也不代表已证明协同机制。无路径样本保留显式空掩码，使用全局 Cell/协同和模态信息；一侧邻居 fallback 仅保存在审计记录，不当作连接机制读入。

离线准备只读取药物身份及 KG，不读协同标签、实际验证/封存测试样本或表达数值。使用已知全体药物身份属于沿用的 transductive CV1 协议。每折训练时才按原 fit 标签生成协同边，内部验证与封存测试标签不参与构图。Cell 表达输入、全部 Cell 表达边及其 route 内容逐项不变，旧 CODER 数值逐元素相同；本次没有新实体需要重新编码。

10 折原始 fit/valid 行 SHA、fit 正例协同边、Cell 边/routes、路径掩码和 2,425 条不同原始事实行已独立回查通过，见 `artifacts/kpaths_cv5_seed0_20261120/INPUT_AND_SOURCE_AUDIT.json`。该补充审计首次因整数/字符串 route 键比较失败，原失败脚本和日志保留于 `failed_attempts/input_audit_key_type`；规范化键后通过，没有修改模型或输入。

## 固定训练配方

- 全部下游参数从头可训练，不冻结，不加载旧协同 checkpoint。固定 CODER 与 Uni-Mol 输入继续复用；不新增 RL、LLM 或辅助损失。
- 每折完整 3,000 轮，原 BCE，整折梯度累积后每轮一次更新，查询分块 1,024。没有早停，没有新增 seed 或调参网格。
- 沿用修复版 AdamW：所有组 lr=1e-4、eps=1e-6、解耦衰减=1e-4；传播门控和零输出投影 decay=0。全局梯度 L2 裁剪为 1。
- 三层内部传播门控初始为 0.1，外部知识和 Transformer 输出矩阵初始为 0；初始预测与本次模型的纯模态读出一致，不等于原 HERMES 的已训练预测。
- 知识残差与 Transformer 残差分别限制在同一模态 token 范数的 25% 和 10%，合计不超过 35%。这不限制可训练模态分支本身的变化，也不保证验证性能。
- 主 checkpoint 仍按 epoch1000–3000 内部验证 AUROC 保存为 `selected.pt`。额外从首次评估保存 `best_aupr.pt`，只作另一选点规则的诊断，不与主结果混记。AUPRC 继续使用作者梯形积分口径。
- 所有新数据、源码快照、结果、日志、缓存和临时文件均位于 IFS2。

## 正确性与调度

6 项检索检查、2 项有序路径编码检查及两数据集完整图 CPU 三次更新通过。CPU 验证包括：初始预测误差为 0、真实路径/知识模块第二三步梯度非零、零增量与幅度上限、Drug 交换、分块一致性、新实例严格恢复；修改未选路径的读出向量不影响当前 Query，修改选中路径会影响上下文，Cell 改变会影响路径注意力。

正式任务各自等待完整图/1,024 监督查询的 GPU 三步检查通过；每套 fold0 完成三次完整 fit 更新并核对梯度后放行剩余折。这些检查不作为性能估计。

唯一 GPU 控制器仍为 `MCDSP-main/tools/paper_routes/queue.py`，共用 IFS2 `auxiliary_jobs.json`；只增加新桥接白名单，不改全用户进程统计。所有目录合计每卡最多两个本用户 GPU 进程，无并发例外。CPU reporter 只报告、核验和写依赖标记，不启动 GPU。

显存放置按实际 GPU 检查调整，不改变图、查询分块、优化器或训练预算。GPU 检查张量峰值 ALMANAC 约 1,146 MiB、ONEIL 约 772 MiB；ONEIL fold0 正式完整 fit 运行超过 70 轮后，进程实测约 1,310 MiB。最后据此将尚未启动的正式任务放置预留设为 2,560 MiB，保留 CUDA 上下文及额外余量。启动/逐次资源调整凭据为 `LAUNCH.json`、`CHECK_MEMORY_ADMISSION.json`、`MEASURED_GPU_ADMISSION.json`、`ALMANAC_FIT_ADMISSION.json`、`MEASURED_PROCESS_JOB_ADMISSION.json`。最终实际状态读取 `STATUS.json`，不能把提交队列、环境导入或 GPU 冒烟说成正式训练已更新。

## 入口与输出

- 训练模型：`dual_branch/kpaths_model.py`。
- 数据适配：`dual_branch/kpaths_data.py`，原训练公式复用 `stable_training.py`。
- 训练入口：`dual_branch/kpaths_cli.py`，由共享队列桥接 `train_hermes_kpaths_cv5.py` 调用。
- 已执行的一次提交入口：`scripts/enqueue_kpaths_cv5.py`，有防重复检查，不应重复提交。
- 正式结果：`runs/kpaths_cv5_seed0_20261120/{ALMANAC,ONEIL}/fold{0..4}`。
- 状态和汇总：同一 run 下 `STATUS.json`、`RESULTS.md`；各数据集完整五折后才汇总其均值与标准差，均为内部验证。
- 源码/输入冻结与验证：`artifacts/kpaths_cv5_seed0_20261120`，64 份源码、42 份输入，原 53 份训练源码保持不变。
- 指针：`KPATHS_RUN.json` 和 `CURRENT_RUN.json`；旧指针内容与所有旧 run 保留。

只读刷新报告可在当前工程运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/ifs2/home/liushuhong/anaconda3/envs/py38/bin/python -B -m scripts.report_kpaths_cv5
```

没有推送 GitHub，也没有读取封存测试。最终性能结论须等待这套独立训练完成。
