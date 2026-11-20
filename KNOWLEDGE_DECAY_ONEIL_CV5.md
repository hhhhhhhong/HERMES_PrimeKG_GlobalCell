# ONEIL：知识分支不衰减的五折训练

2026-11-19 用户明确选择将新“不衰减方案”扩展到 ONEIL 五折。ALMANAC 新方案仍只有原先获准的 seed0/fold0；原全参数衰减版五折和历史结果继续保留。

唯一配方变动是 Adam 的参数分组：`initializer.*` 和 `head.*` 保持 coupled weight decay 0.01；其余知识参数（CODER 类型投影、三层 α/β、传播门控、关系交互、条件 γ 和知识读出）使用 0。不改 Adam 为 AdamW，不增加模块、辅助目标或训练 seed。

模型仍为统一15类异构超图、三层零初始化门控残差 α/β、真实共享实体的 HYPER 式关系交互、DrugPair×Cell 条件 γ 和原式预测头。药物使用 HERMES 分子图编码，Cell 使用表达 MLP，外部实体使用固定 CODER 输入。Synergy_Hyperedge 仅由各折 fit 正样本构建，角色为 `[0,0,2]`，训练期整条协同消息 Dropout 20%。

ONEIL seed0/fold0–4 均从随机初始化开始，全部下游参数可训练，不载入任何旧模型或检查 checkpoint。初始化逐折与原全参数衰减实验的哈希对齐；旧 checkpoint 只作完成后的同折分数参照。保留原划分、BCE、学习率1e-4、截断指数调度、query chunk1024、整折一次更新和3000轮，不早停。仍在 epoch1000–3000 依据内部验证 AUROC 选择 checkpoint，AUPRC 报告该同一 checkpoint 的结果。

新入口为 `scripts/train_knowledge_decay_oneil_cv5.py`，复用原五折训练器和已有 `scripts/knowledge_decay.py`，不修改正在训练的 ALMANAC 新方案或原五折冻结文件。每折保留监督梯度与衰减项、门控、实际传播增量和固定选中 checkpoint 的验证干预；这些干预不用于选参。五折全部核验3000次更新、源码/划分/初始化和 checkpoint 哈希后才汇总均值及总体标准差；封存测试始终不读取。

先完成 ONEIL 完整图 CPU 三步检查，再经同一个 `auxiliary_jobs.json` / `queue.py` 提交一项 GPU 三步检查和五个正式 fit。GPU 检查通过后启动 fold0；其前两次真实整折更新和门控/传播梯度时序通过后放行 fold1–4。检查不是性能结果。

五折候选卡为物理 GPU0、GPU1；统一队列按所有目录合计每卡最多两个本用户 GPU 进程，显存不足则等待。依据旧 ONEIL 实测约4952MiB进程占用，放置预留6144MiB，不因抢跑缩图、减 query chunk 或改配方。原 GPU2/GPU3 worker 保持运行，不另启 GPU 调度器。所有新增源码、结果、日志、状态、缓存和临时文件在 `/ifs2`。

结果：`runs/knowledge_decay_ONEIL_cv5_seed0_20261119/`。

冻结、检查、入队与核验：`artifacts/knowledge_decay_ONEIL_cv5_seed0_20261119/`。

本轮是用户直接授权的五折验证，不以 ALMANAC 单折是否先提升作为 ONEIL 的启动条件。不能把门控保留下来等同于知识带来性能提升，也不能将该整组衰减实验的收益归因于某一个门控参数。

2026-11-19显存恢复更新：fold1、2首步前因共享显存竞争退出，失败记录保留，使用同seed/同配方技术重跑并继续保持5个唯一正式fold。剩余新ONEIL任务在每张候选GPU上的放置上限收紧为1个本用户进程；统一全局上限仍为2，显存预留仍6144MiB。详情见本批MEMORY_RECOVERY.json和MEMORY_RECOVERY_VERIFIED.json，不覆盖历史提交计划。
