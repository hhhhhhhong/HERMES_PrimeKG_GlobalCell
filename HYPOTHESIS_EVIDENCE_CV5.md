# 精确目标边屏蔽与协同引导证据读取：正式五折

入口 `python -m dual_branch.hypothesis_cli`，实际 GPU 由现有共享 paper_routes.queue 通过 train_hermes_hypothesis_evidence_cv5 桥接启动。不得另开GPU调度器。

ALMANAC、ONEIL 各 seed0、原划分五折，3000完整轮，全部下游参数从头训练，不加载旧协同checkpoint、不冻结。复用固定CODER/UniMol输入。每轮一次全fit BCE更新；query_chunk1024，每batch在双传播支路中精确移除当前查询全部重复/逆序Synergy边并重算图表，backward在屏蔽上下文内完成。仅传播前的模态/CODER投影缓存。Refiner协同表示经非仿射归一化后拼入γ Query，保持梯度。保留20%随机超边Dropout；无新损失。

AdamW，lr1e-4、eps1e-6、weight_decay1e-4，传播gate不衰减，全局梯度裁剪1，原0.9997截断指数调度，最低2e-6。首次评估起按内部Val AUPRC选selected.pt，同时保存全程/1000轮以后AUROC诊断点。每轮1/2/3、1000前每5轮、1000后每轮评估，无早停。AUPRC为原作者precision-recall曲线梯形积分。

评估图移除全部评估查询对应的训练正例边，再分块读出；不使用验证标签构图。沿用原split存在重复三元组监督，因此不宣称消除所有泄漏或严格独立泛化。具体统计在artifacts/hypothesis_evidence_20261122/TARGET_OVERLAP_AUDIT.json。封存测试不参与。

先在两数据集真实完整图上各检查2048监督、chunk1024、3次更新及checkpoint重载、交换/空知识/表作用域/CF Query作用。GPU检查通过后放行fold0；每个数据集fold0前三次全fit更新及各组非零有限梯度通过后放行其他四折。CPU观察器只写状态与依赖标记，不启动GPU。所有用户任务每卡合计最多2进程；旧实验继续运行。

新结果 runs/hypothesis_evidence_cv5_seed0_20261122；来源冻结、GPU检查与调度凭据 artifacts/hypothesis_evidence_cv5_seed0_20261122。每batch重算使ALMANAC/ONEIL每轮分别需要54/14张传播图，不能沿用旧版耗时估计。GPU检查与最初全fit轮提供实测耗时，不把正确性检查解释成性能提升。
