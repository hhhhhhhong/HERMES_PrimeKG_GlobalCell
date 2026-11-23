# 输入适配版：ALMANAC / ONEIL seed0 五折

用户授权“跑一下五折”，随后明确“可以每个卡跑三个”。本轮注册两数据集各seed0原CV1五折，每折3000完整轮，无额外seed或参数网格；所有下游参数从头可训练，不加载旧协同checkpoint。封存测试不参与。

模型为 EntityAdaptedHypothesisModel：三路门控融合后的Drug A/B共享一个256维输入适配器、Cell独立适配器，均为非仿射LN→无bias Linear256→GELU→非仿射LN，随后原1层4头Transformer和原对称HERMES头。没有改变已验证适配版的目标边精确屏蔽、Cell候选、γ协同Query、20%随机Dropout、图谱、原BCE或编码器；取消屏蔽是此前讨论过的另一个变量，没有混入本次适配对照。

复用原完整epoch训练器，每query_chunk1024重算屏蔽图，完整fit仅一次AdamW更新、梯度裁剪1、eps1e-6、lr1e-4、weight_decay1e-4，原截断指数调度。按首次评估起内部验证AUPRC主选点，同时保留全程AUROC与1000轮以后AUROC；评估频率、3000轮预算及原划分重复监督限制均保持。原数据划分重复记录的限制见TARGET_OVERLAP_AUDIT.json。

已有5单元与双数据集CPU64监督两次更新证明两个适配器梯度/实际更新非零、440项原state初始化保真、交换/分块/严格重载通过。正式放行前两数据各进行2048监督、chunk1024、完整真实图3次GPU更新。每个数据集fold0先启动，前三次完整fit更新及所有模块、两个适配器和CF Query梯度通过后，放行剩余四折。

唯一GPU调度器仍为MCDSP-main/tools/paper_routes/queue.py。显式设置GPU0/1/2/3全用户进程上限各3，新job上限也为3；各卡所有目录用户任务合计，不按调度器分开计数。旧job自身更严格的上限保留，显存不足继续排队。仅重启CPU控制器接管旧GPU worker，不停止、迁移或重复启动旧worker。已取消旧路线不恢复。

桥接：train_hermes_entity_adapter_cv5 → dual_branch.adapted_readout_cli，后者在新worker进程内替换模型工厂并复用原冻结训练器，不修改正在运行源码。新增结果：runs/entity_token_adapter_cv5_seed0_20261122；检查、冻结、授权和调度记录：artifacts/entity_token_adapter_cv5_seed0_20261122。旧结果保持原目录和原模型身份。所有新增结果/checkpoint/log/cache/tmp在IFS2；--output保留兼容入口，--runtime-directory仍指向IFS2共享运行目录。

CPU观察器report_entity_adapter_cv5.py只更新状态、显存放置预算、GPU_READY和ENTRY标记，不启动GPU。STATUS.json和RESULTS.md报告当前进度；全部五折完成后才报告五折内部验证均值与标准差，不将正确性检查或未完成折当作最终性能。
