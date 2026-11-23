# Drug / Drug / Cell 的 Transformer 输入适配

本次的“三个输入”指融合后的 Drug A、Drug B、Cell 三个实体 token。模态、协同、知识是每个实体的三个信息来源，仍由既有三路门控融合处理。适配器放在该融合之后、最后的一层四头 Transformer 之前。

```text
每个实体：模态 / Refiner协同 / γ知识证据 → 原三路门控融合 → 256维实体向量

Drug A → 药物适配器 ─┐
Drug B → 同一药物适配器 ├→ [B,3,256] → 原共享角色编码 → 原Transformer → 原对称HERMES预测头
Cell   → 独立Cell适配器 ┘
```

令 `t` 为 Drug 或 Cell：

`A_t(x) = LN_out(GELU(W_t LN_in(x)))`

输入、输出 LayerNorm 都沿单个 token 的最后256维执行、无仿射参数；`W_t` 为无bias的256→256投影。药物A/B共享投影，Cell独立。共新增131,072个参数，没有额外Dropout、残差融合、CLS token、注意力层、辅助损失或学习率超参数。现有Transformer保留1层、4头、FFN512、Dropout0.2、Pre-LN与标准内部残差。

现有融合与Transformer本来已有归一化；本次的新增自由度是Drug/Cell两套可学习通道映射。输出归一化使非退化token的RMS接近1，但不保证两个表示空间自动获得相同生物学含义。语义映射需要与预测目标共同训练，不能凭RMS或注意力熵宣称性能提升。LayerNorm的有限epsilon也意味着缩放前后坐标并非严格不变。

两药共享适配器和角色编码，保留原Drug交换对称读出；注意力只在当前样本的三个token之间计算。适配器未改原Cell表达MLP、药物BRICS/UniMol编码、CODER、αβ、关系交互、γ或三路门控。实现不改变是否屏蔽目标边：此独立验证类继承当前Hypothesis的精确屏蔽策略。之前讨论的不屏蔽训练协议需要单独版本化，不将两个改动混作单因素结论。

安装发生于新模型CPU构造阶段，只复制同一次新构造产生的Transformer参数，并保留父模型初始化随机流；不加载旧协同checkpoint。全部新增参数可训练，分为transformer_drug_adapter、transformer_cell_adapter两个优化器组，沿用同一lr和weight_decay。旧Transformer/图/模态/头的初始权重逐张量保持；适配后前向输出会变化，必须重新训练，不能插入已训练checkpoint直接当作性能改进。

实现：`dual_branch/entity_token_adapter.py`。独立入口：`python -m dual_branch.adapted_readout_cli`，复用原完整epoch训练器，仅在当前新worker进程中替换模型构造工厂。CPU检查示例：

```bash
python -m dual_branch.adapted_readout_cli \
  --dataset ALMANAC --fold 0 --stage check --device cpu --check-rows 64 \
  --result-directory /ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell/artifacts/entity_token_adapter_20261122/example_check
```

GPU调用仍必须通过现有统一队列，每卡该用户训练进程合计最多2；本次没有注册新GPU训练或重启五折。新输出、临时文件、缓存和检查checkpoint均在IFS2。旧五折继续原冻结配方，不能将旧结果归到适配版。

检查来源：5项单元测试；ALMANAC/ONEIL各完整真实图、64fit/64valid、2次epoch更新，每epoch两个32行屏蔽图。记录两个适配器独立梯度和实际权重更新、全部模块梯度、初始化/RNG保真、交换/分块、磁盘checkpoint严格重载、真实token尺度和Q/K注意力。检查结果只证明实现可训练，未检验泛化提升或CUDA/AMP行为。

凭据：`artifacts/entity_token_adapter_20261122`。最初单元测试错误地要求0.01倍缩放下坐标近乎严格不变，忽略LayerNorm epsilon相对输入方差的影响；模型未修改，测试改为明确的输出RMS和加性平移检查，原诊断保留于UNIT_TEST_FIRST_ATTEMPT.json。
