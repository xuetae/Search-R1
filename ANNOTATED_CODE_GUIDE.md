# Search-R1 注释版代码导读与复现指南

本仓库已按复现学习用途为 Python、Shell、YAML、TOML 代码补充中文注释。注释目标是帮助阅读 Search-R1 的端到端训练与推理流程，原则上保持原始执行逻辑不变。

## 复现入口

1. 数据处理：`scripts/data_process/nq_search.py` 将 QA 数据整理成 parquet，核心字段包括 `prompt`、`data_source`、`reward_model.ground_truth` 和 `extra_info.index`。
2. 检索服务：`retrieval_launch.sh` 或 `search_r1/search/retrieval.sh` 启动本地检索接口，训练时默认访问 `http://127.0.0.1:8000/retrieve`。
3. PPO 训练：`train_ppo.sh` 调用 `python -m verl.trainer.main_ppo`，通过 Hydra 覆盖模型、数据、rollout、critic、retriever 和 trainer 配置。
4. GRPO 训练：`train_grpo.sh` 与 PPO 类似，但通常设置 `algorithm.adv_estimator=grpo` 并使用多采样组内优势。
5. 单样本推理：`infer.py` 加载训练后的 Search-R1 模型，循环生成 `<search>` 或 `<answer>`，并把搜索结果拼回上下文。

## 核心模块地图

- `search_r1/llm_agent/`：Search-R1 Agent 环境。`generation.py` 管理多轮 LLM 生成、动作解析、检索调用和轨迹拼接；`tensor_helper.py` 处理 padding、mask、position ids 和上下文裁剪。
- `search_r1/search/`：检索后端。支持 BM25、Dense FAISS、rerank、Google/SerpAPI 等服务形态。
- `verl/protocol.py`：`DataProto` 数据协议，是 Actor、Critic、Reward、Rollout 和 Trainer 之间传递 batch 的统一容器。
- `verl/trainer/ppo/`：PPO/GRPO 主训练逻辑。`ray_trainer.py` 串联 rollout、reward、KL、advantage、critic update 和 actor update；`core_algos.py` 放算法公式。
- `verl/workers/`：分布式模型角色。`fsdp_workers.py` 构建 FSDP Actor/Critic/Reward；`actor/` 和 `critic/` 实现策略损失和值函数损失；`rollout/` 适配 HF/vLLM 生成。
- `verl/single_controller/`：Ray 和 Megatron WorkerGroup 调度层，负责把 `DataProto` 切分到各 GPU worker 并收集结果。
- `verl/models/`：模型适配，包括 HuggingFace Llama/Qwen monkey patch 和 Megatron Llama 实现。
- `verl/third_party/vllm/`：vendored vLLM 多版本适配代码，用于训练权重和推理引擎之间同步。

## 模型角色

- Actor：`AutoModelForCausalLM`，被 PPO/GRPO 更新，学习何时搜索、如何利用检索信息、如何输出最终答案。
- Rollout：Actor 的推理侧封装，可用 HF 或 vLLM，高效生成训练轨迹。
- Reference Policy：冻结的 CausalLM，用于计算 `ref_log_prob`，给 KL 约束提供基准。
- Critic：`AutoModelForTokenClassification(num_labels=1)`，为 response token 预测价值 `V(s_t)`。
- Reward Model：可选的 token classification 奖励模型；默认复现路径更多使用规则奖励。
- Retriever：环境工具，不参与 RL 更新；模型通过生成 `<search>query</search>` 触发检索。

## 训练数据流

```text
RLHFDataset
  -> DataProto(input_ids, attention_mask, position_ids, non_tensor metadata)
  -> Actor/Rollout generate
  -> Search loop with <search>/<information>/<answer>
  -> RewardManager token_level_scores
  -> KL penalty token_level_rewards
  -> GAE/GRPO advantages + returns
  -> Critic update
  -> Actor PPO update
```

## 阅读建议

优先阅读顺序：

1. `train_ppo.sh`
2. `verl/trainer/main_ppo.py`
3. `verl/trainer/ppo/ray_trainer.py`
4. `search_r1/llm_agent/generation.py`
5. `verl/trainer/ppo/core_algos.py`
6. `verl/workers/actor/dp_actor.py`
7. `verl/workers/critic/dp_critic.py`
8. `search_r1/search/retrieval_server.py`

第三方 vLLM 与 Megatron 适配代码较多，建议在理解主链路后再按调用点追踪。

## 注释说明

本次注释采用“源码内中文注释 + 总导读”形式。部分文件原本已经含有中文说明和实验残留代码；对于原始仓库中会导致编译失败的明显损坏片段，已做最小可编译修复，并保留注释说明。
