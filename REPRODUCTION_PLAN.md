# Search-R1 代码理解与复现执行手册

这份手册把“理解项目代码”和“复现最小 PPO 搜索训练链路”拆成可执行步骤。建议先按阅读路径建立脑内链路图，再做环境、数据、检索服务和 smoke test。

## 1. 先建立全局图

阅读顺序：

1. `ANNOTATED_CODE_GUIDE.md`：看项目模块地图、模型角色、训练数据流。
2. `README.md`：看官方安装、数据、检索、训练、推理说明。
3. `train_ppo.sh`：看实际复现入口，重点关注 Hydra 覆盖参数。

需要先记住的主线：

```text
数据处理
  -> 检索服务
  -> Ray PPO Trainer
  -> Search Agent rollout
  -> Reward / KL / Advantage
  -> Critic update
  -> Actor update
  -> 推理验证
```

## 2. 按源码追踪数据流

按这个顺序读代码并做笔记：

1. `scripts/data_process/nq_search.py`
   - 理解 NQ 样本如何变成 parquet。
   - 关注字段：`prompt`、`data_source`、`reward_model.ground_truth`、`extra_info.index`。

2. `verl/utils/dataset/rl_dataset.py`
   - 理解 parquet 如何被 tokenizer 转成 `input_ids`、`attention_mask`、`position_ids`。

3. `verl/protocol.py`
   - 理解 `DataProto` 如何统一承载张量、非张量 metadata 和 `meta_info`。

建议整理一张表：

| 阶段 | 关键字段 | 形状/含义 |
|---|---|---|
| Dataset | `input_ids` | `[batch, prompt_len]` |
| Rollout | `responses` | `[batch, response_len]` |
| Reward | `token_level_scores` | `[batch, response_len]` |
| Critic | `values` | `[batch, response_len]` |
| Advantage | `advantages`, `returns` | `[batch, response_len]` |

## 3. 按源码追踪训练循环

重点读：

1. `verl/trainer/main_ppo.py`
   - 入口：Hydra 配置、Ray 启动、tokenizer、worker mapping、`RewardManager`。

2. `verl/trainer/ppo/ray_trainer.py`
   - 入口：`RayPPOTrainer.fit()`。
   - 按顺序标出：生成、搜索循环、reference logprob、critic values、reward、KL、advantage、critic update、actor update。

3. `verl/trainer/ppo/core_algos.py`
   - 重点公式：`compute_gae_advantage_return()`、`compute_policy_loss()`、`compute_value_loss()`、`kl_penalty()`。

## 4. 按源码追踪模型角色

重点读：

1. `verl/workers/fsdp_workers.py`
   - Actor/Ref/Critic/Reward model 的构建与 FSDP 包装。

2. `verl/workers/actor/dp_actor.py`
   - Actor 如何计算 response token 的 logprob。
   - PPO clipped policy loss 如何回传。

3. `verl/workers/critic/dp_critic.py`
   - Critic 如何输出 token-level value。
   - Value clipping loss 如何更新。

模型角色对应：

| 角色 | 模型类型 | 是否训练 | 用途 |
|---|---|---:|---|
| Actor | `AutoModelForCausalLM` | 是 | 生成搜索/回答动作 |
| Rollout | HF/vLLM 推理封装 | 间接同步 | 高效采样轨迹 |
| Reference Policy | 冻结 CausalLM | 否 | KL 参考基线 |
| Critic | `AutoModelForTokenClassification(num_labels=1)` | 是 | 预测 value |
| Reward Model | 可选 TokenClassification | 可选 | 学习式奖励 |
| Retriever | BM25/Dense/online search | 否 | 环境工具 |

## 5. 按源码追踪搜索 Agent

重点读：

1. `search_r1/llm_agent/generation.py`
   - `LLMGenerationManager.run_llm_loop()` 是多轮搜索 Agent 的核心。
   - 关注 `<search>`、`<information>`、`<answer>` 如何改变轨迹。

2. `search_r1/llm_agent/tensor_helper.py`
   - 理解多轮生成时如何拼接 response 和 observation。

3. `search_r1/search/retrieval_server.py`
   - 理解 `/retrieve` 接口如何返回 top-k 文档。

## 6. 环境准备

Search-R1 训练环境：

```bash
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install wandb
```

检索环境：

```bash
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini uvicorn fastapi
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## 7. 数据与索引

处理 NQ search 数据：

```bash
conda activate searchr1
python scripts/data_process/nq_search.py
```

如果使用官方索引和语料：

```bash
save_path=/path/to/save
python scripts/download.py --save_path "$save_path"
cat "$save_path"/part_* > "$save_path"/e5_Flat.index
gzip -d "$save_path"/wiki-18.jsonl.gz
```

然后确认 `retrieval_launch.sh` 中的索引、语料路径与实际路径一致。

## 8. 启动检索服务并验收

```bash
conda activate retriever
bash retrieval_launch.sh
```

另开终端验证：

```bash
python search_r1/search/retrieval_request.py
```

验收点：

- 服务地址是 `http://127.0.0.1:8000/retrieve`。
- 返回 JSON 中有 `result`。
- 每条文档里有 `document.contents`。

## 9. 训练前 smoke test

不要直接改 `train_ppo.sh` 做小步数测试，优先使用：

```bash
conda activate searchr1
bash scripts/smoke_train_ppo.sh
```

可用环境变量覆盖：

```bash
DATA_DIR=data/nq_search \
BASE_MODEL=Qwen/Qwen2.5-3B \
N_GPUS=8 \
TOTAL_STEPS=2 \
bash scripts/smoke_train_ppo.sh
```

验收点：

- 能进入 `RayPPOTrainer.fit()`。
- 至少完成 1-2 个 global steps。
- 没有 Ray、CUDA、vLLM、retriever 连接错误。
- 日志出现 `critic/score/mean`、`actor/pg_loss`、`actor/ppo_kl`、`env/number_of_valid_search` 等指标。

## 10. 正式 PPO 复现

```bash
conda activate searchr1
bash train_ppo.sh
```

主要观察：

- `critic/score/mean`：规则奖励均值。
- `critic/rewards/mean`：扣 KL 后的奖励。
- `actor/pg_loss`：策略梯度损失。
- `actor/ppo_kl`：PPO 近似 KL。
- `response_length/*`：生成长度是否频繁打满。
- `env/number_of_valid_search`：搜索动作是否有效。

## 11. 推理验证

先启动检索服务，再运行：

```bash
conda activate searchr1
python infer.py
```

验收点：

- 模型能输出 `<search>...</search>`。
- 检索结果能被包装成 `<information>...</information>`。
- 最终能输出 `<answer>...</answer>`。
- `verl/utils/reward_score/qa_em.py` 能解析最终答案。

## 12. 常见问题排查

- 缺少 `tensordict`：说明 `pip install -e .` 或依赖安装不完整。
- 检索服务连接失败：检查 `retriever.url`、端口 8000、`retrieval_launch.sh` 路径。
- HuggingFace 模型无权限：把 `BASE_MODEL` 改成可访问的 Qwen2.5 base 模型。
- CUDA OOM：降低 batch size、response length、max turns，或打开/offload 相关配置。
- vLLM 初始化失败：确认 `vllm==0.6.3` 和 `VLLM_ATTENTION_BACKEND=XFORMERS`。

## 13. 最终理解产物

复现前后建议自己补一张链路图，至少包含：

- `DataProto` 字段从哪里来、在哪里被新增。
- Actor 输入输出：`input_ids -> log_probs`。
- Critic 输入输出：`input_ids -> values`。
- Reward 到 Advantage：`token_level_scores -> token_level_rewards -> advantages/returns`。
- Loss：policy loss、entropy loss、optional KL loss、value loss。
