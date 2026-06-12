# 第五部分：复现过程对比分析

本文是 veRL 下 Search-R1 源码拆解笔记的第五部分，重点讨论复现过程中的差异、实验结果、失败原因和新的改进 insight。前四部分可分别围绕原始实现、veRL rollout、veRL train、实验结果分析展开；本部分将这些内容放到复现链路中统一比较。

---

## 1. 复现目标与问题定义

Search-R1 的目标不是简单地让模型直接回答问题，而是训练模型在推理时主动调用搜索工具：

```text
Question
-> <think> 是否需要搜索 </think>
-> <search> 查询词 </search>
-> <information> 检索结果 </information>
-> <think> 基于证据整理答案 </think>
-> <answer> 最终答案 </answer>
```

在 veRL 中，这条链路被拆成三个核心系统：

1. rollout：模型生成 action，环境解析 `<search>` 或 `<answer>`，必要时调用 retriever。
2. reward：根据最终答案、格式、检索证据等规则打分。
3. train：使用 PPO/GRPO 更新 actor policy。

本次复现的目标不是完全复现论文中的 3B/7B 多卡实验，而是在单卡 48GB 环境下，用 `Qwen2.5-1.5B` 复现一个小型 Search-R1 链路，并验证其是否相对 baseline 有提升。

实际硬件和模型约束如下：

| 项目 | 复现设置 |
| --- | --- |
| GPU | 单卡 48GB |
| CPU | 12 vCPU |
| 内存 | 96GB |
| base model | `Qwen2.5-1.5B` |
| retriever | BM25 top1/top3 |
| dataset | NQ search split |
| trainer | veRL GRPO |
| max_turns | 1 |
| rollout samples | `n_agent=2` |

这些约束决定了本复现无法照搬论文大模型、大 batch、多 turn 的设置。核心问题变成：

> 小模型是否能学会稳定的 Search-R1 行为？如果不能，主要瓶颈是在搜索、格式、证据整理、reward 还是策略优化？

---

## 2. 官方论文设置与本地小型复现的关键差异

论文中的 Search-R1 主要使用 `Qwen2.5-3B/7B` 和 `LLaMA3.2-3B` 等模型，并在更大的 batch、更充分的 rollout 和更长训练步数下进行 RL。官方结果显示，Search-R1 可以显著提升开卷问答性能。

从公开指标看，论文中的代表性结果大致为：

| Model | NQ EM | Avg EM |
| --- | ---: | ---: |
| Qwen2.5-7B Search-R1-base | 0.412 | 0.373 |
| Qwen2.5-7B Search-R1-instruct | 0.397 | 0.384 |
| Qwen2.5-3B Search-R1-base | 0.341 | 0.292 |
| Qwen2.5-3B Search-R1-instruct | 0.323 | 0.327 |

本地复现得到的最好结果是：

```text
Qwen2.5-1.5B
-> search-lite SFT
-> grounding SFT
-> GRPO top3 shaped reward
-> NQ val/test_score = 0.250
```

这个数值低于论文中的 3B/7B 结果，但在当前硬件和模型约束下是合理的。主要差异如下：

| 维度 | 论文/官方设置 | 本地复现 |
| --- | --- | --- |
| 模型规模 | 3B/7B | 1.5B |
| rollout 数量 | 更大 group | `n_agent=2` |
| 搜索 turn | 多 turn 更完整 | `max_turns=1` |
| 训练数据量 | 更大 | 512-2048 样本级 smoke/small run |
| batch size | 更大 | `train_batch_size=2` |
| reward | EM/格式/检索 | 额外 shaped reward 和 collapse penalty |
| 目标 | 论文级指标 | 单卡小链路验证 |

因此，本地结果不能直接和论文结果做绝对值比较，更适合看相对提升：

```text
base / BM25 baseline / SFT / SFT+GRPO / grounding SFT+GRPO
```

---

## 3. 复现链路演进

### 3.1 直接从 base model 做 GRPO：不稳定

最早尝试使用 `Qwen2.5-3B` 或直接用 base model 跑 Search-R1 GRPO 时，主要问题是：

- 3B 模型在 48GB 单卡上显存压力过大。
- actor + rollout + ref + optimizer 状态导致 OOM。
- base model 不稳定输出 `<think>/<search>/<answer>` 格式。
- 初始 reward 大量为 0，GRPO 学习信号稀疏。

典型现象：

```text
TypeError: prompt must be a string...
CUDA out of memory
response_length/clip_ratio 高
env/finish_ratio 低
critic/rewards/mean 长期为 0
```

这说明，直接做端到端 Search-R1 RL 对小模型太难。

### 3.2 切换到 Qwen2.5-1.5B：解决显存，但不解决行为

切到 `Qwen2.5-1.5B` 后，显存问题明显缓解，可以完成 400 step 训练。但模型仍有几个行为缺陷：

1. 能生成 `<search>`，但 query 不稳定。
2. 能检索，但不会整理 evidence。
3. 能偶尔答对，但格式漂移明显。
4. RL 会强化一些“最后答对但过程很坏”的轨迹。

早期 top3 GRPO 得到过：

```text
SFT+GRPO shaped top3: val/test_score/nq = 0.203125
```

这说明 Search-R1 行为开始出现，但还不稳定。

### 3.3 Search-lite SFT：先拆技能，再接 RL

为降低任务难度，本地引入 Search-R1-lite SFT。它把 Search-R1 拆成几个监督子任务：

| task | 训练目标 |
| --- | --- |
| query | question -> concise search query |
| evidence_qa | question + evidence -> answer |
| decision_search | 判断需要搜索时输出 `<search>` |
| decision_answer | 判断证据足够时输出 `<answer>` |

对应脚本：

```text
scripts/data_process/nq_search_lite_sft.py
scripts/sft_qwen15_search_lite_1gpu.sh
```

这一步解决的是格式先验和子技能分解问题。训练后得到：

```text
verl_checkpoints/qwen2.5-1.5b-search-lite-sft/global_step_1000
```

验证 checkpoint 可正常加载：

```text
tokenizer ok: Qwen2TokenizerFast
model ok: Qwen2ForCausalLM
num params: 1543714304
```

### 3.4 Grounding SFT：补“搜索后整理证据”的缺口

Search-lite SFT 之后，模型会搜索，但经常不会从 `<information>` 中抽取答案。于是继续构建 grounding SFT：

```text
question + previous search + <information> evidence
-> <think> The relevant evidence supports the answer: ... </think>
-> <answer> ... </answer>
```

对应数据：

```text
data/nq_search_lite_grounding_sft
```

数据检查结果：

```text
train rows: 2802
test rows: 354
tasks:
- decision_search
- decision_answer
- post_search_answer
dirty rows: 0
missing think: 0
missing answer: 0 for answer tasks
```

继续 SFT 后得到：

```text
verl_checkpoints/qwen2.5-1.5b-search-lite-grounding-sft/global_step_300
verl_checkpoints/qwen2.5-1.5b-search-lite-grounding-sft/global_step_500
```

后续 GRPO 主要使用：

```text
global_step_300
```

因为它比更长 SFT 更不容易过拟合模板。

### 3.5 Grounding SFT + GRPO：当前最好基线

从 grounding SFT 300 接 GRPO top3，得到当前最好结果：

```text
val/test_score/nq = 0.250
```

这条链路是当前小模型的核心 baseline：

```text
Qwen2.5-1.5B
-> search-lite SFT global_step_1000
-> grounding SFT global_step_300
-> GRPO top3 shaped reward
-> NQ EM = 0.250
```

它说明：

- 小模型可以学会一定搜索行为。
- SFT 分阶段训练对 Search-R1 小模型很关键。
- 直接 RL 不如先 SFT 子技能对齐。

---

## 4. 实验结果对比

本地关键实验可以整理如下：

| 实验 | 起点 | 设置 | 结果 | 结论 |
| --- | --- | --- | ---: | --- |
| base 直接 GRPO | Qwen2.5-3B | 单卡 | OOM | 3B 全参 GRPO 不适合当前硬件 |
| Qwen1.5B top1/top3 GRPO | Qwen2.5-1.5B | shaped reward | 低且不稳 | base 缺格式和 grounding |
| search-lite SFT + GRPO | SFT1000 | top3 obs512 | 0.203125 | SFT 有效，但 evidence 整理不足 |
| grounding SFT300 + GRPO | grounding300 | top3 shaped | 0.250 | 当前最好基线 |
| LLDS chunk 强正则 | grounding300 | coef 0.05 | 0.000 | 正则过强，策略被锁死 |
| LLDS positive token | grounding300 | coef 0.01 | 仍有乱码 | LLDS 未解决 reward 漏洞 |
| LLDS adapted + penalty | grounding300 | coef 0.01 + penalty | 0.140625 | 过程更稳，但 EM 下降 |

### 4.1 为什么 `0.250` 是一个重要节点

`0.250` 不只是一个分数，它代表 Search-R1 小链路已经具备基本能力：

```text
会搜索
会读部分证据
能在一部分样本中输出最终答案
格式大体可解析
```

但它还没有解决：

```text
稳定格式
高质量 query
多证据归纳
避免无效中间轨迹
避免重复/乱码坍塌
```

### 4.2 为什么 LLDS 实验没有提升

LLDS 来自 “Lazy Likelihood Displacement” 论文，目标是防止有用 action 的 likelihood 在 GRPO 中下降。官方项目中有以下机制：

```text
NO_REDUCE_LAMBDA
REDUCE_THRES
ISCHUNK
MASK_ANS
USE_GSPO
```

本地适配了其中核心思想：

```text
old_log_prob - log_prob > threshold 时惩罚下降
可选 chunk/action-level gate
只保护指定 advantage gate 的 token/span
```

但在本地小模型设置中，LLDS 很难发挥：

1. `n_agent=2` 时 group reward 经常打平，advantage 可能全 0。
2. `positive` gate 太严格，LLDS 不触发。
3. `non_negative` gate 会保护 advantage=0 的坏轨迹。
4. chunk-level 正则对 1.5B 和小 batch 太强。
5. 如果 reward 允许“乱码后最终答对”，LLDS 可能保护坏轨迹。

这解释了两个现象：

```text
LLDS positive: actor/llds_loss = 0, preserve_ratio = 0
LLDS non_negative/chunk 强正则: val/test_score = 0
```

因此，LLDS 对本地设置不是不能用，而是需要非常小的系数和更好的 reward gate。

---

## 5. 失败样本分析

### 5.1 格式半对半错

典型输出：

```text
Answer: July 4, 1776 </answer>
```

这说明模型知道要回答，也知道有 `</answer>`，但漏了 `<answer>`。原因包括：

- SFT 只提供先验，不是硬约束。
- rollout 是采样，不是 greedy。
- 小模型容易混合自然语言 `Answer:` 和 XML tag。
- invalid feedback 后如果能修正，原 reward 可能仍给正反馈。

### 5.2 搜索后不会整理证据

典型失败：

```text
<search> query </search>
<information> ... </information>
<answer> Doc 1 </answer>
```

这说明模型把检索结果当作答案源编号，而不是抽取内容。Grounding SFT 的目的正是修复这类问题。

### 5.3 重复乱码坍塌

典型失败：

```text
rząd
rządassistant
rząduserassistant
...
Invalid action...
<answer> 12 </answer>
```

这个样本最关键，因为最终答案可能是对的，但中间过程已经坍塌。如果 reward 只看最终 EM，就会错误强化它。

因此当前代码额外加入：

```text
INVALID_ACTION_PENALTY
COLLAPSE_PENALTY
```

这不是 LLDS 原论文算法，而是针对小模型复现时出现的 reward 漏洞做的补丁。

---

## 6. Reward 设计对结果的影响

veRL 中 reward 是 Search-R1 复现成败的核心。当前 format-aware reward 在：

```text
verl/utils/reward_score/qa_em_format.py
```

原始逻辑大致是：

```text
如果最终答案 EM 正确:
    给高分
否则:
    根据格式、检索是否命中、答案是否 grounded 给 shaped reward
```

这对早期小模型训练很有帮助，因为完全 sparse EM reward 会导致大量 0 分。

但它也引入副作用：

```text
只要最终答案正确，中间过程可以很差
```

因此本地补充了：

```python
has_invalid_action_feedback()
has_repetition_collapse()
```

并在训练 reward 中扣分：

```text
base_score - invalid_action_penalty - collapse_penalty
```

注意：validation 默认仍用原始 EM，以保证和旧实验可比。

### Insight 1：Search-R1 小模型不是只缺搜索，而是缺 reward credit assignment

模型乱码后最终答对，本质是 credit assignment 错误：

```text
最终 answer 正确
-> 整条 trajectory 被当成好轨迹
-> 中间坏 token 也被间接强化
```

解决方向不是单纯调温度，而是让 reward 知道：

```text
answer correct != trajectory good
```

### Insight 2：Shaped reward 需要“正向塑形”和“负向过滤”同时存在

只给正向 shaped reward：

```text
格式分
检索分
grounding 分
```

会让模型获得学习信号，但也可能奖励坏路径。

必须配合负向过滤：

```text
invalid action penalty
collapse penalty
repetition penalty
```

否则小模型很容易学到 reward hacking。

---

## 7. Rollout 设置对复现的影响

### 7.1 `max_turns=1`

本地多次使用：

```text
max_turns=1
```

这降低了显存和时间成本，但也限制了 Search-R1 的完整能力。论文中的 Search-R1 更强调多步搜索和反思，而本地设置更接近：

```text
一次搜索 + 一次回答
```

优点：

- 适合单卡。
- 日志更短。
- 更容易定位问题。

缺点：

- 多跳问题能力弱。
- 如果首次 query 失败，难以补救。
- LLDS 的 action-level gate 可发挥空间更小。

### 7.2 `TOPK=3`

top3 相比 top1 增加 recall，但会带来：

- 更长 observation。
- 更多无关证据。
- 更强 evidence 整理压力。

因此本地加入 evidence compression：

```text
SEARCH_R1_COMPRESS_EVIDENCE=1
SEARCH_R1_EVIDENCE_SENTENCES_PER_DOC=2
SEARCH_R1_EVIDENCE_MAX_CHARS_PER_DOC=360
```

这对单卡训练很关键，否则 `max_obs_length` 会频繁截断。

### 7.3 temperature / top_p

高采样会提高探索，但小模型更容易格式漂移：

```text
temperature=0.6 -> 更易搜索，但更易乱码
temperature=0.2/0.3 -> 更稳，但探索弱
```

当前更推荐：

```text
TEMPERATURE=0.2~0.3
TOP_P=0.75~0.8
```

---

## 8. Train 设置对复现的影响

本地训练脚本：

```text
scripts/train_grpo_qwen15_bm25_shaped_1gpu.sh
```

核心配置：

```bash
TRAIN_BATCH_SIZE=2
VAL_BATCH_SIZE=2
actor_rollout_ref.rollout.n_agent=2
actor_rollout_ref.actor.ppo_micro_batch_size=1
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
```

这些配置保证了单卡可跑，但也引入统计问题：

```text
batch 小
group 小
advantage 方差大
reward 打平常见
LLDS gate 难稳定触发
```

### Insight 3：n_agent=2 是 LLDS 不稳定的重要原因

GRPO advantage 依赖组内比较。如果 `n_agent=2`，两个样本 reward 经常相同：

```text
reward: [0.1, 0.1]
advantage: [0, 0]
```

这时：

- `positive` gate 不触发。
- `non_negative` gate 又会保护 0 advantage 轨迹。

因此 LLDS 在小 batch 下会陷入两难。

更好的方向是：

```text
n_agent=3 或 4
更小 batch
更短 response
更低 max_obs_length
```

但这需要重新平衡显存。

---

## 9. 官方复现与本地复现的对比结论

### 9.1 官方 Search-R1 更依赖大模型本身能力

论文中的模型已有较强：

```text
格式遵循
阅读理解
证据抽取
多步推理
```

RL 主要是在增强搜索策略。

本地 `Qwen2.5-1.5B` 则同时缺：

```text
格式稳定性
搜索 query 质量
证据整理能力
最终答案抽取能力
抗重复坍塌能力
```

因此必须先 SFT，再 GRPO。

### 9.2 官方 LLDS 针对 policy collapse，本地还要处理 reward hacking

LLDS 解决的是：

```text
有用 action 的 likelihood 被 GRPO 压低
```

本地还遇到：

```text
坏中间过程最终答对
```

这不是 LLDS 单独能解决的，需要 reward 层过滤。

### 9.3 本地最有效改动不是 LLDS，而是 grounding SFT

目前结果显示：

```text
grounding SFT + GRPO = 0.250
LLDS adapted + penalty = 0.141
```

因此当前证据表明：

```text
小模型阶段，数据/监督对齐 > 策略正则
```

LLDS 仍值得研究，但它不是当前第一优先级。

---

## 10. 推荐后续实验路线

### 10.1 短期：稳定超过 0.250

优先从当前最好链路做小改：

```text
grounding_sft300
GRPO top3
no LLDS
soft collapse penalty
temperature 0.25
top_p 0.8
```

建议配置：

```bash
LLDS_ENABLE=false
INVALID_ACTION_PENALTY=0.3
COLLAPSE_PENALTY=0.5
TEMPERATURE=0.25
TOP_P=0.8
```

目标：

```text
保持格式稳定
减少乱码
不大幅牺牲 EM
```

### 10.2 中期：让 LLDS 真正可用

需要提高 GRPO group 信息量：

```bash
actor_rollout_ref.rollout.n_agent=3
TRAIN_BATCH_SIZE=1
MAX_RESPONSE_LENGTH=160
MAX_OBS_LENGTH=384
```

再尝试：

```bash
LLDS_ENABLE=true
LLDS_COEF=0.003
LLDS_ADV_GATE=non_negative
LLDS_CHUNK=true
INVALID_ACTION_PENALTY=0.3
COLLAPSE_PENALTY=0.5
```

核心看：

```text
actor/llds_preserve_ratio > 0
actor/llds_active_ratio > 0
actor/llds_loss > 0
val/test_score 不下降
```

### 10.3 长期：构建更强 grounding 数据

当前 grounding SFT 数据仍比较模板化。可以扩展：

1. query 改写多样化。
2. evidence selection 训练。
3. answer extraction with distractors。
4. invalid action repair SFT。
5. post-search reasoning 多样化。

特别是 invalid repair SFT：

```text
bad output:
Answer: xxx </answer>

target:
<think> I should provide the answer in the required tags. </think>
<answer> xxx </answer>
```

这比单靠 RL 纠正格式更直接。

---

## 11. 复现经验总结

### 11.1 小模型复现 Search-R1 的核心难点

不是“跑不起来”，而是：

```text
跑起来之后 reward 信号是否正确
```

如果 reward 只看最终答案，小模型会利用漏洞：

```text
乱码 -> invalid feedback -> 最终修正 -> 拿分
```

### 11.2 SFT 不是可选项，而是小模型 Search-R1 的必要前置

对 1.5B 模型来说：

```text
base -> GRPO
```

太难。必须拆成：

```text
query SFT
evidence QA SFT
decision SFT
grounding SFT
GRPO
```

### 11.3 LLDS 对小模型不是即插即用

LLDS 的论文思想是对的，但小模型复现要注意：

- 系数必须远小于官方。
- `n_agent=2` 下 advantage gate 不稳定。
- `non_negative` gate 可能保护坏样本。
- `positive` gate 可能完全不触发。
- 必须先修 reward 漏洞。

### 11.4 当前最可信的结论

本地复现已经证明：

```text
Qwen2.5-1.5B 可以通过 SFT + grounding SFT + GRPO 学到小型 Search-R1 行为。
```

但也说明：

```text
LLDS 在当前小 batch 设置下尚未带来稳定提升。
```

下一阶段应优先围绕：

```text
更好的 grounding 数据
更软的 collapse penalty
更大的 n_agent
更稳定的 reward credit assignment
```

而不是继续单纯加大 LLDS 系数。

---

## 12. 推荐保存的实验基线

必须保留：

```text
verl_checkpoints/qwen2.5-1.5b-search-lite-sft/global_step_1000
verl_checkpoints/qwen2.5-1.5b-search-lite-grounding-sft/global_step_300
verl_checkpoints/qwen2.5-1.5b-search-lite-grounding-sft/global_step_500
verl_checkpoints/nq-search-r1-qwen15-grounding-sft300-grpo-shaped-top3-400step
```

可清理：

```text
LLDS coef=0.05 结果为 0 的实验
LLDS token positive 仍乱码的实验
LLDS adapted 结果 0.141 的实验
旧 top1/top3 smoke 实验
```

核心对照表建议持续维护：

| 实验名 | 起点 | LLDS | penalty | val/test_score |
| --- | --- | --- | --- | ---: |
| baseline-shaped-top3 | grounding300 | no | no | 0.250 |
| llds-chunk-0.05 | grounding300 | yes | no | 0.000 |
| llds-adapt-0.01 | grounding300 | yes | yes | 0.141 |
| soft-penalty-no-llds | grounding300 | no | soft | 待跑 |
| n_agent3-llds-soft | grounding300 | yes | soft | 待跑 |

---

## 13. 最重要的 insight

如果只看源码，很容易认为 Search-R1 的核心是：

```text
rollout + retriever + GRPO
```

但复现过程中真正决定成败的是：

```text
模型初始行为分布
reward 是否正确归因
检索证据是否可被小模型消化
rollout 中错误轨迹是否被错误奖励
```

因此，小模型复现 Search-R1 的实际路径不是直接复刻论文，而是：

```text
先让模型会做
再让模型做稳
最后再让模型做优
```

对应到实验就是：

```text
SFT 子技能
-> grounding SFT
-> shaped GRPO
-> reward penalty
-> 小剂量 LLDS / 更大 n_agent
```

这是本次复现最核心的经验。

