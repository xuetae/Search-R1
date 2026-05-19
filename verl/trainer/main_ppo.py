# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
PPO训练主脚本
=============
功能：使用PPO（近端策略优化）算法训练LLM
流程：
1. 收集轨迹数据（使用当前策略生成回应）
2. 计算奖励（基于任务或RM模型）
3. 计算优势估计
4. 更新策略网络
5. 重复直到收敛

PPO是一种流行的强化学习算法，在LLM微调中广泛使用

注意：我们不将main与ray_trainer组合，因为ray_trainer被其他main使用
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto  # veRL数据协议
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch张量库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.reward_score import qa_em  # QA精确匹配评分
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # Ray PPO训练器
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re  # 正则表达式
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np  # NumPy数组

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _select_rm_score_fn(data_source):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    根据数据源选择相应的奖励计算函数
    
    参数：
        data_source: 数据集名称（nq、hotpotqa等）
        
    返回：
        计算奖励的函数
    """
    # 对于QA数据集，使用精确匹配作为奖励
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return qa_em.compute_score_em  # 返回精确匹配计算函数
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError  # 其他数据集未实现


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RewardManager():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    奖励管理器
    ==========
    计算样本的奖励分数
    
    支持两种奖励来源：
    1. 预计算的RM分数（如果数据中包含）
    2. 基于规则的分数计算（如精确匹配）
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化奖励管理器
        
        参数：
            tokenizer: 分词器（用于解码）
            num_examine: 要打印到控制台的解码响应的批次数
            format_score: 格式分数奖励（0-1之间的浮点数）
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = tokenizer  # 保存分词器
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_examine = num_examine  # 要检查的样本数
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.format_score = format_score  # 格式奖励权重

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        计算数据批次的奖励
        
        参数：
            data: DataProto对象，包含提示和响应
            
        返回：
            奖励张量 [batch_size, response_len]
        """

        # 如果已经包含RM分数，直接返回
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if 'rm_scores' in data.batch.keys():
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return data.batch['rm_scores']

        # 否则，创建零初始化的奖励张量
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        already_print_data_sources = {}

        # 逐个样本处理
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(len(data)):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_item = data[i]  # 获取单个样本

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_ids = data_item.batch['prompts']

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_length = prompt_ids.shape[-1]

            # 计算提示的有效长度（排除填充）
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # 获取响应
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_ids = data_item.batch['responses']
            # 计算响应的有效长度
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_response_ids = response_ids[:valid_response_length]

            # 组合提示和响应
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # 解码为字符串
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sequences_str = self.tokenizer.decode(sequences)

            # 获取真值标签
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # 根据数据源选择计分函数
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_source = data_item.non_tensor_batch['data_source']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            compute_score_fn = _select_rm_score_fn(data_source)

            # 计算分数
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            score = compute_score_fn(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                solution_str=sequences_str, 
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                ground_truth=ground_truth, 
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                format_score=self.format_score
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 将分数放在响应末尾
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reward_tensor[i, valid_response_length - 1] = score

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return reward_tensor
            # all_scores.append(score)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return reward_tensor


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import hydra


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def main(config):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not ray.is_initialized():
        # this is for local ray cluster
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ray.get(main_task.remote(config))


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@ray.remote
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def main_task(config):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.utils.fs import copy_local_path_from_hdfs
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from transformers import AutoTokenizer

    # print initial config
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from pprint import pprint
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from omegaconf import OmegaConf
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.utils import hf_tokenizer
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.single_controller.ray import RayWorkerGroup
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ray_worker_group_cls = RayWorkerGroup

    # 中文注释：下一行继续判断其他条件分支。
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError

    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    role_worker_mapping = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Role.Critic: ray.remote(CriticWorker),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global_pool_id = 'global_pool'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    resource_pool_spec = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mapping = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Role.ActorRollout: global_pool_id,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Role.Critic: global_pool_id,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Role.RefPolicy: global_pool_id,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.reward_model.enable:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if config.reward_model.strategy == 'fsdp':
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.fsdp_workers import RewardModelWorker
        # 中文注释：下一行继续判断其他条件分支。
        elif config.reward_model.strategy == 'megatron':
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.megatron_workers import RewardModelWorker
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mapping[Role.RewardModel] = global_pool_id

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    trainer = RayPPOTrainer(config=config,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            tokenizer=tokenizer,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            role_worker_mapping=role_worker_mapping,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            resource_pool_manager=resource_pool_manager,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            ray_worker_group_cls=ray_worker_group_cls,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            reward_fn=reward_fn,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            val_reward_fn=val_reward_fn,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            )
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    trainer.init_workers()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    trainer.fit()


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    main()
