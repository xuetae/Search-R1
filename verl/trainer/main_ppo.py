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

from verl import DataProto  # veRL数据协议
import torch  # PyTorch张量库
from verl.utils.reward_score import qa_em  # QA精确匹配评分
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # Ray PPO训练器
import re  # 正则表达式
import numpy as np  # NumPy数组

def _select_rm_score_fn(data_source):
    """
    根据数据源选择相应的奖励计算函数
    
    参数：
        data_source: 数据集名称（nq、hotpotqa等）
        
    返回：
        计算奖励的函数
    """
    # 对于QA数据集，使用精确匹配作为奖励
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em  # 返回精确匹配计算函数
    else:
        raise NotImplementedError  # 其他数据集未实现


class RewardManager():
    """
    奖励管理器
    ==========
    计算样本的奖励分数
    
    支持两种奖励来源：
    1. 预计算的RM分数（如果数据中包含）
    2. 基于规则的分数计算（如精确匹配）
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        """
        初始化奖励管理器
        
        参数：
            tokenizer: 分词器（用于解码）
            num_examine: 要打印到控制台的解码响应的批次数
            format_score: 格式分数奖励（0-1之间的浮点数）
        """
        self.tokenizer = tokenizer  # 保存分词器
        self.num_examine = num_examine  # 要检查的样本数
        self.format_score = format_score  # 格式奖励权重

    def __call__(self, data: DataProto):
        """
        计算数据批次的奖励
        
        参数：
            data: DataProto对象，包含提示和响应
            
        返回：
            奖励张量 [batch_size, response_len]
        """

        # 如果已经包含RM分数，直接返回
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # 否则，创建零初始化的奖励张量
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        # 逐个样本处理
        for i in range(len(data)):
            data_item = data[i]  # 获取单个样本

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            # 计算提示的有效长度（排除填充）
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # 获取响应
            response_ids = data_item.batch['responses']
            # 计算响应的有效长度
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 组合提示和响应
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # 解码为字符串
            sequences_str = self.tokenizer.decode(sequences)

            # 获取真值标签
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # 根据数据源选择计分函数
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # 计算分数
            score = compute_score_fn(
                solution_str=sequences_str, 
                ground_truth=ground_truth, 
                format_score=self.format_score
            )

            # 将分数放在响应末尾
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
