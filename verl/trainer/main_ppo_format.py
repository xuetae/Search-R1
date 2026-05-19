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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.reward_score import qa_em, qa_em_format
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _select_rm_score_fn(data_source):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return qa_em_format.compute_score_em
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RewardManager():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """The reward manager.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.format_score = format_score
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.structure_format_score = structure_format_score
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.final_format_score = final_format_score
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_score = retrieval_score

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if 'rm_scores' in data.batch.keys():
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return data.batch['rm_scores']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        already_print_data_sources = {}

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(len(data)):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_item = data[i]  # DataProtoItem

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_ids = data_item.batch['prompts']

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_length = prompt_ids.shape[-1]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_ids = data_item.batch['responses']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sequences_str = self.tokenizer.decode(sequences)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_source = data_item.non_tensor_batch['data_source']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            compute_score_fn = _select_rm_score_fn(data_source)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     structure_format_score=self.structure_format_score, 
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     final_format_score=self.final_format_score, 
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     retrieval_score=self.retrieval_score,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     format_score=self.format_score)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if data_source not in already_print_data_sources:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                already_print_data_sources[data_source] = 0

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if already_print_data_sources[data_source] < self.num_examine:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                already_print_data_sources[data_source] += 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(sequences_str)

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
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              structure_format_score=config.reward_model.structure_format_score, 
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              final_format_score=config.reward_model.final_format_score,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              retrieval_score=config.reward_model.retrieval_score)

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
