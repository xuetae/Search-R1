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
A unified tracking interface that supports logging data to different backend
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import dataclasses
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from enum import Enum
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from functools import partial
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pathlib import Path
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Union, Dict, Any


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Tracking(object):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    supported_backend = ['wandb', 'mlflow', 'console']

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(default_backend, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            default_backend = [default_backend]
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for backend in default_backend:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if backend == 'tracking':
                # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
                import warnings
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert backend in self.supported_backend, f'{backend} is not supported'

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.logger = {}

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if 'tracking' in default_backend or 'wandb' in default_backend:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import wandb
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import os
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if WANDB_API_KEY:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                wandb.login(key=WANDB_API_KEY)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            wandb.init(project=project_name, name=experiment_name, config=config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.logger['wandb'] = wandb

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if 'mlflow' in default_backend:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import mlflow
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mlflow.start_run(run_name=experiment_name)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if 'console' in default_backend:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.utils.logger.aggregate_logger import LocalLogger
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.console_logger = LocalLogger(print_to_console=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.logger['console'] = self.console_logger

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def log(self, data, step, backend=None):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for default_backend, logger_instance in self.logger.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if backend is None or default_backend in backend:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logger_instance.log(data=data, step=step)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class _MlflowLoggingAdapter:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def log(self, data, step):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import mlflow
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mlflow.log_metrics(metrics=data, step=step)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if params is None:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {}

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if dataclasses.is_dataclass(x):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return _transform(dataclasses.asdict(x))
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(x, dict):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {k: _transform(v) for k, v in x.items()}
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(x, list):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if convert_list_to_dict:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return [_transform(v) for v in x]
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(x, Path):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return str(x)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(x, Enum):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return x.value

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import pandas as pd
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(ans, dict)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return ans
