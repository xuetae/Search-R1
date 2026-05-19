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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def set_basic_config(level):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=level)
