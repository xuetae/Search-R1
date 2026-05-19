#!/usr/bin/env python
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

# -*- coding: utf-8 -*-
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""File-system agnostic IO APIs"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import tempfile
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import hashlib

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .hdfs_io import copy, makedirs, exists

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ["copy", "exists", "makedirs"]

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_HDFS_PREFIX = "hdfs://"


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _is_non_local(path):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return path.startswith(_HDFS_PREFIX)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def md5_encode(path: str) -> str:
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return hashlib.md5(path.encode()).hexdigest()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Return a local temp path that joins cache_dir and basename of hdfs_path

    Args:
        hdfs_path:
        cache_dir:

    Returns:

    """
    # make a base64 encoding of hdfs_path to avoid directory conflict
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    encoded_hdfs_path = md5_encode(hdfs_path)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    os.makedirs(temp_dir, exist_ok=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dst


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def copy_local_path_from_hdfs(src: str, cache_dir=None, filelock='.file.lock', verbose=False) -> str:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Copy src from hdfs to local if src is on hdfs or directly return src.
    If cache_dir is None, we will use the default cache dir of the system. Note that this may cause conflicts if
    the src name is the same between calls

    Args:
        src (str): a HDFS path of a local path

    Returns:
        a local path of the copied file
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from filelock import FileLock

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert src[-1] != '/', f'Make sure the last char in src is not / because it will cause error. Got {src}'

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if _is_non_local(src):
        # download from hdfs to local
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if cache_dir is None:
            # get a temp folder
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_dir = tempfile.gettempdir()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.makedirs(cache_dir, exist_ok=True)
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert os.path.exists(cache_dir)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_path = get_local_temp_path(src, cache_dir)
        # get a specific lock
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        filelock = md5_encode(src) + '.lock'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lock_file = os.path.join(cache_dir, filelock)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with FileLock(lock_file=lock_file):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not os.path.exists(local_path):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if verbose:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    print(f'Copy from {src} to {local_path}')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                copy(src, local_path)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return local_path
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return src
