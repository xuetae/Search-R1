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
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import shutil
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = logging.getLogger(__file__)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_HDFS_PREFIX = "hdfs://"

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_HDFS_BIN_PATH = shutil.which('hdfs')


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def exists(path: str, **kwargs) -> bool:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""Works like os.path.exists() but supports hdfs.

    Test whether a path exists. Returns False for broken symbolic links.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path exists, False otherwise
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if _is_non_local(path):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return _exists(path, **kwargs)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return os.path.exists(path)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _exists(file_path: str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """ hdfs capable to check whether a file_path is exists """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if file_path.startswith("hdfs"):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return os.path.exists(file_path)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _mkdir(name, **kwargs)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _mkdir(file_path: str) -> bool:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """hdfs mkdir"""
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if file_path.startswith("hdfs"):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.makedirs(file_path, exist_ok=True)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return True


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def copy(src: str, dst: str, **kwargs) -> bool:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if _is_non_local(src) or _is_non_local(dst):
        # TODO(haibin.lin):
        # - handle SameFileError for hdfs files(?)
        # - return file destination for hdfs files
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return _copy(src, dst)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if os.path.isdir(src):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return shutil.copytree(src, dst, **kwargs)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return shutil.copy(src, dst, **kwargs)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if to_path.startswith("hdfs"):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if from_path.startswith("hdfs"):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if from_path.startswith("hdfs"):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            returncode = _run_cmd(_hdfs_cmd(f"-get \
                {from_path} {to_path}"), timeout=timeout)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
            try:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                shutil.copy(from_path, to_path)
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                returncode = 0
            # 中文注释：下一行处理异常分支，保证错误可控。
            except shutil.SameFileError:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                returncode = 0
            # 中文注释：下一行处理异常分支，保证错误可控。
            except Exception as e:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                returncode = -1
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return returncode == 0


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _run_cmd(cmd: str, timeout=None):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return os.system(cmd)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _hdfs_cmd(cmd: str) -> str:
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _is_non_local(path: str):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return path.startswith(_HDFS_PREFIX)
