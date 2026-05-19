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
项目设置脚本
============
功能：定义项目的安装配置
当pyproject.toml不能正常工作时使用此脚本
"""

# setup.py是当pyproject.toml不工作时的备用安装脚本
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from setuptools import setup, find_packages
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os

# 获取项目根目录（setup.py所在目录）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# 读取版本号
# 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
with open(os.path.join(version_folder, 'verl/version/version')) as f:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    __version__ = f.read().strip()


# 读取依赖包列表
# 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
with open('requirements.txt') as f:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    required = f.read().splitlines()
    # 过滤掉空行和注释行（以#开头）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    install_requires = [item.strip() for item in required if item.strip()[0] != '#']

# 可选的额外依赖
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
extras_require = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'test': ['pytest', 'yapf']  # 测试和代码格式化工具
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}

# 读取README作为长描述
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pathlib import Path
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
this_directory = Path(__file__).parent
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
long_description = (this_directory / "README.md").read_text()

# 配置设置
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
setup(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    name='verl',  # 包名
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    version=__version__,  # 版本号
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    package_dir={'': '.'},  # 包目录
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    packages=find_packages(where='.'),  # 自动发现所有包
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    url='https://github.com/volcengine/verl',  # 项目URL
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    license='Apache 2.0',  # 许可证
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    author='Bytedance - Seed - MLSys',  # 作者
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    author_email='zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk',  # 作者邮件
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    description='veRL: Volcano Engine Reinforcement Learning for LLM',  # 项目简描
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    install_requires=install_requires,  # 必需的依赖包
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    extras_require=extras_require,  # 可选依赖
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    package_data={
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        '': ['version/*'],  # 包含版本文件
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'verl': ['trainer/config/*.yaml'],  # 包含配置文件
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    },
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    include_package_data=True,  # 包含package_data中指定的文件
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    long_description=long_description,  # 长描述（README内容）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    long_description_content_type='text/markdown'  # 长描述的格式
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
)