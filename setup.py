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
项目设置脚本
============
功能：定义项目的安装配置
当pyproject.toml不能正常工作时使用此脚本
"""

# setup.py是当pyproject.toml不工作时的备用安装脚本
from setuptools import setup, find_packages
import os

# 获取项目根目录（setup.py所在目录）
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# 读取版本号
with open(os.path.join(version_folder, 'verl/version/version')) as f:
    __version__ = f.read().strip()


# 读取依赖包列表
with open('requirements.txt') as f:
    required = f.read().splitlines()
    # 过滤掉空行和注释行（以#开头）
    install_requires = [item.strip() for item in required if item.strip()[0] != '#']

# 可选的额外依赖
extras_require = {
    'test': ['pytest', 'yapf']  # 测试和代码格式化工具
}

# 读取README作为长描述
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# 配置设置
setup(
    name='verl',  # 包名
    version=__version__,  # 版本号
    package_dir={'': '.'},  # 包目录
    packages=find_packages(where='.'),  # 自动发现所有包
    url='https://github.com/volcengine/verl',  # 项目URL
    license='Apache 2.0',  # 许可证
    author='Bytedance - Seed - MLSys',  # 作者
    author_email='zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk',  # 作者邮件
    description='veRL: Volcano Engine Reinforcement Learning for LLM',  # 项目简描
    install_requires=install_requires,  # 必需的依赖包
    extras_require=extras_require,  # 可选依赖
    package_data={
        '': ['version/*'],  # 包含版本文件
        'verl': ['trainer/config/*.yaml'],  # 包含配置文件
    },
    include_package_data=True,  # 包含package_data中指定的文件
    long_description=long_description,  # 长描述（README内容）
    long_description_content_type='text/markdown'  # 长描述的格式
)