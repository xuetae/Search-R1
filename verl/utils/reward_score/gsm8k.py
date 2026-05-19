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
import re


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def extract_solution(solution_str, method='strict'):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert method in ['strict', 'flexible']

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if method == 'strict':
        # this also tests the formatting of the model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if solution is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            final_answer = None
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            final_answer = solution.group(0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    # 中文注释：下一行继续判断其他条件分支。
    elif method == 'flexible':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_answer = None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(answer) == 0:
            # no reward is there is no answer
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pass
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            invalid_str = ['', '.']
            # find the last number that is not '.'
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for final_answer in reversed(answer):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if final_answer not in invalid_str:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    break
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return final_answer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    answer = extract_solution(solution_str=solution_str, method=method)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if answer is None:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return 0
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if answer == ground_truth:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return score
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return format_score