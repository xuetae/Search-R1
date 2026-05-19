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
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import string
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import random

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def normalize_answer(s):
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def remove_articles(text):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return re.sub(r"\b(a|an|the)\b", " ", text)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def white_space_fix(text):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return " ".join(text.split())

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def remove_punc(text):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        exclude = set(string.punctuation)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return "".join(ch for ch in text if ch not in exclude)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def lower(text):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return text.lower()

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def em_check(prediction, golden_answers):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(golden_answers, str):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        golden_answers = [golden_answers]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    normalized_prediction = normalize_answer(prediction)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    score = 0
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for golden_answer in golden_answers:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        golden_answer = normalize_answer(golden_answer)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if golden_answer == normalized_prediction:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            score = 1
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            break
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return score


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def subem_check(prediction, golden_answers):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(golden_answers, str):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        golden_answers = [golden_answers]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    normalized_prediction = normalize_answer(prediction)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    score = 0
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for golden_answer in golden_answers:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        golden_answer = normalize_answer(golden_answer)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if golden_answer in normalized_prediction:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            score = 1
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            break
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return score


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def extract_solution(solution_str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    answer_pattern = r'<answer>(.*?)</answer>'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(matches) <= 1:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None
    
    # If there are 2 or more matches, return the last one
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return matches[-1].group(1).strip()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    answer = extract_solution(solution_str=solution_str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    do_print = random.randint(1, 64) == 1
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if do_print:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"--------------------------------")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Golden answers: {ground_truth['target']}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Extracted answer: {answer}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Solution string: {solution_str}")
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if answer is None:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return 0
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if em_check(answer, ground_truth['target']):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return score
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return format_score


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    answer = extract_solution(solution_str=solution_str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    do_print = random.randint(1, 64) == 1
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if do_print:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"--------------------------------")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Golden answers: {ground_truth['target']}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Extracted answer: {answer}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Solution string: {solution_str}")
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if answer is None:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return 0
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if subem_check(answer, ground_truth['target']):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return score
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return format_score
