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
def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    assistant_match = re.search(assistant_pattern, text)
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not assistant_match:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    start_pos = assistant_match.end()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    content = text[start_pos:]
    
    # Check for balanced tags
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tags_to_check = ["think", "search", "information", "answer"]
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for tag in tags_to_check:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        opening_count = len(re.findall(f"<{tag}>", content))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        closing_count = len(re.findall(f"</{tag}>", content))
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if opening_count != closing_count:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i, part in enumerate(parts):
        # Skip empty parts
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not part.strip():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
            
        # Check if this is a tag
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if part == "<think>" and state in ["start", "information"]:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "in_think"
            # 中文注释：下一行继续判断其他条件分支。
            elif part == "</think>" and state == "in_think":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "after_think"
            # 中文注释：下一行继续判断其他条件分支。
            elif part == "<search>" and state == "after_think":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "in_search"
            # 中文注释：下一行继续判断其他条件分支。
            elif part == "</search>" and state == "in_search":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "after_search"
            # 中文注释：下一行继续判断其他条件分支。
            elif part == "<information>" and state == "after_search":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "in_information"
            # 中文注释：下一行继续判断其他条件分支。
            elif part == "</information>" and state == "in_information":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "information"
            # 中文注释：下一行继续判断其他条件分支。
            elif part == "<answer>" and state == "after_think":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "in_answer"
            # 中文注释：下一行继续判断其他条件分支。
            elif part == "</answer>" and state == "in_answer":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state = "end"
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return False, f"Unexpected tag {part} in state {state}"
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # This is content, check if it's valid in the current state
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pass
            # 中文注释：下一行继续判断其他条件分支。
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if part.strip():
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if state != "end":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False, f"Incomplete sequence, ended in state {state}"
        
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return True, "Valid sequence format"


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def extract_solution(solution_str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Extract the equation from the solution string."""

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
def extract_information_blocks(text: str) -> list[str]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pattern = r"<information>(.*?)</information>"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    matches = re.findall(pattern, text, re.DOTALL)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return [match.strip() for match in matches]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seqs = extract_information_blocks(text)
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for seq in seqs:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for golden_answer in golden_answers:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if normalize_answer(golden_answer) in normalize_answer(seq):
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return True
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return False


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
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
    is_valid_format, _ = is_valid_sequence(solution_str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retrieval_correct = False
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if is_valid_format:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
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
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if is_valid_format:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if retrieval_correct:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return structure_format_score + retrieval_score # 0.3
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return structure_format_score # 0.2
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return 0
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if em_check(answer, ground_truth['target']):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_valid_format:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return score # 1
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return score - structure_format_score # 0.8
        # 中文注释：下一行继续判断其他条件分支。
        elif is_valid_format:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if retrieval_correct:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return structure_format_score + retrieval_score # 0.3
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return structure_format_score # 0.2
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return final_format_score # 0.1
