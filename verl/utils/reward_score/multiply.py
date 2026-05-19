# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import random


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "Assistant:" in solution_str:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        solution_str = solution_str.split("Assistant:", 1)[1]
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    answer_pattern = r'<answer>(.*?)</answer>'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    match = re.finditer(answer_pattern, solution_str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    matches = list(match)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if matches:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_answer = matches[-1].group(1).strip()
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_answer = None
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if final_answer is not None:
        # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
        try:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            int_final_answer = int(final_answer)
        # 中文注释：下一行处理异常分支，保证错误可控。
        except ValueError:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            final_answer = None
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return final_answer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
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
    answer = extract_solution(solution_str=solution_str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    do_print = random.randint(1, 64) == 1
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if do_print:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"--------------------------------")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Ground truth: {ground_truth} | Extracted answer: {answer}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Solution string: {solution_str}")

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if answer is None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if do_print:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f"No answer found")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return 0
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if int(answer) == int(ground_truth):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if do_print:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f"Correct answer: {answer}")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return score
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if do_print:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f"Incorrect answer {answer} | Ground truth: {ground_truth}")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return format_score
