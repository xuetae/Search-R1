# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import random
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ast
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import operator


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def extract_solution(solution_str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "Assistant:" in solution_str:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        solution_str = solution_str.split("Assistant:", 1)[1]
    # 中文注释：下一行继续判断其他条件分支。
    elif "<|im_start|>assistant" in solution_str:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    solution_str = solution_str.split('\n')[-1]

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
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return final_answer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def validate_equation(equation_str, available_numbers):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Validate that equation only uses available numbers and each number once."""
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # Extract all numbers from the equation
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        available_numbers = sorted(available_numbers)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return numbers_in_eq == available_numbers
    # 中文注释：下一行处理异常分支，保证错误可控。
    except:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def evaluate_equation(equation_str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not re.match(allowed_pattern, equation_str):
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        result = eval(equation_str, {"__builtins__": None}, {})
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return result
    # 中文注释：下一行处理异常分支，保证错误可控。
    except Exception as e:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    target = ground_truth['target']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    numbers = ground_truth['numbers']
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    equation = extract_solution(solution_str=solution_str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    do_print = random.randint(1, 64) == 1
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if do_print:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"--------------------------------")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Target: {target} | Numbers: {numbers}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Extracted equation: {equation}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"Solution string: {solution_str}")

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if equation is None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if do_print:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f"No equation found")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return 0
    
    # Validate equation uses correct numbers
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not validate_equation(equation, numbers):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if do_print:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f"Invalid equation")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return format_score
        
    # Evaluate equation
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        result = evaluate_equation(equation)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if result is None:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if do_print:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f"Could not evaluate equation")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return format_score
            
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if abs(result - target) < 1e-5:  # Account for floating point precision
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if do_print:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f"Correct equation: {equation} = {result}")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return score
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if do_print:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f"Wrong result: equation = {result}, target = {target}")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return format_score
    # 中文注释：下一行处理异常分支，保证错误可控。
    except:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if do_print:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f"Error evaluating equation")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return format_score 