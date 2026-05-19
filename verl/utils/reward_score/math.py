# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_score(solution_str, ground_truth) -> float:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retval = 0.
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        string_in_last_boxed = last_boxed_only_string(solution_str)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if string_in_last_boxed is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            answer = remove_boxed(string_in_last_boxed)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_equiv(answer, ground_truth):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                retval = 1.
    # 中文注释：下一行处理异常分支，保证错误可控。
    except Exception as e:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(e)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def is_equiv(str1, str2, verbose=False):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if str1 is None and str2 is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print("WARNING: Both None")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return True
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if str1 is None or str2 is None:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False

    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ss1 = strip_string(str1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ss2 = strip_string(str2)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if verbose:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(ss1, ss2)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return ss1 == ss2
    # 中文注释：下一行处理异常分支，保证错误可控。
    except Exception:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return str1 == str2


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def remove_boxed(s):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "\\boxed " in s:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        left = "\\boxed "
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert s[:len(left)] == left
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return s[len(left):]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    left = "\\boxed{"

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert s[:len(left)] == left
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert s[-1] == "}"

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return s[len(left):-1]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def last_boxed_only_string(string):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    idx = string.rfind("\\boxed")
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "\\boxed " in string:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if idx < 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idx = string.rfind("\\fbox")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if idx < 0:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return None

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    i = idx
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    right_brace_idx = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_left_braces_open = 0
    # 中文注释：下一行开始循环，直到条件不再满足。
    while i < len(string):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if string[i] == "{":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_left_braces_open += 1
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if string[i] == "}":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_left_braces_open -= 1
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if num_left_braces_open == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                right_brace_idx = i
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                break
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        i += 1

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if right_brace_idx is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retval = None
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retval = string[idx:right_brace_idx + 1]

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return retval


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def fix_fracs(string):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    substrs = string.split("\\frac")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    new_str = substrs[0]
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(substrs) > 1:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        substrs = substrs[1:]
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for substr in substrs:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_str += "\\frac"
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if substr[0] == "{":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                new_str += substr
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
                try:
                    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                    assert len(substr) >= 2
                # 中文注释：下一行处理异常分支，保证错误可控。
                except AssertionError:
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return string
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                a = substr[0]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                b = substr[1]
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if b != "{":
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if len(substr) > 2:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        post_substr = substr[2:]
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    # 中文注释：下一行处理前面条件都不满足时的默认分支。
                    else:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        new_str += "{" + a + "}{" + b + "}"
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if len(substr) > 2:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        post_substr = substr[2:]
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        new_str += "{" + a + "}" + b + post_substr
                    # 中文注释：下一行处理前面条件都不满足时的默认分支。
                    else:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        new_str += "{" + a + "}" + b
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = new_str
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return string


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def fix_a_slash_b(string):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(string.split("/")) != 2:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return string
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    a = string.split("/")[0]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    b = string.split("/")[1]
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        a = int(a)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        b = int(b)
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert string == "{}/{}".format(a, b)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return new_string
    # 中文注释：下一行处理异常分支，保证错误可控。
    except AssertionError:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return string


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "\\text{ " in string:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        splits = string.split("\\text{ ")
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(splits) == 2
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return splits[0]
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return string


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def fix_sqrt(string):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "\\sqrt" not in string:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return string
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splits = string.split("\\sqrt")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    new_string = splits[0]
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for split in splits[1:]:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if split[0] != "{":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            a = split[0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_substr = "\\sqrt" + split
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_string += new_substr
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return new_string


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def strip_string(string):
    # linebreaks
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("\n", "")

    # remove inverse spaces
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("tfrac", "frac")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("\\left", "")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("^{\\circ}", "")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("^\\circ", "")

    # remove dollar signs
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("\\$", "")

    # remove units (on the right)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = remove_right_units(string)

    # remove percentage
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("\\%", "")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace(" .", " 0.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(string) == 0:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return string
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if string[0] == ".":
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(string.split("=")) == 2:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(string.split("=")[0]) <= 2:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = fix_sqrt(string)

    # remove spaces
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if string == "0.5":
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    string = fix_a_slash_b(string)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return string
