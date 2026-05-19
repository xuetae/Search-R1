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
The base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from abc import ABC, abstractmethod
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, List, Union

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['HybridEngineBaseTokenizer']


# 中文注释：下一行定义类，用于组织相关状态与行为。
class HybridEngineBaseTokenizer(ABC):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """the tokenizer property and function name should align with HF's to meet vllm requirement"""

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def vocab_size(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pad_token_id(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def eos_token_id(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def all_special_ids(self) -> List[int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def all_special_tokens(self) -> List[str]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def encode(self, text):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers.

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def decode(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        skip_special_tokens: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        clean_up_tokenization_spaces: bool = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **kwargs,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> str:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def convert_ids_to_tokens(self,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              ids: Union[int, List[int]],
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              skip_special_tokens: bool = False) -> Union[str, List[str]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_added_vocab(self) -> Dict[str, int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def is_fast(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False
