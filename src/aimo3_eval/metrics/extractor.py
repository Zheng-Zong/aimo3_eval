import re
from typing import Optional

class AnswerExtractor:
    """
    负责从模型输出文本中提取最终答案。
    策略优先级：
    1. \boxed{...} (AIMO 标准格式)
    2. "The answer is ..." 后面的数字
    3. 文本中出现的最后一个数字 (Fallback，慎用，适合 Maj@k 投票)
    """

    # 预编译正则以提升速度
    BOXED_PATTERN = re.compile(r'\\boxed\s*\{(.*?)\}')
    # 匹配末尾数字 (支持整数、小数、分数，忽略千位分隔符)
    # 逻辑：寻找可能是数字的子串，允许 comma, dot, slash
    LAST_NUMBER_PATTERN = re.compile(r'([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)')

    @staticmethod
    def extract(text: str, strategy: str = "strict") -> Optional[str]:
        if not text:
            return None

        # --- 策略 1: 标准 Boxed 提取 ---
        # 寻找最后一个 \boxed{}，因为模型可能在 CoT 中间写了 boxed，最后又修正了
        matches = AnswerExtractor.BOXED_PATTERN.findall(text)
        if matches:
            candidate = matches[-1]
            return AnswerExtractor._clean_candidate(candidate)

        # 如果是 strict 模式，找不到 boxed 就直接放弃
        if strategy == "strict":
            return None

        # --- 策略 2: 关键词提取 (简易版) ---
        # 有些模型不听话，不写 boxed，但会写 "The answer is 42."
        if "answer is" in text.lower():
            part = text.lower().split("answer is")[-1]
            # 在后半部分找第一个数字
            num_match = AnswerExtractor.LAST_NUMBER_PATTERN.search(part)
            if num_match:
                return AnswerExtractor._clean_candidate(num_match.group(1))

        # --- 策略 3: 激进提取 (Fallback) ---
        # 直接找全最后出现的数字。这在 Maj@k 中很有用，因为只要有一个是对的就行。
        all_nums = AnswerExtractor.LAST_NUMBER_PATTERN.findall(text)
        if all_nums:
            return AnswerExtractor._clean_candidate(all_nums[-1])

        return None

    @staticmethod
    def _clean_candidate(text: str) -> str:
        """清洗提取出的字符串，去除 LaTeX 干扰符"""
        # 去除常见的单位和货币符号
        for garbage in ["$", "£", "€", "\\text", "{", "}", "cm", "kg", "units"]:
            text = text.replace(garbage, "")
        # 去除逗号 (1,000 -> 1000)
        text = text.replace(",", "").strip()
        # 处理结尾句号
        if text.endswith("."):
            text = text[:-1]
        return text