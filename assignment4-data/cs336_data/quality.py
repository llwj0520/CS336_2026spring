"""基于 Gopher 规则过滤低质量文档。"""

import re
from functools import lru_cache

import fasttext

from cs336_data.common import get_shared_assets_path


WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
QUALITY_CHUNK_WORDS = 200


@lru_cache(maxsize=1)
def load_quality_model():
    """加载质量分类模型，并在进程中缓存。"""
    model_path = (
        get_shared_assets_path()
        / "classifiers"
        / "quality_classifier.bin"
    )
    return fasttext.load_model(str(model_path))


def classify_quality(text: str) -> tuple[str, float]:
    """分块预测文档质量，并汇总为标签和置信度。"""
    words = text.split()
    if not words:
        return "cc", 1.0

    model = load_quality_model()
    wiki_scores = []

    for start in range(0, len(words), QUALITY_CHUNK_WORDS):
        chunk = " ".join(words[start : start + QUALITY_CHUNK_WORDS])
        labels, scores = model.predict(chunk, k=2)
        score_by_label = dict(zip(labels, scores, strict=True))
        wiki_scores.append(
            float(score_by_label.get("__label__wiki", 0.0))
        )

    mean_wiki_score = sum(wiki_scores) / len(wiki_scores)
    mean_wiki_score = min(1.0, max(0.0, mean_wiki_score))

    if mean_wiki_score >= 0.5:
        return "wiki", mean_wiki_score

    return "cc", 1.0 - mean_wiki_score


def gopher_quality_filter(text: str) -> bool:
    """判断文本是否通过 Gopher 质量规则。"""

    # \w+ 会提取由字母、数字或下划线组成的词，
    # 同时排除单独的标点符号。
    words = WORD_PATTERN.findall(text)
    word_count = len(words)

    # 规则 1：词数必须在 50 到 100000 之间。
    if word_count < 50 or word_count > 100_000:
        return False

    # 规则 2：平均词长必须在 3 到 10 个字符之间。
    mean_word_length = sum(len(word) for word in words) / word_count
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # 规则 3：以省略号结尾的行不能超过全部非空行的 30%。
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(line.endswith("...") for line in lines)
        if ellipsis_lines / len(lines) > 0.30:
            return False

    # 规则 4：至少 80% 的词必须包含字母。
    alphabetic_words = sum(
        any(character.isalpha() for character in word)
        for word in words
    )
    if alphabetic_words / word_count < 0.80:
        return False

    return True
