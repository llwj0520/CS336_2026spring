"""检测网页中的有害内容。"""

from functools import lru_cache

import fasttext

from cs336_data.common import get_shared_assets_path


@lru_cache(maxsize=1)
def load_nsfw_model():
    """加载 NSFW 分类模型，并在进程中缓存。"""
    model_path = (
        get_shared_assets_path()
        / "classifiers"
        / "dolma_fasttext_nsfw_jigsaw_model.bin"
    )
    return fasttext.load_model(str(model_path))


def classify_nsfw(text: str) -> tuple[str, float]:
    """判断文本是否包含 NSFW 内容。"""
    model = load_nsfw_model()

    # fastText 不接受换行符，所以先转换为单行文本。
    single_line_text = text.replace("\n", " ").strip()
    if not single_line_text:
        single_line_text = " "

    labels, scores = model.predict(single_line_text, k=1)
    
    #去掉 __label__ 前缀
    label = labels[0].removeprefix("__label__")
    confidence = float(scores[0])
    confidence = min(1.0, max(0.0, confidence))

    return label, confidence



@lru_cache(maxsize=1)
def load_toxic_speech_model():
    """加载有毒言论分类模型，并在进程中缓存。"""
    model_path = (
        get_shared_assets_path()
        / "classifiers"
        / "dolma_fasttext_hatespeech_jigsaw_model.bin"
    )
    return fasttext.load_model(str(model_path))


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """判断文本是否包含有毒言论。"""
    model = load_toxic_speech_model()

    single_line_text = text.replace("\n", " ").strip()
    if not single_line_text:
        single_line_text = " "

    labels, scores = model.predict(single_line_text, k=1)

    label = labels[0].removeprefix("__label__")
    confidence = float(scores[0])
    confidence = min(1.0, max(0.0, confidence))

    return label, confidence