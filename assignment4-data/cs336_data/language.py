from functools import lru_cache

import fasttext

from cs336_data.common import get_shared_assets_path

#模型只加载一次，否则每次判断语言都重新读取 125 MB 模型。
@lru_cache(maxsize=1)
def load_language_model():
    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    return fasttext.load_model(str(model_path))


def identify_language(text: str) -> tuple[str, float]:
    model = load_language_model()

    single_line_text = text.replace("\n", " ").strip()
    if not single_line_text:
        single_line_text = " "

    labels, scores = model.predict(single_line_text, k=1)
    
    #取第一个预测。
    language = labels[0].removeprefix("__label__")
    confidence = float(scores[0])
    confidence = min(1.0, max(0.0, confidence))

    return language, confidence