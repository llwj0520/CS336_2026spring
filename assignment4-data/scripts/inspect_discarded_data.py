import gzip
import json
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.harmful import (
    classify_nsfw,
    classify_toxic_speech,
)
from cs336_data.language import identify_language
from cs336_data.quality import (
    classify_quality,
    gopher_quality_filter,
)


WET_PATH = Path(
    "local-shared-data/CC/example.warc.wet.gz"
)
FILTERED_PATH = Path(
    "local-shared-data/filtered/filtered_documents.jsonl"
)

SAMPLE_SIZE = 5
RANDOM_SEED = 336
HARMFUL_THRESHOLD = 0.8
QUALITY_THRESHOLD = 0.65


def iter_wet_documents():
    with gzip.open(WET_PATH, "rb") as input_file:
        records = ArchiveIterator(
            input_file,
            record_types=WarcRecordType.conversion,
        )

        for record in records:
            text = record.reader.read().decode(
                "utf-8",
                errors="replace",
            ).strip()

            if not text:
                continue

            url = (
                record.headers.get("WARC-Target-URI")
                or ""
            )

            yield url, text


def get_removal_reason(text: str) -> str:
    """返回文档遇到的第一个删除原因。"""
    language, language_confidence = identify_language(text)

    if language != "en" or language_confidence < 0.7:
        return (
            "语言过滤："
            f"{language=}, "
            f"confidence={language_confidence:.4f}"
        )

    if not gopher_quality_filter(text):
        return "Gopher 质量规则"

    nsfw_label, nsfw_confidence = classify_nsfw(text)
    if (
        nsfw_label == "nsfw"
        and nsfw_confidence >= HARMFUL_THRESHOLD
    ):
        return (
            "NSFW 过滤："
            f"confidence={nsfw_confidence:.4f}"
        )

    toxic_label, toxic_confidence = classify_toxic_speech(text)
    if (
        toxic_label == "toxic"
        and toxic_confidence >= HARMFUL_THRESHOLD
    ):
        return (
            "Toxic 过滤："
            f"confidence={toxic_confidence:.4f}"
        )

    quality_label, quality_confidence = classify_quality(text)
    if (
        quality_label == "cc"
        and quality_confidence >= QUALITY_THRESHOLD
    ):
        return (
            "质量分类器："
            f"label={quality_label}, "
            f"confidence={quality_confidence:.4f}"
        )

    return "未找到删除原因"


# 读取最终保留文档的 URL。
kept_urls = set()

with FILTERED_PATH.open(encoding="utf-8") as input_file:
    for line in input_file:
        record = json.loads(line)
        kept_urls.add(record["url"])


# 从未保留的原始文档中进行蓄水池抽样。
rng = random.Random(RANDOM_SEED)
discarded_samples = []
discarded_seen = 0

for url, text in iter_wet_documents():
    if url in kept_urls:
        continue

    discarded_seen += 1
    item = {
        "url": url,
        "text": text,
    }

    if len(discarded_samples) < SAMPLE_SIZE:
        discarded_samples.append(item)
    else:
        replacement_index = rng.randrange(discarded_seen)

        if replacement_index < SAMPLE_SIZE:
            discarded_samples[replacement_index] = item


print(f"未保留文档数量：{discarded_seen}")
print(f"随机样本数量：{len(discarded_samples)}")


for sample_number, document in enumerate(
    discarded_samples,
    start=1,
):
    reason = get_removal_reason(document["text"])
    excerpt = " ".join(document["text"].split())[:700]

    print()
    print("=" * 80)
    print(f"丢弃样本：{sample_number}")
    print(f"URL：{document['url']}")
    print(f"删除原因：{reason}")
    print(f"文本片段：{excerpt}")