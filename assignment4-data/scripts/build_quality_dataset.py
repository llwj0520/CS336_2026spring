import gzip
import json
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.language import identify_language
from cs336_data.quality import gopher_quality_filter


POSITIVE_PATH = Path(
    "local-shared-data/wiki/quality_positive.jsonl"
)
WARC_PATH = Path(
    "local-shared-data/CC/example-first-20MiB.warc.gz"
)

OUTPUT_DIRECTORY = Path(
    "local-shared-data/quality"
)
NEGATIVE_PATH = OUTPUT_DIRECTORY / "quality_negative.jsonl"
TRAIN_PATH = OUTPUT_DIRECTORY / "quality_train.txt"
VALID_PATH = OUTPUT_DIRECTORY / "quality_valid.txt"

RANDOM_SEED = 336
TRAIN_FRACTION = 0.8


def prepare_fasttext_line(label: str, text: str) -> str:
    """把一篇文档转换成 fastText 的单行训练格式。"""
    single_line_text = " ".join(text.split())
    return f"__label__{label} {single_line_text}\n"


rng = random.Random(RANDOM_SEED)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


# 读取 Wikipedia 引用正例
positive_examples = []

with POSITIVE_PATH.open(encoding="utf-8") as input_file:
    for line in input_file:
        record = json.loads(line)
        text = record["text"].strip()

        if text:
            positive_examples.append(text)


# 从 Common Crawl 中收集英语且通过 Gopher 规则的作为负例。
negative_records = []

with gzip.open(WARC_PATH, "rb") as warc_file:
    records = ArchiveIterator(
        warc_file,
        record_types=WarcRecordType.response,
    )

    try:
        for record in records:
            text = extract_text_from_html_bytes(
                record.reader.read()
            ).strip()

            if not text:
                continue

            language, confidence = identify_language(text)
            if language != "en" or confidence < 0.7:
                continue

            if not gopher_quality_filter(text):
                continue

            negative_records.append(
                {
                    "url": (
                        record.headers.get("WARC-Target-URI")
                        or ""
                    ),
                    "text": text,
                }
            )

    except (EOFError, OSError):
        # 截断 WARC 文件的最后一条记录可能不完整。
        pass


# 正负样本取相同数量，防止类别不平衡。
rng.shuffle(positive_examples)
rng.shuffle(negative_records)

example_count = min(
    len(positive_examples),
    len(negative_records),
)

positive_examples = positive_examples[:example_count]
negative_records = negative_records[:example_count]
negative_examples = [
    record["text"]
    for record in negative_records
]


# 保存负例，便于后续检查。
with NEGATIVE_PATH.open("w", encoding="utf-8") as output_file:
    for record in negative_records:
        output_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# 分别切分正例和负例，确保训练集、验证集都保持平衡。
train_count_per_class = int(
    example_count * TRAIN_FRACTION
)

train_examples = []
valid_examples = []

for label, examples in (
    ("wiki", positive_examples),
    ("cc", negative_examples),
):
    train_texts = examples[:train_count_per_class]
    valid_texts = examples[train_count_per_class:]

    train_examples.extend(
        (label, text)
        for text in train_texts
    )
    valid_examples.extend(
        (label, text)
        for text in valid_texts
    )


rng.shuffle(train_examples)
rng.shuffle(valid_examples)


with TRAIN_PATH.open("w", encoding="utf-8") as output_file:
    for label, text in train_examples:
        output_file.write(
            prepare_fasttext_line(label, text)
        )


with VALID_PATH.open("w", encoding="utf-8") as output_file:
    for label, text in valid_examples:
        output_file.write(
            prepare_fasttext_line(label, text)
        )


print(f"可用 Wikipedia 正例：{len(positive_examples)}")
print(f"可用 Common Crawl 负例：{len(negative_examples)}")
print(f"每类训练样本：{train_count_per_class}")
print(
    "每类验证样本："
    f"{example_count - train_count_per_class}"
)
print(f"训练集总数：{len(train_examples)}")
print(f"验证集总数：{len(valid_examples)}")
print(f"训练文件：{TRAIN_PATH}")
print(f"验证文件：{VALID_PATH}")