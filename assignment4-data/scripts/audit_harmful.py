import gzip
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.harmful import classify_nsfw, classify_toxic_speech


WARC_PATH = "local-shared-data/CC/example-first-20MiB.warc.gz"
SAMPLE_SIZE = 20
RANDOM_SEED = 336


rng = random.Random(RANDOM_SEED)
samples = []

total_documents = 0
nsfw_documents = 0
toxic_documents = 0
harmful_documents = 0


with gzip.open(WARC_PATH, "rb") as warc_file:
    records = ArchiveIterator(
        warc_file,
        record_types=WarcRecordType.response,
    )

    try:
        for record_index, record in enumerate(records, start=1):
            text = extract_text_from_html_bytes(
                record.reader.read()
            ).strip()

            if not text:
                continue

            nsfw_label, nsfw_confidence = classify_nsfw(text)
            toxic_label, toxic_confidence = classify_toxic_speech(text)

            is_nsfw = nsfw_label == "nsfw"
            is_toxic = toxic_label == "toxic"
            is_harmful = is_nsfw or is_toxic

            total_documents += 1
            nsfw_documents += int(is_nsfw)
            toxic_documents += int(is_toxic)
            harmful_documents += int(is_harmful)

            url = record.headers.get("WARC-Target-URI") or ""
            excerpt = " ".join(text.split())[:500]

            item = {
                "record_index": record_index,
                "url": url,
                "nsfw_label": nsfw_label,
                "nsfw_confidence": nsfw_confidence,
                "toxic_label": toxic_label,
                "toxic_confidence": toxic_confidence,
                "excerpt": excerpt,
            }

            # 从所有非空网页中均匀随机抽取 20 篇。
            if len(samples) < SAMPLE_SIZE:
                samples.append(item)
            else:
                replacement_index = rng.randrange(total_documents)
                if replacement_index < SAMPLE_SIZE:
                    samples[replacement_index] = item

    except (EOFError, OSError):
        # 文件只有完整 WARC 的前 20 MiB，最后一条记录可能不完整。
        pass


print(f"非空文档总数：{total_documents}")
print(
    f"预测为 NSFW：{nsfw_documents} "
    f"({nsfw_documents / total_documents:.2%})"
)
print(
    f"预测为 toxic：{toxic_documents} "
    f"({toxic_documents / total_documents:.2%})"
)
print(
    f"至少一种有害标签：{harmful_documents} "
    f"({harmful_documents / total_documents:.2%})"
)
print(f"随机样本数：{len(samples)}")


for sample_number, sample in enumerate(samples, start=1):
    print()
    print("=" * 80)
    print(f"样本：{sample_number}")
    print(f"WARC 记录编号：{sample['record_index']}")
    print(f"URL：{sample['url']}")
    print(
        f"NSFW：{sample['nsfw_label']}，"
        f"置信度={sample['nsfw_confidence']:.4f}"
    )
    print(
        f"有毒言论：{sample['toxic_label']}，"
        f"置信度={sample['toxic_confidence']:.4f}"
    )
    print(f"文本片段：{sample['excerpt']}")





