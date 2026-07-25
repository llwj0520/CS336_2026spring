import gzip
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.quality import WORD_PATTERN, gopher_quality_filter


WARC_PATH = "local-shared-data/CC/example-first-20MiB.warc.gz"
SAMPLE_SIZE = 20
RANDOM_SEED = 336


rng = random.Random(RANDOM_SEED)
samples = []

total_documents = 0
passed_documents = 0


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

            passed = gopher_quality_filter(text)

            words = WORD_PATTERN.findall(text)
            word_count = len(words)

            if words:
                mean_word_length = (
                    sum(len(word) for word in words) / word_count
                )
                alphabetic_fraction = (
                    sum(
                        any(character.isalpha() for character in word)
                        for word in words
                    )
                    / word_count
                )
            else:
                mean_word_length = 0.0
                alphabetic_fraction = 0.0

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]
            if lines:
                ellipsis_fraction = (
                    sum(line.endswith("...") for line in lines)
                    / len(lines)
                )
            else:
                ellipsis_fraction = 0.0

            total_documents += 1
            passed_documents += int(passed)

            item = {
                "record_index": record_index,
                "url": record.headers.get("WARC-Target-URI") or "",
                "passed": passed,
                "word_count": word_count,
                "mean_word_length": mean_word_length,
                "ellipsis_fraction": ellipsis_fraction,
                "alphabetic_fraction": alphabetic_fraction,
                "excerpt": " ".join(text.split())[:500],
            }

            # 从全部非空文档中均匀随机抽取 20 篇。
            if len(samples) < SAMPLE_SIZE:
                samples.append(item)
            else:
                replacement_index = rng.randrange(total_documents)
                if replacement_index < SAMPLE_SIZE:
                    samples[replacement_index] = item

    except (EOFError, OSError):
        # 截断的 WARC 文件可能包含不完整的最后一条记录。
        pass


failed_documents = total_documents - passed_documents

print(f"非空文档总数：{total_documents}")
print(
    f"通过质量过滤：{passed_documents} "
    f"({passed_documents / total_documents:.2%})"
)
print(
    f"未通过质量过滤：{failed_documents} "
    f"({failed_documents / total_documents:.2%})"
)
print(f"随机样本数：{len(samples)}")


for sample_number, sample in enumerate(samples, start=1):
    print()
    print("=" * 80)
    print(f"样本：{sample_number}")
    print(f"WARC 记录编号：{sample['record_index']}")
    print(f"URL：{sample['url']}")
    print(f"过滤结果：{'通过' if sample['passed'] else '未通过'}")
    print(f"词数：{sample['word_count']}")
    print(f"平均词长：{sample['mean_word_length']:.2f}")
    print(f"省略号行比例：{sample['ellipsis_fraction']:.2%}")
    print(f"含字母词比例：{sample['alphabetic_fraction']:.2%}")
    print(f"文本片段：{sample['excerpt']}")