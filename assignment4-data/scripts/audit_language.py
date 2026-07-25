import gzip
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.language import identify_language


WARC_PATH = "local-shared-data/CC/example-first-20MiB.warc.gz"
SAMPLE_SIZE = 20
RANDOM_SEED = 336


rng = random.Random(RANDOM_SEED)
samples: list[tuple[int, str, str]] = []
seen_documents = 0


with gzip.open(WARC_PATH, "rb") as warc_file:
    records = ArchiveIterator(
        warc_file,
        record_types=WarcRecordType.response,
    )

    try:
        for record_index, record in enumerate(records, start=1):
            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes).strip()

            if not text:
                continue

            url = record.headers.get("WARC-Target-URI") or ""
            seen_documents += 1
            item = (record_index, url, text)

            # 蓄水池抽样：在不知道总记录数时，均匀保留 20 条。
            if len(samples) < SAMPLE_SIZE:
                samples.append(item)
            else:
                replacement_index = rng.randrange(seen_documents)
                if replacement_index < SAMPLE_SIZE:
                    samples[replacement_index] = item

    except (EOFError, OSError):
        # 文件只有完整 WARC 的前 20 MiB，末尾记录可能被截断。
        pass


print(f"完整读取的非空网页数量：{seen_documents}")
print(f"随机样本数量：{len(samples)}")


for sample_number, (record_index, url, text) in enumerate(samples, start=1):
    language, confidence = identify_language(text)
    excerpt = " ".join(text.split())[:300]

    print()
    print("=" * 80)
    print(f"样本：{sample_number}")
    print(f"WARC 记录编号：{record_index}")
    print(f"URL：{url}")
    print(f"预测语言：{language}")
    print(f"置信度：{confidence:.4f}")
    print(f"文本片段：{excerpt}")