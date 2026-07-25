import gzip
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.pii import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    IPV4_PATTERN,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
)


WARC_PATH = "local-shared-data/CC/example-first-20MiB.warc.gz"
SAMPLE_SIZE = 20
RANDOM_SEED = 336


rng = random.Random(RANDOM_SEED)
samples = []
documents_with_pii = 0


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

            # 先记录正则实际匹配了哪些字符串。
            emails = EMAIL_PATTERN.findall(text)
            phones = PHONE_PATTERN.findall(text)
            ips = IPV4_PATTERN.findall(text)

            masked_text, email_count = mask_emails(text)
            masked_text, phone_count = mask_phone_numbers(masked_text)
            masked_text, ip_count = mask_ips(masked_text)

            total_count = email_count + phone_count + ip_count
            if total_count == 0:
                continue

            documents_with_pii += 1
            url = record.headers.get("WARC-Target-URI") or ""

            # 将掩码附近的内容截取出来，方便人工检查。
            single_line_text = " ".join(masked_text.split())
            placeholder_positions = [
                single_line_text.find("|||EMAIL_ADDRESS|||"),
                single_line_text.find("|||PHONE_NUMBER|||"),
                single_line_text.find("|||IP_ADDRESS|||"),
            ]
            placeholder_positions = [
                position
                for position in placeholder_positions
                if position >= 0
            ]

            first_position = min(placeholder_positions)
            start = max(0, first_position - 180)
            end = first_position + 420
            context = single_line_text[start:end]

            item = {
                "record_index": record_index,
                "url": url,
                "emails": emails,
                "phones": phones,
                "ips": ips,
                "email_count": email_count,
                "phone_count": phone_count,
                "ip_count": ip_count,
                "context": context,
            }

            # 对所有发生过替换的文档进行蓄水池抽样。
            if len(samples) < SAMPLE_SIZE:
                samples.append(item)
            else:
                replacement_index = rng.randrange(documents_with_pii)
                if replacement_index < SAMPLE_SIZE:
                    samples[replacement_index] = item

    except (EOFError, OSError):
        # WARC 只下载了前 20 MiB，允许最后一条记录不完整。
        pass


print(f"包含至少一个 PII 匹配的文档数：{documents_with_pii}")
print(f"随机样本数：{len(samples)}")


for sample_number, sample in enumerate(samples, start=1):
    print()
    print("=" * 80)
    print(f"样本：{sample_number}")
    print(f"WARC 记录编号：{sample['record_index']}")
    print(f"URL：{sample['url']}")
    print(f"匹配邮箱：{sample['emails']}")
    print(f"匹配电话：{sample['phones']}")
    print(f"匹配 IP：{sample['ips']}")
    print(
        "替换数量："
        f"email={sample['email_count']}, "
        f"phone={sample['phone_count']}, "
        f"ip={sample['ip_count']}"
    )
    print(f"掩码后上下文：{sample['context']}")