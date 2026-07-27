"""过滤 Common Crawl WET 文档。"""

import gzip
from collections.abc import Iterator
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.language import identify_language
from cs336_data.quality import gopher_quality_filter
from cs336_data.harmful import (
    classify_nsfw,
    classify_toxic_speech,
)
from cs336_data.quality import (
    classify_quality,
    gopher_quality_filter,
)
import json

from cs336_data.pii import (
    mask_emails,
    mask_ips,
    mask_phone_numbers,
)


INPUT_PATH = Path(
    "local-shared-data/CC/example.warc.wet.gz"
)
OUTPUT_DIRECTORY = Path(
    "local-shared-data/filtered"
)
OUTPUT_JSONL_PATH = (
    OUTPUT_DIRECTORY / "filtered_documents.jsonl"
)
OUTPUT_TEXT_PATH = (
    OUTPUT_DIRECTORY / "filtered_data.txt"
)
HARMFUL_CONFIDENCE_THRESHOLD = 0.8
QUALITY_REJECTION_THRESHOLD = 0.60

#把 WET 压缩文件中的网页一篇一篇读出来，确认后续流水线能够取得网页文本和 URL
def iter_wet_documents(input_path: Path,) -> Iterator[tuple[str, str]]:
    """依次产生 WET 文档的 URL 和纯文本。"""
    with gzip.open(input_path, "rb") as input_file:
        #只遍历网页正文记录,跳过文件头等元数据记录
        records = ArchiveIterator(
            input_file,
            record_types=WarcRecordType.conversion,
        )

        for record in records:
            #读取网页文本
            text_bytes = record.reader.read()
            #变成普通字符串
            text = text_bytes.decode(
                "utf-8",
                errors="replace",
            ).strip()

            if not text:
                continue
            
            #读取网页 URL
            url = (
                record.headers.get("WARC-Target-URI")
                or ""
            )

            yield url, text


total_documents = 0
english_documents = 0
removed_by_language = 0
passed_gopher= 0
removed_by_gopher = 0
removed_by_nsfw = 0
removed_by_toxic = 0
passed_harmful = 0
removed_by_quality = 0
passed_quality = 0
email_replacements = 0
phone_replacements = 0
ip_replacements = 0

filtered_documents = []

for url, text in iter_wet_documents(INPUT_PATH):
    total_documents += 1

    language, confidence = identify_language(text)
    
    #判断是否是英文的
    if language != "en" or confidence < 0.7:
        removed_by_language += 1
        continue

    english_documents += 1
    
    #判断是否通过gopher
    if not gopher_quality_filter(text):
        removed_by_gopher += 1
        continue

    passed_gopher += 1
    
    nsfw_label, nsfw_confidence = classify_nsfw(text)
    toxic_label, toxic_confidence = classify_toxic_speech(text)
    
    #进行有害/劣质的内容过滤（之前定义的nsfw/toxic）
    if (
        nsfw_label == "nsfw"
        and nsfw_confidence >= HARMFUL_CONFIDENCE_THRESHOLD
    ):
        removed_by_nsfw += 1
        continue

    if (
        toxic_label == "toxic"
        and toxic_confidence >= HARMFUL_CONFIDENCE_THRESHOLD
    ):
        removed_by_toxic += 1
        continue

    passed_harmful += 1
    
    #通过质量分类器，删除来源于cc且对劣质的置信度高于阈值的example
    quality_label, quality_confidence = classify_quality(text)

    if (
        quality_label == "cc"
        and quality_confidence >= QUALITY_REJECTION_THRESHOLD
    ):
        removed_by_quality += 1
        continue

    passed_quality += 1

    #统计并替换保留下来的文档汇总的个人信息，处理后保存为可以后续继续使用的文件
    masked_text, email_count = mask_emails(text)
    masked_text, phone_count = mask_phone_numbers(
        masked_text
    )
    masked_text, ip_count = mask_ips(masked_text)

    email_replacements += email_count
    phone_replacements += phone_count
    ip_replacements += ip_count
    
    #保存过滤结果
    filtered_documents.append(
        {
            "url": url,
            "text": masked_text,
        }
    )

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_JSONL_PATH.open(
    "w",
    encoding="utf-8",
) as jsonl_file:
    with OUTPUT_TEXT_PATH.open(
        "w",
        encoding="utf-8",
    ) as text_file:
        for document in filtered_documents:
            jsonl_file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # 每篇文档写成一行，方便后续 GPT-2 分词。
            single_line_text = " ".join(
                document["text"].split()
            )
            text_file.write(single_line_text + "\n")

print(
    f"Gopher 规则删除：{removed_by_gopher} "
    f"（占全部文档 "
    f"{removed_by_gopher / total_documents:.2%}，"
    f"占英语文档 "
    f"{removed_by_gopher / english_documents:.2%}）"
)
print(
    f"通过语言和 Gopher 过滤：{passed_gopher} "
    f"({passed_gopher / total_documents:.2%})"
)
print(
    f"NSFW 过滤删除：{removed_by_nsfw} "
    f"({removed_by_nsfw / passed_gopher:.2%} "
    f"of Gopher-passed)"
)
print(
    f"Toxic 过滤删除：{removed_by_toxic} "
    f"({removed_by_toxic / passed_gopher:.2%} "
    f"of Gopher-passed)"
)
print(
    f"通过有害内容过滤：{passed_harmful} "
    f"({passed_harmful / total_documents:.2%} "
    f"of all documents)"
)
print(
    f"质量分类器删除：{removed_by_quality} "
    f"({removed_by_quality / passed_harmful:.2%} "
    f"of harmful-filter-passed)"
)
print(
    f"通过全部丢弃型过滤器：{passed_quality} "
    f"({passed_quality / total_documents:.2%} "
    f"of all documents)"
)
print(f"邮箱替换数量：{email_replacements}")
print(f"电话替换数量：{phone_replacements}")
print(f"IP 地址替换数量：{ip_replacements}")
print(f"最终保存文档数量：{len(filtered_documents)}")
print(f"JSONL 输出：{OUTPUT_JSONL_PATH}")
print(f"训练文本输出：{OUTPUT_TEXT_PATH}")