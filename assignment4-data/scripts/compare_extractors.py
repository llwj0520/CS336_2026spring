import gzip

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extraction import extract_text_from_html_bytes


warc_path = "local-shared-data/CC/example-first-20MiB.warc.gz"
wet_path = "local-shared-data/CC/example.warc.wet.gz"


# 读取 WARC 中的第一个 HTML 响应。
with gzip.open(warc_path, "rb") as warc_file:
    records = ArchiveIterator(
        warc_file,
        record_types=WarcRecordType.response,
    )
    first_response = next(records)
    url = first_response.headers.get("WARC-Target-URI")
    html_bytes = first_response.reader.read()

our_text = extract_text_from_html_bytes(html_bytes)


# 读取对应 WET 文件中的第一条纯文本记录。
with gzip.open(wet_path, "rb") as wet_file:
    records = ArchiveIterator(
        wet_file,
        record_types=WarcRecordType.conversion,
    )
    first_conversion = next(records)
    wet_text = first_conversion.reader.read().decode(
        "utf-8",
        errors="replace",
    )


print(f"URL: {url}")
print(f"Our extractor: {len(our_text)} characters")
print(f"WET extractor: {len(wet_text)} characters")

print("\n===== OUR EXTRACTOR =====")
print(our_text[:1500])

print("\n===== WET EXTRACTOR =====")
print(wet_text[:1500])