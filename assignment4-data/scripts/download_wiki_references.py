import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.language import identify_language
from cs336_data.quality import gopher_quality_filter


URL_PATH = Path(
    "local-shared-data/wiki/subsampled_positive_urls.txt"
)
OUTPUT_PATH = Path(
    "local-shared-data/wiki/quality_positive.jsonl"
)

MAX_DOWNLOAD_BYTES = 2_000_000
MAX_WORKERS = 16


def download_page(url: str) -> tuple[str, bytes | None]:
    """下载一个 HTML 页面；失败时返回 None。"""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "CS336-quality-classifier/1.0"
            )
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=8,
        ) as response:
            content_type = (
                response.headers.get("Content-Type", "")
                .lower()
            )

            allowed_types = (
                "text/html",
                "application/xhtml+xml",
                "text/plain",
            )
            if not any(
                allowed_type in content_type
                for allowed_type in allowed_types
            ):
                return url, None

            content = response.read(MAX_DOWNLOAD_BYTES)
            return url, content

    except Exception:
        return url, None


urls = [
    line.strip()
    for line in URL_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

downloaded_count = 0
english_count = 0
passed_count = 0

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_page, url)
            for url in urls
        ]

        for completed_count, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            url, html_bytes = future.result()

            if html_bytes is None:
                continue

            downloaded_count += 1

            try:
                text = extract_text_from_html_bytes(
                    html_bytes
                ).strip()
            except Exception:
                continue

            if not text:
                continue

            language, confidence = identify_language(text)
            if language != "en" or confidence < 0.7:
                continue

            english_count += 1

            if not gopher_quality_filter(text):
                continue

            passed_count += 1

            record = {
                "url": url,
                "text": text,
            }
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            if completed_count % 25 == 0:
                print(
                    f"已处理 {completed_count}/{len(urls)}，"
                    f"当前保留 {passed_count} 篇"
                )


print()
print(f"候选 URL：{len(urls)}")
print(f"成功下载文本页面：{downloaded_count}")
print(f"英语页面：{english_count}")
print(f"最终保留的高质量正例：{passed_count}")
print(f"输出文件：{OUTPUT_PATH}")