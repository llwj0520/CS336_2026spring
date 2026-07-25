import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://en.wikipedia.org/w/api.php"
TARGET_URL_COUNT = 600

OUTPUT_PATH = Path(
    "local-shared-data/wiki/subsampled_positive_urls.txt"
)

SKIPPED_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".zip",
    ".gz",
    ".pdf",
)


def query_random_external_links() -> set[str]:
    """随机选择 Wikipedia 条目，并返回其中的外部链接。"""
    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": "50",
        "prop": "extlinks",
        "ellimit": "max",
    }

    query = urllib.parse.urlencode(parameters)
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={
            "User-Agent": (
                "cs336-quality-classifier/1.0 "
                "(local educational project)"
            )
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)

    urls = set()

    for page in data.get("query", {}).get("pages", []):
        for link_data in page.get("extlinks", []):
            # formatversion=2 通常使用 url；保留 * 作为兼容处理。
            url = link_data.get("url") or link_data.get("*")
            if not url:
                continue

            parsed = urllib.parse.urlparse(url)

            if parsed.scheme not in {"http", "https"}:
                continue

            hostname = (parsed.hostname or "").lower()
            if any(
                domain in hostname
                for domain in (
                    "wikipedia.org",
                    "wikimedia.org",
                    "wikidata.org",
                )
            ):
                continue

            if parsed.path.lower().endswith(SKIPPED_SUFFIXES):
                continue

            # 同一页面带不同 #fragment 时只保留一份。
            clean_url = parsed._replace(fragment="").geturl()
            urls.add(clean_url)

    return urls


all_urls: set[str] = set()
attempt = 0

while len(all_urls) < TARGET_URL_COUNT and attempt < 30:
    attempt += 1

    try:
        new_urls = query_random_external_links()
        all_urls.update(new_urls)
        print(
            f"第 {attempt} 次请求完成，"
            f"当前共有 {len(all_urls)} 个不同链接"
        )
    except Exception as error:
        print(f"第 {attempt} 次请求失败：{error}")

    time.sleep(0.5)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

selected_urls = sorted(all_urls)[:TARGET_URL_COUNT]

with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
    for url in selected_urls:
        output_file.write(url + "\n")

print(f"最终保存 {len(selected_urls)} 个 URL")
print(f"输出文件：{OUTPUT_PATH}")