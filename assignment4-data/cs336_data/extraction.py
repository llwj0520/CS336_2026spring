"""从原始 HTML 字节中提取纯文本。"""

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """将 HTML 字节解码，并提取其中的可见文本。"""
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes) or "utf-8"
        html = html_bytes.decode(encoding, errors="replace")

    return extract_plain_text(html)