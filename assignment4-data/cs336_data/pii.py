import re

#用正则表达式找出邮箱。
EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])"
    r"[\w.+-]+"
    r"@"
    r"[A-Za-z0-9-]+"
    r"(?:\.[A-Za-z0-9-]+)+"
    r"(?![\w-])"
)

#用 re.subn 替换所有匹配项，并同时统计数量。
def mask_emails(text: str) -> tuple[str, int]:
    return EMAIL_PATTERN.subn(
        "|||EMAIL_ADDRESS|||",
        text,
    )


#正则化拆分电话号码
PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[ .-]?)?"
    r"(?:\([ \t]*\d{3}[ \t]*\)|\d{3})"
    r"[ .-]?"
    r"\d{3}"
    r"[ .-]?"
    r"\d{4}"
    r"(?!\d)"
)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_PATTERN.subn(
        "|||PHONE_NUMBER|||",
        text,
    )


IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"

IPV4_PATTERN = re.compile(
    rf"(?<![\d.])"
    rf"{IPV4_OCTET}(?:\.{IPV4_OCTET}){{3}}"
    rf"(?!\d|\.\d)"
)


def mask_ips(text: str) -> tuple[str, int]:
    return IPV4_PATTERN.subn(
        "|||IP_ADDRESS|||",
        text,
    )