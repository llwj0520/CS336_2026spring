"""使用 GPT-2 tokenizer 序列化最终训练数据。"""

from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


INPUT_PATH = Path(
    "local-shared-data/filtered/filtered_data.txt"
)
OUTPUT_PATH = Path(
    "local-shared-data/filtered/filtered_data_gpt2.bin"
)


tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    model_max_length=1_000_000_000,
)

all_token_ids = []
document_count = 0


with INPUT_PATH.open(encoding="utf-8") as input_file:
    for line in tqdm(
        input_file,
        desc="Tokenizing documents",
    ):
        text = line.strip()

        if not text:
            continue

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        # 每篇文档末尾加入 GPT-2 的结束标记。
        token_ids.append(tokenizer.eos_token_id)

        all_token_ids.extend(token_ids)
        document_count += 1


# GPT-2 的 token ID 小于 65536，可以安全存为 uint16。
ids_array = np.asarray(
    all_token_ids,
    dtype=np.uint16,
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

ids_array.tofile(OUTPUT_PATH)


print(f"分词文档数：{document_count}")
print(f"Token 总数：{len(ids_array):,}")
print(f"EOS token ID：{tokenizer.eos_token_id}")
print(f"输出文件：{OUTPUT_PATH}")
print(
    f"输出大小："
    f"{OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MiB"
)