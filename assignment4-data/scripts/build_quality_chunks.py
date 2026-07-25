import random
from pathlib import Path


FULL_TRAIN_PATH = Path(
    "local-shared-data/quality/quality_train.txt"
)
FULL_VALID_PATH = Path(
    "local-shared-data/quality/quality_valid.txt"
)

CHUNK_TRAIN_PATH = Path(
    "local-shared-data/quality/quality_train_chunks.txt"
)
CHUNK_VALID_PATH = Path(
    "local-shared-data/quality/quality_valid_chunks.txt"
)

CHUNK_WORDS = 200
MIN_CHUNK_WORDS = 50
RANDOM_SEED = 336


def split_into_chunks(text: str) -> list[str]:
    """把文档切成最多包含 200 个词的文本块。"""
    words = text.split()
    chunks = []

    for start in range(0, len(words), CHUNK_WORDS):
        chunk_words = words[start : start + CHUNK_WORDS]

        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))

    return chunks


def build_balanced_chunk_file(
    input_path: Path,
    output_path: Path,
    rng: random.Random,
) -> tuple[int, int]:
    chunks_by_label = {
        "wiki": [],
        "cc": [],
    }

    with input_path.open(encoding="utf-8") as input_file:
        for line in input_file:
            label_token, text = line.rstrip("\n").split(
                " ",
                maxsplit=1,
            )
            label = label_token.removeprefix("__label__")

            chunks_by_label[label].extend(
                split_into_chunks(text)
            )

    balanced_count = min(
        len(chunks_by_label["wiki"]),
        len(chunks_by_label["cc"]),
    )

    output_examples = []

    for label in ("wiki", "cc"):
        rng.shuffle(chunks_by_label[label])

        for chunk in chunks_by_label[label][:balanced_count]:
            output_examples.append((label, chunk))

    rng.shuffle(output_examples)

    with output_path.open("w", encoding="utf-8") as output_file:
        for label, chunk in output_examples:
            output_file.write(
                f"__label__{label} {chunk}\n"
            )

    return balanced_count, len(output_examples)


rng = random.Random(RANDOM_SEED)

train_per_class, train_total = build_balanced_chunk_file(
    FULL_TRAIN_PATH,
    CHUNK_TRAIN_PATH,
    rng,
)
valid_per_class, valid_total = build_balanced_chunk_file(
    FULL_VALID_PATH,
    CHUNK_VALID_PATH,
    rng,
)

print(f"每类训练文本块：{train_per_class}")
print(f"训练文本块总数：{train_total}")
print(f"每类验证文本块：{valid_per_class}")
print(f"验证文本块总数：{valid_total}")
