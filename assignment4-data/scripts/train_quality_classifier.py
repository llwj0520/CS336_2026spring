from pathlib import Path

import fasttext


TRAIN_PATH = Path(
    "local-shared-data/quality/quality_train_chunks.txt"
)
VALID_PATH = Path(
    "local-shared-data/quality/quality_valid_chunks.txt"
)
MODEL_PATH = Path(
    "local-shared-data/classifiers/quality_classifier.bin"
)

model = fasttext.train_supervised(
    input=str(TRAIN_PATH),
    lr=0.3,
    epoch=25,
    dim=50,
    wordNgrams=2,
    minCount=1,
    bucket=100_000,
    loss="softmax",
    thread=4,
)


example_count, precision, recall = model.test(
    str(VALID_PATH)
)

print(f"验证样本数：{example_count}")
print(f"Precision@1：{precision:.4f}")
print(f"Recall@1：{recall:.4f}")
print(f"模型标签：{model.get_labels()}")


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(MODEL_PATH))

print(f"模型已保存到：{MODEL_PATH}")
