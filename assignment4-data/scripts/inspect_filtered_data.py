import json
import random
from pathlib import Path


INPUT_PATH = Path(
    "local-shared-data/filtered/filtered_documents.jsonl"
)

SAMPLE_SIZE = 5
RANDOM_SEED = 336


documents = []

with INPUT_PATH.open(encoding="utf-8") as input_file:
    for line in input_file:
        documents.append(json.loads(line))


rng = random.Random(RANDOM_SEED)
samples = rng.sample(documents, SAMPLE_SIZE)


print(f"最终数据集文档数：{len(documents)}")
print(f"随机样本数：{len(samples)}")


for sample_number, document in enumerate(samples, start=1):
    excerpt = " ".join(
        document["text"].split()
    )[:700]

    print()
    print("=" * 80)
    print(f"保留样本：{sample_number}")
    print(f"URL：{document['url']}")
    print(f"文本片段：{excerpt}")