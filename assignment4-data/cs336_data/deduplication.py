"""文本数据去重。"""

import hashlib
import os
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import mmh3
from xopen import xopen

def _hash_line(line: str) -> bytes:
    """把一行文本转换成固定长度的 128 位哈希。"""
    normalized_line = line.rstrip("\r\n")

    return hashlib.blake2b(
        normalized_line.encode("utf-8"),
        digest_size=16,
    ).digest()


def exact_line_deduplication(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
) -> None:
    input_paths = [Path(path) for path in input_files]
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    line_counts: Counter[bytes] = Counter()

    """第一遍
        读取全部文件
        → 计算每一行的哈希
        → 统计每个哈希出现次数"""
    for input_path in input_paths:
        with xopen(
            input_path,
            mode="rt",
            encoding="utf-8",
            errors="replace",
        ) as input_file:
            for line in input_file:
                line_hash = _hash_line(line)
                line_counts[line_hash] += 1
    
    """
    第二遍
    再次读取全部文件
    → 查询这一行的出现次数
    → 次数等于 1：保留
    → 次数大于 1：删除
    """
    for input_path in input_paths:
        destination = output_path / input_path.name

        with xopen(
            input_path,
            mode="rt",
            encoding="utf-8",
            errors="replace",
        ) as input_file:
            with xopen(
                destination,
                mode="wt",
                encoding="utf-8",
            ) as output_file:
                for line in input_file:
                    line_hash = _hash_line(line)

                    if line_counts[line_hash] == 1:
                        output_file.write(line)

def _normalize_text(text: str) -> str:
    """规范化文本，以提高近似重复检测的召回率。"""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)

    # NFD 会把 é 分解为 e 和重音符号；
    # Unicode 类别 Mn 表示非间距组合符号。
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )

    # 将标点替换为空格，避免删除标点后把两个词粘在一起。
    text = "".join(
        " "
        if unicodedata.category(character).startswith("P")
        else character
        for character in text
    )

    # 将连续空格、换行和制表符统一成一个空格。
    return " ".join(text.split())

#把文章拆成小片段
def _word_ngrams(text: str, n: int) -> set[str]:
    """创建由连续 n 个单词组成的 n-gram 集合。"""
    if n <= 0:
        raise ValueError("n must be positive")

    normalized_text = _normalize_text(text)
    words = normalized_text.split()

    if len(words) < n:
        # 用整篇短文本作为一个特征，避免所有短文档都得到相同空集合。
        return {normalized_text} if normalized_text else set()

    return {
        " ".join(words[start : start + n])
        for start in range(len(words) - n + 1)
    }

#对词片段生成 MinHash 数字指纹，减少内存的占用
def _minhash_signature(
    ngrams: set[str],
    num_hashes: int,
) -> tuple[int, ...]:
    """根据 n-gram 集合生成 MinHash 签名。"""
    if not ngrams:
        return tuple(
            2**32 - 1
            for _ in range(num_hashes)
        )

    signature = []
    
    #不同的 seed 会产生不同的 MurmurHash 函数
    for seed in range(num_hashes):
        minimum_hash = min(
            mmh3.hash(
                ngram,
                seed=seed,
                signed=False,
            )
            for ngram in ngrams
        )
        signature.append(minimum_hash)

    return tuple(signature)

#LSH 快速寻找候选文档
def _lsh_candidate_pairs(
    signatures: list[tuple[int, ...]],
    num_bands: int,
) -> set[tuple[int, int]]:
    """使用 LSH 找出可能相似的文档编号对。"""
    if not signatures:
        return set()

    if num_bands <= 0:
        raise ValueError("num_bands must be positive")

    num_hashes = len(signatures[0])

    if num_hashes % num_bands != 0:
        raise ValueError(
            "num_hashes must be divisible by num_bands"
        )

    rows_per_band = num_hashes // num_bands

    buckets = defaultdict(list)
    candidate_pairs: set[tuple[int, int]] = set()

    for document_index, signature in enumerate(signatures):
        for band_index in range(num_bands):
            start = band_index * rows_per_band
            end = start + rows_per_band

            band = signature[start:end]
            bucket_key = (band_index, band)

            # 当前 bucket 中已有的文档都与当前文档构成候选对。
            for previous_document_index in buckets[bucket_key]:
                candidate_pairs.add(
                    (
                        previous_document_index,
                        document_index,
                    )
                )

            buckets[bucket_key].append(document_index)

    return candidate_pairs

#Jaccard 相似度
def _jaccard_similarity(
    left: set[str],
    right: set[str],
) -> float:
    """计算两个 n-gram 集合的真实 Jaccard 相似度。"""
    union = left | right

    if not union:
        return 1.0

    return len(left & right) / len(union)

#组成重复簇并每组保留一篇
class _DisjointSet:
    """用并查集维护具有传递关系的重复文档簇。"""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root

        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    """使用 MinHash、LSH 和真实 Jaccard 相似度进行文档去重。"""
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    if num_hashes % num_bands != 0:
        raise ValueError(
            "num_hashes must be divisible by num_bands"
        )
    if ngrams <= 0:
        raise ValueError("ngrams must be positive")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError(
            "jaccard_threshold must be between 0 and 1"
        )

    input_paths = [Path(path) for path in input_files]
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    documents = []
    ngram_sets = []

    for input_path in input_paths:
        with xopen(
            input_path,
            mode="rt",
            encoding="utf-8",
            errors="replace",
        ) as input_file:
            text = input_file.read()

        documents.append(text)
        #第一步：把文章拆分成小片段（n-gram）
        ngram_sets.append(_word_ngrams(text, ngrams))
    
    #第二步：生成独特的minihash签数字签名
    signatures = [
        _minhash_signature(ngram_set, num_hashes)
        for ngram_set in ngram_sets
    ]

    #第三步：LSH快速分组，寻找可能相似的文章
    candidate_pairs = _lsh_candidate_pairs(
        signatures,
        num_bands,
    )
    
    #聚类并且保留一篇
    clusters = _DisjointSet(len(input_paths))

    # 第五步：用真实 Jaccard 相似度确认 LSH 只产生候选对；最终是否重复由真实 Jaccard 相似度决定
    for left_index, right_index in candidate_pairs:
        similarity = _jaccard_similarity(
            ngram_sets[left_index],
            ngram_sets[right_index],
        )

        if similarity >= jaccard_threshold:
            clusters.union(left_index, right_index)

    documents_by_cluster: dict[int, list[int]] = defaultdict(list)

    for document_index in range(len(input_paths)):
        cluster_root = clusters.find(document_index)
        documents_by_cluster[cluster_root].append(document_index)

    # 使用固定种子，使随机保留结果可重复。
    rng = random.Random(336)
    kept_indices = {
        rng.choice(cluster_members)
        for cluster_members in documents_by_cluster.values()
    }

    for document_index in sorted(kept_indices):
        destination = output_path / input_paths[document_index].name

        with xopen(
            destination,
            mode="wt",
            encoding="utf-8",
        ) as output_file:
            output_file.write(documents[document_index])
