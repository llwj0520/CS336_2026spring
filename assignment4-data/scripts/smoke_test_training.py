from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM


TRAIN_PATH = Path(
    "local-shared-data/filtered/filtered_data_gpt2.bin"
)
VALID_PATH = Path(
    "local-shared-data/tokenized_paloma_c4_100_domains_validation.bin"
)


def main() -> None:
    # 读取二进制训练集和验证集
    train_data = np.memmap(
        TRAIN_PATH,
        dtype=np.uint16,
        mode="r",
    )
    valid_data = np.memmap(
        VALID_PATH,
        dtype=np.uint16,
        mode="r",
    )

    torch.manual_seed(0)

    # 创建一个很小的模型，只用于 CPU 流程检查
    model = BasicsTransformerLM(
        vocab_size=50257,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        rope_theta=10000.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # 从训练集中抽取一个批次
    batch_x, batch_y = get_batch(
        train_data,
        batch_size=2,
        context_length=32,
        device="cpu",
    )

    # 前向传播
    model.train()
    logits = model(batch_x)

    # 计算下一个 token 的交叉熵损失
    train_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch_y.reshape(-1),
    )

    if not torch.isfinite(train_loss):
        raise RuntimeError("训练损失不是有限数值")

    # 反向传播并更新一次参数
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 在 Paloma 验证集上检查一次
    model.eval()
    with torch.no_grad():
        valid_x, valid_y = get_batch(
            valid_data,
            batch_size=2,
            context_length=32,
            device="cpu",
        )
        valid_logits = model(valid_x)
        valid_loss = F.cross_entropy(
            valid_logits.reshape(-1, valid_logits.size(-1)),
            valid_y.reshape(-1),
        )

    print("训练输入形状：", tuple(batch_x.shape))
    print("训练输入类型：", batch_x.dtype)
    print("模型输出形状：", tuple(logits.shape))
    print("训练损失：", float(train_loss))
    print("验证损失：", float(valid_loss))
    print("前向传播、反向传播和参数更新均成功")


if __name__ == "__main__":
    main()