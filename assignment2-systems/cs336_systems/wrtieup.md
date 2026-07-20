# CS336 Assignment 2 Writeup

## 2.1.5 混合精度：累加实验

实验结果如下：

| 累加器类型  | 加数类型    |    输出结果 |
| ----------- | ----------- | ----------: |
| `float32` | `float32` | `10.0001` |
| `float16` | `float16` |  `9.9531` |
| `float32` | `float16` | `10.0021` |

使用 `float32` 累加时，结果接近预期的 10，但仍有微小误差，因为十进制小数 `0.01` 无法被二进制浮点数精确表示。完全使用 `float16` 累加时，每次加法后的结果都必须被重新舍入为低精度数值，误差在 1000 次累加后逐渐累积，因此得到明显偏小的 `9.9531`。第三个实验中，累加器保持为 `float32`，避免了每次累加时的低精度舍入；但加数 `0.01` 已先被表示为近似的 `float16` 数值，所以结果仍有较小偏差。

## 2.1.5 混合精度：ToyModel 的数据类型

`Linear -> LayerNorm -> Linear -> CrossEntropyLoss`，并假设模型原始参数都是 `float32`，在 CUDA 的 `autocast(..., dtype=torch.float16)` 中运行

| 组件                 | dtype       | 原因                                                                  |
| -------------------- | ----------- | --------------------------------------------------------------------- |
| 模型参数             | `float32` | autocast 不会修改模型中保存的参数类型。                               |
| `Linear`的输出     | `float16` | `Linear` 本质是矩阵乘法，autocast 会使用低精度以利用 Tensor Cores。 |
| `LayerNorm` 的输出 | `float32` | LayerNorm 包含均值、方差和归约计算，对精度敏感，因此保留较高精度。    |
| 模型最终 logits      | `float16` | 第二个`Linear` 也会在 autocast 下以低精度计算。                     |
| loss                 | `float32` | CrossEntropyLoss 包含指数、对数和归约等易受精度影响的操作。           |
| 模型梯度             | `float32` | 模型参数是`float32`，PyTorch 会将对应梯度保存为相同类型。           |

## 2.1.5 混合精度：LayerNorm 的数值稳定性

LayerNorm 中计算均值和方差的归约操作、`x - mean` 的操作，以及方差的平方和开方都对低精度敏感。FP16 的有效精度和数值范围较小，可能使均值和方差出现明显的舍入误差、溢出或下溢，进而放大归一化后的误差。BF16 的指数范围与 FP32 相同，因此较少出现溢出和下溢；但其有效数字仍明显少于 FP32，所以计算 LayerNorm 的统计量时仍应使用 FP32。

## 6 Optimizer State Sharding Accounting

> 注意：由于当前没有 CUDA 环境，以下显存和训练速度均为理论估算，不是实际 benchmark 结果。

### 6.1 显存使用分析

XL 模型约有 1,998,235,200 个参数。假设模型参数和梯度均使用 FP32，每个数占 4 字节，则每个 rank 的理论内存组成如下：

| 内存内容                   |  理论显存 |
| -------------------------- | --------: |
| 模型参数                   |  7.44 GiB |
| 梯度                       |  7.44 GiB |
| 普通 AdamW 优化器状态      | 14.89 GiB |
| 两 rank 分片后的优化器状态 |  7.44 GiB |

各测量位置的理论显存如下：

| 测量位置            | 普通 AdamW | ShardedOptimizer |
| ------------------- | ---------: | ---------------: |
| 模型初始化后        |   7.44 GiB |         7.44 GiB |
| optimizer.step() 前 |  14.89 GiB |        14.89 GiB |
| optimizer.step() 后 |  29.78 GiB |        22.33 GiB |

普通 AdamW 在每个 rank 上保存全部一阶动量和二阶动量，而两 rank 的 `ShardedOptimizer` 每个 rank 只保存约一半优化器状态。因此，分片后每个 rank 理论上可以节省约 7.44 GiB。实际峰值还会包含激活值、临时计算 tensor、通信缓冲区和 PyTorch CUDA 缓存，因此实际结果通常高于上述理论值。

### 6.2 训练速度分析

在没有 CUDA 环境的情况下，无法提供可靠的每步训练时间。理论上，`ShardedOptimizer` 每个 rank 只更新一部分参数，减少了本地优化器计算，但更新后需要广播所有参数分片，因此会增加通信开销。

具体速度取决于 GPU、NCCL、NVLink 或 PCIe 带宽。粗略预计分片优化器可能比普通 AdamW 慢约 5%–30%，但该范围不能代替实际 benchmark 结果。

### 6.3 与 ZeRO Stage 1 的区别

我们的实现和 ZeRO Stage 1 都将优化器状态分片到不同 rank，同时让每个 rank 保留完整模型参数。我们的实现先对完整梯度执行 all-reduce，再由参数所属 rank 更新参数并逐个 broadcast；ZeRO Stage 1 通常使用 reduce-scatter 将梯度直接发送到负责对应分片的 rank，更新后再 all-gather 参数。

因此，两者的主要内存收益相似，但 ZeRO Stage 1 的通信组织更加高效。当前实现会产生完整梯度 all-reduce 和多次参数 broadcast，通信调用次数和通信开销可能更高。

## 7 Fully-Sharded Data Parallel

### 7.1 显存估算

忽略 all-gather 预分配缓冲区和较小的复制参数，在两个 rank
上，FSDP 将模型参数、梯度和优化器状态都近似分成两份。
对于约 19.98 亿个 FP32 参数，每个 rank 的核心持久显存从
Optimizer State Sharding 的约 22.33 GiB 降低到约
14.89 GiB，理论上继续节省约 7.44 GiB，即约 33.3%。

### 7.2 FSDP All-Gather 分析

由于当前没有 NVIDIA CUDA 和 Nsight Systems 环境，无法提供
实际时间线截图。当前 FSDP 实现在每个 Linear 或 Embedding
的 forward pre-hook 中同步执行权重 all-gather，因此权重会在
前向计算前准备完成，但前向计算必须等待通信，all-gather 没有
与前面层的计算重叠。

理论上，加入提前两个可分片层的权重预取后，all-gather 可以与
前面层的计算重叠；是否能够完全隐藏通信取决于 NCCL 通信时间
是否小于这段可重叠的计算时间，需要在双 GPU 环境中通过 Nsight
时间线验证。


# 8  Analyzing Parallelism Strategies

以下推导采用 PDF 的假设：设备计算速度为 $C$ FLOP/s，每个设备的出口带宽为 $W$ bytes/s，权重与激活均为 FP16，因此每个元素占 2 bytes。矩阵乘法 $(A,B)(B,C)\to(A,C)$ 需要 $2ABC$ FLOPs。

## 8.1 Alternate Ring All-Reduce

该算法执行 $N-1$ 个通信步骤，每一步发送大小为 $S$ 的完整张量。每一步耗时为 $S/W$，因此总时间为

$$
T_{\mathrm{alternate}}=(N-1)\frac{S}{W}.
$$

普通 ring all-reduce 每一步发送大小为 $S/N$ 的分块，而这里每一步发送完整张量，因此该替代算法的通信量更大。

## 8.2 Analyzing Data Parallel

### (a) Backward FLOPs

每个数据并行设备处理的 batch size 为 $B/N_{\mathrm{DP}}$。忽略逐元素操作，backward 包含计算 $dZ$ 的 1 个矩阵乘法、计算 $dX$ 的 2 个矩阵乘法，以及计算 $dW_1,dW_2,dW_3$ 的 3 个矩阵乘法，共 6 个。因此

$$
F_{\mathrm{DP,bwd}}
=6\left(2\frac{B}{N_{\mathrm{DP}}}DD_{\mathrm{FF}}\right)
=\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{DP}}}.
$$

### (b) Backward Communication

三个权重梯度共有 $3DD_{\mathrm{FF}}$ 个 FP16 元素，即 $6DD_{\mathrm{FF}}$ bytes。Ring all-reduce 的时间为 $2(N-1)S/(NW)$，所以

$$
T_{\mathrm{DP,bwd,comm}}
=2\frac{N_{\mathrm{DP}}-1}{N_{\mathrm{DP}}}
\frac{6DD_{\mathrm{FF}}}{W}
=\frac{12(N_{\mathrm{DP}}-1)DD_{\mathrm{FF}}}
{N_{\mathrm{DP}}W}.
$$

### (c) Communication Bottleneck

计算时间为

$$
T_{\mathrm{DP,bwd,compute}}
=\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{DP}}C}.
$$

要求通信时间不超过计算时间，化简得到

$$
\boxed{N_{\mathrm{DP}}\le 1+\frac{BW}{C}}.
$$

因此，更大的 batch size 或更高的带宽允许扩展到更多数据并行设备，而更快的计算设备会更早受到通信限制。

## 8.3 Analyzing Fully Sharded Data Parallel

### (a) Forward and Backward FLOPs

FSDP 同样将 batch 分成 $N_{\mathrm{FSDP}}$ 份。Forward 有 3 个矩阵乘法，backward 有 6 个，因此

$$
F_{\mathrm{FSDP,fwd}}
=\frac{6BDD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}},
$$

$$
F_{\mathrm{FSDP,bwd}}
=\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}.
$$

### (b) Forward and Backward Communication

Forward 需要 all-gather 三个权重矩阵，其总大小为 $6DD_{\mathrm{FF}}$ bytes，因此

$$
T_{\mathrm{FSDP,fwd,comm}}
=\frac{6(N_{\mathrm{FSDP}}-1)DD_{\mathrm{FF}}}
{N_{\mathrm{FSDP}}W}.
$$

Backward 既需要再次 all-gather 权重，也需要 reduce-scatter 三个权重梯度。二者通信量相同，所以

$$
T_{\mathrm{FSDP,bwd,comm}}
=\frac{12(N_{\mathrm{FSDP}}-1)DD_{\mathrm{FF}}}
{N_{\mathrm{FSDP}}W}.
$$

### (c) Communication Bottleneck

Forward 和 backward 分别比较通信时间与对应计算时间，两个不等式化简后都得到

$$
\boxed{N_{\mathrm{FSDP}}\le 1+\frac{BW}{C}}.
$$

Forward 的通信量和计算量都正好是 backward 的一半，所以两者得到相同的扩展上限。

## 8.4 Analyzing Tensor Parallel

### (a) Backward Pass

每个 TP rank 已经持有相同的 $dY$。首先计算各自的局部中间梯度：

$$
dZ^{(i)}=dY\left(W_3^{(i)}\right)^\top,
$$

$$
dX_2^{(i)}=dZ^{(i)}\odot f\left(X_1^{(i)}\right),
$$

$$
dX_1^{(i)}=dZ^{(i)}\odot f'\left(X_1^{(i)}\right)\odot X_2^{(i)}.
$$

局部权重梯度为

$$
dW_3^{(i)}=\left(Z^{(i)}\right)^\top dY,
$$

$$
dW_2^{(i)}=X^\top dX_2^{(i)},
$$

$$
dW_1^{(i)}=X^\top dX_1^{(i)}.
$$

每个 rank 对输入梯度产生一个局部贡献：

$$
dX^{(i)}
=dX_1^{(i)}\left(W_1^{(i)}\right)^\top
+dX_2^{(i)}\left(W_2^{(i)}\right)^\top.
$$

最后沿 TP 轴执行 all-reduce：

$$
dX=\operatorname{all\text{-}reduce}
\left(\left\{dX^{(i)}\right\}_{i=0}^{N_{\mathrm{TP}}-1}\right).
$$

### (b) Forward and Backward FLOPs

每个权重矩阵沿 $D_{\mathrm{FF}}$ 维度分成 $N_{\mathrm{TP}}$ 份，因此

$$
F_{\mathrm{TP,fwd}}
=\frac{6BDD_{\mathrm{FF}}}{N_{\mathrm{TP}}},
$$

$$
F_{\mathrm{TP,bwd}}
=\frac{12BDD_{\mathrm{FF}}}{N_{\mathrm{TP}}}.
$$

### (c) Forward and Backward Communication

Forward 对形状为 $(B,D)$ 的 $Y$ 执行一次 all-reduce。Backward 对同样形状的 $dX$ 执行一次 all-reduce。每个张量大小为 $2BD$ bytes，因此

$$
T_{\mathrm{TP,fwd,comm}}
=T_{\mathrm{TP,bwd,comm}}
=\frac{4(N_{\mathrm{TP}}-1)BD}{N_{\mathrm{TP}}W}.
$$

### (d) Communication Bottleneck

Forward 要求通信时间不超过 $6BDD_{\mathrm{FF}}/(N_{\mathrm{TP}}C)$，得到

$$
\boxed{N_{\mathrm{TP}}\le
1+\frac{3D_{\mathrm{FF}}W}{2C}}.
$$

Backward 的计算量是 forward 的两倍，而通信量相同，因此

$$
\boxed{N_{\mathrm{TP}}\le
1+\frac{3D_{\mathrm{FF}}W}{C}}.
$$

## 8.5 2D Parallelism: FSDP + TP

令总设备数为

$$
N=N_{\mathrm{FSDP}}N_{\mathrm{TP}}.
$$

### (a) Forward FLOPs

FSDP 将 batch 分成 $N_{\mathrm{FSDP}}$ 份，TP 将每个矩阵乘法分成 $N_{\mathrm{TP}}$ 份，所以

$$
F_{\mathrm{2D,fwd}}
=\frac{6BDD_{\mathrm{FF}}}
{N_{\mathrm{FSDP}}N_{\mathrm{TP}}}
=\frac{6BDD_{\mathrm{FF}}}{N}.
$$

### (b) Forward Communication with Overlap

每个 TP rank 上的三个权重分片共有 $6DD_{\mathrm{FF}}/N_{\mathrm{TP}}$ bytes，因此 FSDP 轴 all-gather 的时间为

$$
T_{\mathrm{FSDP-axis}}
=\frac{6(N_{\mathrm{FSDP}}-1)DD_{\mathrm{FF}}}
{N_{\mathrm{FSDP}}N_{\mathrm{TP}}W}.
$$

每个 FSDP rank 上的输出形状为 $(B/N_{\mathrm{FSDP}},D)$，因此 TP 轴 all-reduce 的时间为

$$
T_{\mathrm{TP-axis}}
=\frac{4(N_{\mathrm{TP}}-1)BD}
{N_{\mathrm{FSDP}}N_{\mathrm{TP}}W}.
$$

两个轴可以重叠时，总通信时间为

$$
T_{\mathrm{2D,fwd,comm}}
=\max\left(
T_{\mathrm{FSDP-axis}},
T_{\mathrm{TP-axis}}
\right).
$$

### (c) Maximum Devices with Overlapped Axes

计算时间为 $6BDD_{\mathrm{FF}}/(NC)$。分别要求两个通信轴不超过计算时间，得到

$$
N_{\mathrm{FSDP}}
\le 1+\frac{BW}{C},
$$

$$
N_{\mathrm{TP}}
\le 1+\frac{3D_{\mathrm{FF}}W}{2C}.
$$

因此最优情况下

$$
\boxed{
N\le
\left(1+\frac{BW}{C}\right)
\left(1+\frac{3D_{\mathrm{FF}}W}{2C}\right)
}.
$$

由于两个通信轴可以重叠，可以分别把 $N_{\mathrm{FSDP}}$ 和 $N_{\mathrm{TP}}$ 扩展到各自的通信上限，再将两者相乘。

### (d) Maximum Devices without Overlapped Axes

两个通信轴不能重叠时，要求通信时间之和不超过计算时间。消去公共因子后得到

$$
D_{\mathrm{FF}}(N_{\mathrm{FSDP}}-1)
+\frac{2B}{3}(N_{\mathrm{TP}}-1)
\le \frac{BD_{\mathrm{FF}}W}{C}.
$$

令

$$
K=\frac{BD_{\mathrm{FF}}W}{C},\qquad
a=D_{\mathrm{FF}},\qquad
b=\frac{2B}{3}.
$$

在连续且内部最优的情况下，最大化 $N=N_{\mathrm{FSDP}}N_{\mathrm{TP}}$ 得到

$$
N_{\mathrm{FSDP}}^*=\frac{K+a+b}{2a},
\qquad
N_{\mathrm{TP}}^*=\frac{K+a+b}{2b},
$$

因此

$$
\boxed{N\le\frac{(K+a+b)^2}{4ab}}.
$$

当设备数较大、常数 1 可以忽略时，上式近似为

$$
\boxed{N\lesssim
\frac{3BD_{\mathrm{FF}}W^2}{8C^2}}.
$$

不能重叠时，FSDP 和 TP 必须共享同一份通信预算，因此最大设备数小于两个轴可以完全重叠时的结果。

## 9 Leaderboard

由于当前没有两张 NVIDIA B200 GPU，无法运行课程指定的
8B 模型 benchmark，也无法提供有效的 wall-clock time。

当前实现已经包含 FlashAttention、FSDP、优化器状态分片和
通信预取，但 CUDA/NCCL 路径仍需在课程服务器上验证。若获得
B200 环境，后续将优先测试 FlashAttention、FSDP、融合
LM-head cross-entropy 和 activation checkpointing。