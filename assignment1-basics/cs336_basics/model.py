import torch
import math


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
       
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3*std,
            b=3*std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...d,od->...o", x, self.weight)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(
                num_embeddings, 
                embedding_dim, 
                device=device, 
                dtype=dtype
            )
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        
        # 可学习的增益参数 g，PDF 要求初始化为全 1
        self.weight=torch.nn.Parameter(
            torch.ones(d_model,device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype  #记录输入原来的数据类型
        x = x.to(torch.float32)   #临时把输入转换成 float32
        rms=torch.sqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        )
        result=(x/rms)*self.weight

        return result.to(in_dtype)
    
def silu(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self,d_model:int,d_ff:int | None=None,device=None,dtype=None):
        super().__init__()
        #在没有手动传入 d_ff 时，自动计算前馈网络的隐藏维度：
        if d_ff is None:
            d_ff=int(8*d_model/3)
            #将结果向上取整到最接近的 64 的倍数，以提高硬件计算效率。
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.d_model=d_model
        self.d_ff=d_ff

        #w1和w3将输入从 d_model 扩展到 d_ff
        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)
        
        # W2 将门控后的结果投影回 d_model。
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = W2(SiLU(W1x) * W3x)
        gate=silu(self.w1(x))
        value=self.w3(x)
        hidden=gate*value
        return self.w2(hidden)


class SiLUFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))
    

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        #每一个特征对应一个旋转的频率
        frequencies=1.0/theta**(torch.arange(0,d_k,2, device=device,dtype=torch.float32)/d_k)

        #token 位置 i
        positions=torch.arange(0,max_seq_len,1,device=device, dtype=torch.float32)

        #随后再外积相乘 shape: (max_seq_len, d_k // 2)
        angle = torch.einsum("p,f->pf", positions, frequencies)

        # cos 和 sin 固定不训练，所以注册为 buffer
        self.register_buffer("cos",torch.cos(angle), persistent=False)
        self.register_buffer("sin",torch.sin(angle), persistent=False)
    
    def forward(self, x:torch.Tensor,token_postion:torch.Tensor) -> torch.Tensor:
        # 根据 token 的实际位置获取旋转参数
        cos=self.cos[token_postion]  #相当于cos(theta)
        sin=self.sin[token_postion]  #相当于sin(theta)

        # 将最后一维的相邻元素拆成两组
        x_even = x[..., 0::2]
        x_odd =  x[..., 1::2]

        # Batched position IDs omit the attention-head axis. Insert it so
        # broadcasting works for both [sequence] and [batch, sequence] IDs.
        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 应用旋转theta变换
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # 将旋转后的得到的元素重新交错排列
        result=torch.stack([rotated_even, rotated_odd], dim=-1)   #even0, odd0, even1, odd1, ...

        return result.flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value=torch.max(x,dim=dim,keepdim=True).values
    shifted_x=x-max_value
    exp_x=torch.exp(shifted_x)
    return exp_x/torch.sum(exp_x,dim=dim,keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)
    
    output=torch.matmul(attention_weights, V)

    return output


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads  #多头中每个头的维度
        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

        #四个线性层
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self,x:torch.Tensor, token_positions: torch.Tensor | None = None)->torch.Tensor:
        seq_len = x.shape[-2]

        #过线性层
        q = self.q_proj(x)    # [batch_size, seq_len, hidden_size]
        k = self.k_proj(x)    # [batch_size, seq_len, hidden_size]
        v = self.v_proj(x)    # [batch_size, seq_len, hidden_size]

        #将qkv拆分成多头
        q=q.view(*q.shape[:-1],self.num_heads,self.d_head).transpose(-3,-2)    #(batch_size, num_heads, seq_len, d_head)
        k=k.view(*k.shape[:-1],self.num_heads,self.d_head).transpose(-3,-2)
        v=v.view(*v.shape[:-1],self.num_heads,self.d_head).transpose(-3,-2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        #mask
        mask=torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attention_output=scaled_dot_product_attention(q,k,v,mask) #(batch_size, num_heads, seq_len, d_head)

        attention_output = attention_output.transpose(-3, -2)      #(batch_size, seq_len, num_heads, d_head)

        #把多个 head 的输出重新拼回一个完整的 d_model 向量。
        attention_output = attention_output.reshape(*x.shape[:-1], self.d_model)  #(batch_size, seq_len, d_model)
        
        #通过输出线性层
        return self.output_proj(attention_output)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        if norm_style not in {"pre", "post", "none"}:
            raise ValueError("norm_style must be 'pre', 'post', or 'none'")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        self.norm_style = norm_style
        
        #归一化层1
        self.ln1=RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.attn=MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta if use_rope else None,
            device=device,
            dtype=dtype,
        )
        
        #归一化层2
        self.ln2=RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        
        #FFN层
        ffn_cls = SwiGLU if ffn_type == "swiglu" else SiLUFeedForward
        self.ffn=ffn_cls(
            d_model=d_model,
            d_ff=d_ff if ffn_type == "swiglu" else None,
            device=device,
            dtype=dtype,
        )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.norm_style == "pre":
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        if self.norm_style == "post":
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(x)
        return x + self.ffn(x)
    

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embeddings=Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers=torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
                norm_style=norm_style,
                use_rope=use_rope,
                ffn_type=ffn_type,
            )
            #多层transformerblock
            for _ in range(num_layers)
        ])

        self.ln_final = (
            torch.nn.Identity()
            if norm_style == "none"
            else RMSNorm(d_model=d_model, device=device, dtype=dtype)
        )

        self.lm_head=Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs: [..., vocab_size]
    # targets: [...]

    max_logits = inputs.max(dim=-1, keepdim=True).values

    #避免计算 \(e^{o_j}\) 时溢出
    shifted_logits = inputs - max_logits

    log_normalizer = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
    )
    
    #取得正确类别的 logit
    target_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_normalizer - target_logits
    return losses.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)
        
    def step(self,closure=None):
        loss=None 
        #如果用户传了 closure就打开梯度计算,执行 closure()，重新 forward + backward
        if closure is not None:
            with torch.enable_grad():
                loss=closure
        
        for group in self.param_groups:
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad=param.grad
                state=self.state[param]

                # 第一次更新这个参数时，初始化状态
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                m=state["m"]
                v=state["v"]

                state["t"]+=1
                t=state["t"]

                #更新一阶动量m
                m.mul_(beta1)
                m.add_(grad,alpha=1 - beta1)

                #更新二阶动量v
                v.mul_(beta2)
                v.addcmul_(grad,grad,value=1-beta2)

                # 偏差修正
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # AdamW 的 decoupled weight decay
                param.data.mul_(1 - lr * weight_decay)

                # 参数更新，必须原地修改 param.data
                param.data.addcdiv_(
                    m_hat,
                    torch.sqrt(v_hat) + eps,
                    value=-lr,
                )
        
        return loss


#动态学习率
def get_lr_cosine_schedule(it:int,max_learning_rate:float,min_learning_rate:float,warmup_iters:int,cosine_cycle_iters: int,) -> torch.Tensor:
    #warm-up
    if it<warmup_iters:
        return  it/warmup_iters *max_learning_rate

    #cosine decay：从最大学习率平滑下降到最小学习率
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate+0.5*(1+math.cos(math.pi*progress))*(max_learning_rate-min_learning_rate)

    #post-annealing：cosine schedule 已经结束了，之后保持最终最小 learning rate，不继续下降
    
    return min_learning_rate


#gradient clipping限制整体梯度范数,防止某次 batch 导致 gradient 特别大，optimizer 更新时参数会被改得太猛，训练可能不稳定
def gradient_clipping(parameters, max_l2_norm: float):
    eps = 1e-6
    total_norm = 0.0

    # 第一步：计算所有梯度平方和
    for param in parameters:
        if param.grad is None:
            continue

        grad = param.grad
        total_norm += torch.sum(grad ** 2)

    # 第二步：平方和开根号，得到真正的 l2 norm
    total_norm = torch.sqrt(total_norm)

    # 第三步：如果总 norm 没有超过上限，就不需要裁剪
    if total_norm <= max_l2_norm:
        return

    # 第四步：计算缩放系数
    clip_coef = max_l2_norm / (total_norm + eps)

    # 第五步：原地缩放每一个参数的梯度
    for param in parameters:
        if param.grad is None:
            continue

        param.grad.mul_(clip_coef)


        


        


