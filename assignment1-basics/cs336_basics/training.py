import numpy as np
import torch
import time
import json
from cs336_basics.model import TransformerLM,cross_entropy,AdamW,gradient_clipping,get_lr_cosine_schedule

#从一长串 token 里面，随机切出 batch_size 个连续片段。每个片段的 x 是输入，y 是 x 往后错一位。
def get_batch(dataset:np.ndarray,batch_size:int,context_length:int,device:str):
    max_start=len(dataset)-context_length

    #一次随机生成 batch_size 个起点。
    starts=np.random.randint(
        low=0,
        high=max_start,
        size=batch_size,
    )

    x=[]
    y=[]
    
    for start in starts:
        x.append(dataset[start:start+context_length])
        y.append(dataset[start + 1 : start + context_length + 1])
    
    x=torch.tensor(x, dtype=torch.long, device=device)
    y=torch.tensor(y, dtype=torch.long, device=device)

    return x, y

#在训练过程中途保存状态，之后能够恢复训练
# 1. model 的参数状态
# 2. optimizer 的状态
# 3. 当前训练到了第几步 iteration
def save_checkpoint(model,optimizer,iteration:int,out):
    checkpoint = {
        "model" : model.state_dict(),  #模型权重
        "optimizer" : optimizer.state_dict(),   #优化器状态,比如 AdamW 的 m、v、t
        "iteration" : iteration,  #记录当前训练到第几步
    }
    torch.save(checkpoint,out)  #把整个字典保存到文件

def load_checkpoint(src,model,optimizer):

    checkpoint=torch.load(src)  #加载checkpoint
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]


def train(
        train_dataset,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        batch_size:int,
        max_learning_rate: float,
        weight_decay: float,
        num_iters: int,
        max_l2_norm:float,
        min_learning_rate:float,
        warmup_iters:int,
        cosine_cycle_iters:int,
        device=str,
        valid_dataset =None,     #验证集 token ids
        eval_every: int = 100,   #每隔多少步评估一次
        log_path: str | None = None,     #日志文件路径
):
    #模型选择：
    model=TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
       
    )

    #优化器选择
    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
    )

    start_time=time.time()

    for it in range(num_iters):
        model.train()

        #获取数据
        x,y=get_batch(
            dataset=train_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        
        #1、前向传播：把输入 x 送进模型，得到预测结果 logits。
        logits=model(x)

        #2、计算损失：把模型预测 logits 和真实标签 y 做比较，得到损失值 
        loss=cross_entropy(logits,y)
        
        #3、梯度归零
        optimizer.zero_grad()

        #4、反向传播
        loss.backward()

        gradient_clipping(model.parameters(),max_l2_norm)
        
        #采用动态学习率
        lr=get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )

        #把当前 step 计算出来的 learning rate 写进 optimizer，确保参数更新步骤使用的是新的学习率
        for group in optimizer.param_groups:
            group["lr"] = lr

        #5、参数更新
        optimizer.step()

        train_loss = loss.item()
        valid_loss = None
        
        #保存日志
        if valid_dataset is not None and (it + 1) % eval_every == 0:
            model.eval()

            with torch.no_grad():
                valid_x, valid_y = get_batch(
                    dataset=valid_dataset,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )

                valid_logits = model(valid_x)
                valid_loss = cross_entropy(valid_logits, valid_y).item()

            model.train()

        elapsed_time = time.time() - start_time

        log_record = {
            "iter": it + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "lr": lr,
            "elapsed_time": elapsed_time,
        }

        print(log_record)

        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\n")

    return model

#给定一段 prompt token，让语言模型不断预测下一个 token，并把预测结果接到后面，直到生成够 max_new_tokens 或遇到结束符。
def generate(
    model,
    prompt_ids: list[int],   #prompt 编码后的 token id 列表 
    max_new_tokens: int,     #最多生成多少个新 token
    context_length: int,     #模型最大上下文长度
    top_p: float | None = None, 
    temperature: float = 1.0,   #控制随机性
    end_token_id: int | None = None,   #如果生成到这个 token，就停止
    device: str = "mps",
    
):
    
    #把模型切换到评估 / 推理模式。
    model.eval()

    generated=list(prompt_ids)
    
    #关闭梯度计算
    with torch.no_grad():
        #循坏生成新的token
        for _ in range(max_new_tokens):
            #截取最近的上下文输入进模型
            input_ids = generated[-context_length:]

            input_tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=device,
            )

            #模型前向传播
            logits = model(input_tensor)
            
            #取最后一个位置的 logits
            next_token_logits = logits[0, -1, :]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            probs = torch.softmax(next_token_logits, dim=-1)
            
            if top_p is not None:
                #先按概率从大到小排序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                #计算累计概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                #根据 top_p 生成保留 mask
                keep_mask = cumulative_probs <= top_p
                
                #保证至少保留一个 token
                keep_mask[0] = True

                #把不保留的概率置 0
                filtered_sorted_probs = sorted_probs * keep_mask
                 
                #把排序后的概率放回原来的 token 位置
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=filtered_sorted_probs,
                )

                probs = filtered_probs / filtered_probs.sum()

            #按概率采样下一个 token
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            #生成的token加入新结果
            generated.append(next_token_id)

            if end_token_id is not None and next_token_id == end_token_id:
                break

    return generated



