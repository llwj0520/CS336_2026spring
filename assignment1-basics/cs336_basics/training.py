import numpy as np
import torch
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

        print(f"iter {it}: loss = {loss.item()}")

    return model



