"""
大模型预训练脚本 - 详细原理说明

================================================================================
一、核心概念：语言模型预训练（Language Model Pretraining）
================================================================================

【什么是预训练？】
预训练是训练大语言模型的第一阶段，其目标是让模型学习语言的统计规律和语义表示。
这是一种无监督学习任务，只需要大量的文本数据，无需人工标注。

【自回归语言建模（Autoregressive Language Modeling）】
训练过程本质上是"给定前面的token序列，预测下一个token"：

假设文本序列为：x = [x₁, x₂, ..., xₙ]
模型的任务是学习条件概率分布：P(x_i | x₁, x₂, ..., x_{i-1})

目标函数（负对数似然）：
    L = -Σ log P(x_i | x₁, ..., x_{i-1})
    
最大化对数似然等价于最小化交叉熵损失，这是标准的监督学习目标。

【训练数据格式】
输入序列 X = [BOS, "我", "爱", "你"]
目标序列 Y = ["我", "爱", "你", EOS]
模型预测：P("我"|BOS), P("爱"|BOS,"我"), P("你"|BOS,"我","爱"), P(EOS|...)


================================================================================
二、核心数学原理
================================================================================

【1. 交叉熵损失（Cross Entropy Loss）】
用于衡量预测概率分布与真实分布的差异：

对于单个样本：
    L = -Σ y_i * log(p_i)
    
其中：
- y_i 是真实标签的one-hot编码（只有真实类别为1，其他为0）
- p_i 是模型预测的softmax概率：p_i = exp(logits_i) / Σ exp(logits_j)

对于batch中的多个样本，通常取平均：
    L = -1/N * Σ Σ y_{n,i} * log(p_{n,i})

【2. Softmax函数】
将logits（未归一化的分数）转换为概率分布：
    softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
性质：
- 输出所有值在[0,1]区间，且和为1
- 大值会被放大，小值会被抑制
- 温度参数可以调节分布的平滑程度

【3. 反向传播（Backpropagation）】
使用链式法则计算梯度：

对于损失函数 L 和参数 θ：
    ∂L/∂θ = ∂L/∂y * ∂y/∂θ
    
其中：
- ∂L/∂y：损失对输出的梯度（由损失函数定义）
- ∂y/∂θ：输出对参数的梯度（由模型结构决定）

链式法则允许我们逐层反向传播梯度，从输出层到输入层。

【4. 梯度下降与优化器】
基础梯度下降：
    θ_{t+1} = θ_t - lr * ∇L(θ_t)
    
AdamW优化器（本代码使用）：
    m_t = β₁ * m_{t-1} + (1-β₁) * g_t          # 一阶矩（梯度均值）
    v_t = β₂ * v_{t-1} + (1-β₂) * g_t²         # 二阶矩（梯度方差）
    m̂_t = m_t / (1-β₁^t)                       # 偏差校正
    v̂_t = v_t / (1-β₂^t)
    θ_{t+1} = θ_t - lr * (m̂_t / (√v̂_t + ε) + λ*θ_t)  # 参数更新（含权重衰减）
    
优势：
- 自适应学习率（每个参数有独立的学习率）
- 考虑梯度的历史信息（动量）
- 权重衰减解耦，更稳定


================================================================================
三、训练流程概览
================================================================================

1. 数据加载：将文本tokenize成数字序列，构建batch
2. 前向传播：模型输出每个位置的logits（未归一化的概率分布）
3. 损失计算：使用交叉熵损失衡量预测与真实值的差距
4. 反向传播：使用自动微分计算所有参数的梯度
5. 梯度累积：累积多个mini-batch的梯度，模拟更大的batch size
6. 参数更新：使用优化器（如AdamW）更新模型参数
7. 重复上述步骤，直到完成所有epoch


================================================================================
四、关键技术详解
================================================================================

【1. 梯度累积（Gradient Accumulation）】
原理：模拟更大的batch size，而不增加显存占用

假设：
- 实际batch_size = 32
- accumulation_steps = 8
- 等效batch_size = 32 * 8 = 256

实现：
- 每个mini-batch计算损失并反向传播，但不更新参数
- 累积N个mini-batch的梯度
- 每N步更新一次参数，梯度 = 平均梯度

数学表达：
    g_accumulated = (1/N) * Σ g_i
    θ_{t+1} = θ_t - lr * g_accumulated

【2. 混合精度训练（Mixed Precision Training）】
原理：使用FP16/BF16进行前向计算，FP32进行反向传播

优势：
- 显存占用减半（FP16/BF16是16位，FP32是32位）
- 训练速度提升1.5-2倍（现代GPU对低精度运算优化更好）
- 精度损失可忽略（关键操作仍用FP32）

实现：
- 前向传播：大部分操作使用FP16/BF16
- 损失计算：使用FP32（保证精度）
- 反向传播：梯度计算使用FP32（防止下溢）
- 参数更新：使用FP32（保证精度）

【3. 分布式训练（Distributed Training）】
数据并行（Data Parallelism）：
- 每个GPU维护完整的模型副本
- 每个GPU处理不同的数据batch
- 反向传播后，所有GPU的梯度同步（all-reduce）
- 所有GPU使用相同的梯度更新参数

通信模式：
- All-Reduce：所有GPU的梯度求和后广播到所有GPU
- 公式：g_global = (1/N) * Σ g_i（N为GPU数量）

【4. 学习率调度（Learning Rate Scheduling）】
余弦退火策略（本代码使用）：
    lr(t) = lr_max * (0.1 + 0.45 * (1 + cos(π * t / T)))
    
其中：
- t：当前步数
- T：总步数
- lr_max：最大学习率
- 最终学习率 = 0.1 * lr_max（10%的最小值）

优势：
- 训练初期学习率大，快速收敛
- 训练后期学习率小，精细调优
- 平滑过渡，避免震荡

【5. 梯度裁剪（Gradient Clipping）】
原理：防止梯度爆炸，稳定训练过程

方法：如果梯度的L2范数超过阈值，将其缩放至阈值
    if ||g|| > clip_value:
        g = g * clip_value / ||g||

好处：
- 防止参数更新过大导致训练发散
- 稳定训练过程，特别是在训练初期


================================================================================
五、训练监控与调试
================================================================================

【关键指标】
- loss：总损失（包括主损失和辅助损失）
- logits_loss：预测损失（交叉熵损失）
- aux_loss：辅助损失（如MoE的负载均衡损失）
- learning_rate：当前学习率
- epoch_time：预计的epoch时间

【检查点（Checkpoint）】
保存内容：
- 模型权重（model state_dict）
- 优化器状态（optimizer state_dict，包含动量等）
- 训练进度（epoch、step）
- 其他状态（如学习率调度器、梯度缩放器）

作用：
- 断点续训：训练中断后可以继续
- 模型选择：保存多个检查点，选择最佳模型
- 实验管理：记录训练状态，便于实验复现


================================================================================
六、大模型训练的关键挑战与解决方案
================================================================================

挑战1：显存不足
- 解决方案：梯度累积、混合精度、梯度检查点（gradient checkpointing）

挑战2：训练不稳定
- 解决方案：梯度裁剪、学习率调度、权重初始化

挑战3：训练速度慢
- 解决方案：分布式训练、混合精度、数据加载优化

挑战4：过拟合
- 解决方案：数据增强、dropout、权重衰减、早停（early stopping）

挑战5：调试困难
- 解决方案：实验跟踪工具（wandb）、检查点机制、日志记录
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个epoch的核心函数
    
    【函数功能】
    执行一个完整的训练epoch，包括前向传播、损失计算、反向传播和参数更新。
    
    【参数说明】
    - epoch: 当前epoch编号
    - loader: 数据加载器，每次迭代返回一个batch的数据
    - iters: 该epoch的总迭代次数（用于学习率调度）
    - start_step: 起始步数（用于断点续训）
    - wandb: wandb日志对象（可选）
    
    【核心训练循环原理】
    每个训练步骤包含以下关键操作：
    1. 数据准备：将输入序列X和目标序列Y移动到GPU
    2. 学习率调度：根据训练进度动态调整学习率
    3. 前向传播：模型预测下一个token的概率分布
    4. 损失计算：使用交叉熵衡量预测误差
    5. 反向传播：计算梯度
    6. 梯度累积：累积多个mini-batch的梯度
    7. 参数更新：当累积步数达到阈值时更新模型参数
    """
    
    # ========== 损失函数定义 ==========
    # CrossEntropyLoss：交叉熵损失函数
    # 原理：用于多分类问题，衡量预测概率分布与真实分布的差异
    # 公式：L = -Σ y_i * log(p_i)，其中y_i是真实标签（one-hot），p_i是预测概率
    # reduction='none'：不进行reduce操作，保留每个样本的损失值，便于后续使用mask
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    start_time = time.time()  # 记录epoch开始时间，用于计算训练速度
    
    # ========== 训练循环：遍历每个batch ==========
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        """
        【数据格式说明】
        - X: 输入序列，形状为 [batch_size, seq_len]，包含token IDs
        - Y: 目标序列，形状为 [batch_size, seq_len]，是X向右偏移1位的结果
           （因为我们要预测下一个token）
        - loss_mask: 损失掩码，形状为 [batch_size, seq_len]，用于标记哪些位置需要计算损失
           （通常padding位置的mask为0，真实token位置的mask为1）
        
        【语言模型的训练原理】
        假设输入序列是：[BOS, "我", "爱", "你", EOS]
        那么：
        - X = [BOS, "我", "爱", "你"]
        - Y = ["我", "爱", "你", EOS]
        模型的任务是：给定前i个token，预测第i+1个token
        """
        
        # 将数据移动到指定设备（GPU或CPU）
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列（用于计算损失）
        loss_mask = loss_mask.to(args.device)  # 损失掩码
        
        # ========== 学习率调度 ==========
        # 原理：学习率不应该是一个常数，应该随着训练进行动态调整
        # 余弦退火策略：学习率从最大值平滑降低到最小值的10%
        # 公式：lr = lr_max * (0.1 + 0.45 * (1 + cos(π * current_step / total_steps)))
        # 优点：训练初期学习率大，快速收敛；训练后期学习率小，精细调优
        current_step = epoch * iters + step  # 全局步数
        total_steps = args.epochs * iters  # 总步数
        lr = get_lr(current_step, total_steps, args.learning_rate)
        
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 前向传播（混合精度） ==========
        # autocast_ctx：混合精度训练的上下文管理器
        # 原理：使用FP16/BF16进行前向计算，FP32进行反向传播
        # 优势：显存占用减半，训练速度提升1.5-2倍，精度损失可忽略
        with autocast_ctx:
            # 模型前向传播
            # 输入：X [batch_size, seq_len]
            # 输出：res.logits [batch_size, seq_len, vocab_size] - 每个位置对词汇表中每个token的未归一化分数
            #      res.aux_loss - 辅助损失（如果有，如MoE的负载均衡损失）
            res = model(X)
            
            # ========== 损失计算 ==========
            # 步骤1：将logits和targets展平为2D张量
            # res.logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            # Y: [batch_size, seq_len] -> [batch_size * seq_len]
            # 步骤2：计算每个位置的交叉熵损失
            # 公式：loss[i] = -log(softmax(logits[i])[Y[i]])
            # 步骤3：恢复原始形状 [batch_size, seq_len]
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 展平：[N, vocab_size]
                Y.view(-1)  # 展平：[N]
            ).view(Y.size())  # 恢复：[batch_size, seq_len]
            
            # ========== 加权损失计算 ==========
            # 使用loss_mask屏蔽padding位置的损失
            # 公式：logits_loss = Σ(loss * mask) / Σ(mask)
            # 只计算真实token位置的损失，忽略padding
            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # ========== 总损失 ==========
            # 主损失（预测损失）+ 辅助损失（如果有）
            # aux_loss可能是MoE的负载均衡损失、正则化项等
            loss = logits_loss + res.aux_loss
            
            # ========== 梯度累积的损失缩放 ==========
            # 原理：梯度累积模拟更大的batch size
            # 例如：accumulation_steps=8，实际batch_size=32
            # 等价于batch_size=256的训练，但每次只加载32个样本
            # 需要将损失除以累积步数，使得最终梯度 = 平均梯度
            loss = loss / args.accumulation_steps

        # ========== 反向传播（混合精度） ==========
        # scaler.scale：在混合精度训练中，需要先缩放损失再反向传播
        # 原理：FP16的范围较小，梯度可能下溢，需要放大后再计算
        # backward()：自动计算所有参数的梯度（链式法则）
        # 公式：∂L/∂θ = ∂L/∂y * ∂y/∂θ，其中y是模型输出，θ是参数
        scaler.scale(loss).backward()

        # ========== 梯度累积与参数更新 ==========
        # 当累积步数达到阈值时，执行参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 步骤1：取消梯度缩放（混合精度训练的逆操作）
            scaler.unscale_(optimizer)
            
            # 步骤2：梯度裁剪
            # 原理：防止梯度爆炸问题
            # 方法：如果梯度的L2范数超过阈值，将其缩放至阈值
            # 公式：if ||g|| > clip_value: g = g * clip_value / ||g||
            # 好处：稳定训练过程，防止参数更新过大导致训练发散
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 步骤3：优化器更新参数
            # scaler.step：在混合精度训练中，需要先缩放梯度再更新
            # 优化器（如AdamW）使用梯度更新参数：
            #   m_t = β1 * m_{t-1} + (1-β1) * g_t  (一阶矩估计)
            #   v_t = β2 * v_{t-1} + (1-β2) * g_t² (二阶矩估计)
            #   m̂_t = m_t / (1-β1^t)  (偏差校正)
            #   v̂_t = v_t / (1-β2^t)
            #   θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
            scaler.step(optimizer)
            scaler.update()  # 更新scaler的内部状态

            # 步骤4：清零梯度
            # set_to_none=True：将梯度设为None而不是0，节省内存
            optimizer.zero_grad(set_to_none=True)

        # ========== 日志记录 ==========
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time  # 已用时间（秒）
            # 将损失还原为原始值（之前除以了accumulation_steps）
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = logits_loss.item()  # 主损失
            current_aux_loss = res.aux_loss.item()  # 辅助损失
            current_lr = optimizer.param_groups[-1]['lr']  # 当前学习率
            
            # 估算剩余时间（分钟）
            # 公式：剩余时间 = (总时间 / 已完成步数) * 总步数 - 已用时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印训练日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            # 记录到wandb（如果启用）
            if wandb: 
                wandb.log({
                    "loss": current_loss, 
                    "logits_loss": current_logits_loss, 
                    "aux_loss": current_aux_loss, 
                    "learning_rate": current_lr, 
                    "epoch_time": eta_min
                })

        # ========== 模型保存（检查点） ==========
        # 定期保存模型，防止训练中断导致进度丢失
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式（关闭dropout等）
            
            # 构建保存路径
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取模型状态字典
            # DDP模式下，需要从model.module获取，否则直接从model获取
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()  # DDP包装的模型
            else:
                state_dict = model.state_dict()  # 普通模型
            
            # 转换为FP16以节省存储空间，并移到CPU
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)  # 保存模型权重
            
            # 保存完整检查点（包括优化器状态、epoch、step等，用于断点续训）
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints'
            )
            
            model.train()  # 切换回训练模式
            del state_dict  # 释放内存
        
        # ========== 内存清理 ==========
        # 删除不再使用的变量，释放GPU内存
        del X, Y, loss_mask, res, loss


if __name__ == "__main__":
    """
    ========== 主函数：大模型预训练入口 ==========
    
    【训练流程概述】
    1. 解析命令行参数
    2. 初始化分布式训练环境
    3. 设置随机种子（保证可复现性）
    4. 配置模型参数和数据路径
    5. 初始化模型、数据加载器、优化器
    6. 加载检查点（如果启用断点续训）
    7. 包装为DDP模型（如果使用多GPU）
    8. 执行训练循环
    9. 清理分布式进程
    """
    
    # ========== 参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    
    # 【模型保存相关】
    parser.add_argument("--save_dir", type=str, default="../out", 
                       help="模型保存目录，训练过程中会定期将模型权重保存到此目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, 
                       help="保存权重的前缀名，最终文件名格式：{save_weight}_{hidden_size}.pth")
    
    # 【训练超参数】
    parser.add_argument("--epochs", type=int, default=1, 
                       help="训练轮数（epoch数）。预训练通常1-6轮，取决于数据量和训练目标")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="每个GPU的batch size。实际有效batch size = batch_size * accumulation_steps * num_gpus")
    parser.add_argument("--learning_rate", type=float, default=5e-4, 
                       help="初始学习率。大模型通常使用较小的学习率（1e-4到1e-3）")
    
    # 【硬件配置】
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", 
                       help="训练设备，如'cuda:0'表示使用第0块GPU")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                       help="混合精度类型：'bfloat16'（推荐，数值稳定）或'float16'（更快但可能不稳定）")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="数据加载的线程数，用于并行加载数据，提高IO效率")
    
    # 【训练优化策略】
    parser.add_argument("--accumulation_steps", type=int, default=8, 
                       help="梯度累积步数。每累积N个batch的梯度后再更新参数，等效于batch_size*N")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                       help="梯度裁剪阈值。防止梯度爆炸，将梯度L2范数限制在此值内")
    
    # 【日志和保存频率】
    parser.add_argument("--log_interval", type=int, default=100, 
                       help="日志打印间隔（步数）。每N步打印一次训练状态")
    parser.add_argument("--save_interval", type=int, default=1000, 
                       help="模型保存间隔（步数）。每N步保存一次检查点")
    
    # 【模型架构参数】
    parser.add_argument('--hidden_size', default=512, type=int, 
                       help="隐藏层维度（d_model）。决定模型的表达能力，越大模型越强但显存占用也越大")
    parser.add_argument('--num_hidden_layers', default=8, type=int, 
                       help="Transformer层数（深度）。层数越多模型越深，但训练也越困难")
    parser.add_argument('--max_seq_len', default=340, type=int, 
                       help="训练的最大序列长度。超过此长度的文本会被截断。中文1token≈1.5~1.7字符")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                       help="是否使用MoE（Mixture of Experts）架构。MoE可以增大模型容量而不增加计算量")
    
    # 【数据相关】
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", 
                       help="预训练数据路径。数据格式为jsonl，每行一个JSON对象，包含'text'字段")
    
    # 【断点续训相关】
    parser.add_argument('--from_weight', default='none', type=str, 
                       help="基于哪个权重训练。如果指定权重名，会加载该权重继续训练；'none'表示从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], 
                       help="是否自动检测并续训。1表示自动查找最新的检查点并恢复训练状态（包括优化器、学习率等）")
    
    # 【实验跟踪】
    parser.add_argument("--use_wandb", action="store_true", 
                       help="是否使用wandb记录训练过程。wandb可以可视化训练曲线、超参数等")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", 
                       help="wandb项目名，用于组织和管理不同的实验")
    
    args = parser.parse_args()

    # ========== 1. 初始化分布式训练环境和随机种子 ==========
    """
    【分布式训练原理】
    分布式训练用于在多GPU/多机器上并行训练模型，主要有两种方式：
    1. 数据并行（Data Parallelism）：每个GPU处理不同的数据batch，最后同步梯度
    2. 模型并行（Model Parallelism）：将模型的不同部分放在不同的GPU上
    
    本代码使用数据并行（DDP - DistributedDataParallel）：
    - 每个GPU维护一份完整的模型副本
    - 每个GPU处理不同的数据batch
    - 反向传播后，所有GPU的梯度会被同步（all-reduce操作）
    - 所有GPU使用相同的梯度更新参数，保证模型一致性
    
    【随机种子设置】
    设置随机种子保证实验可复现：
    - 相同种子 + 相同代码 = 相同结果
    - 不同GPU使用不同的种子（+rank），保证每个GPU处理的数据顺序不同
    """
    local_rank = init_distributed_mode()  # 初始化DDP环境，返回当前进程的本地GPU编号
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"  # DDP模式下，每个进程使用对应的GPU
    
    # 设置随机种子，42是基础种子，每个进程加上自己的rank保证数据顺序不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查检查点 ==========
    """
    【模型配置】
    MiniMindConfig定义了模型的架构参数：
    - hidden_size: 隐藏层维度，影响模型的表达能力
    - num_hidden_layers: Transformer层数，影响模型的深度
    - use_moe: 是否使用MoE架构，影响模型容量和计算效率
    
    【检查点（Checkpoint）机制】
    检查点用于保存和恢复训练状态，包括：
    - 模型权重（model state_dict）
    - 优化器状态（optimizer state_dict，包含动量等）
    - 训练进度（epoch、step）
    - 其他状态（如学习率调度器的状态）
    
    断点续训的好处：
    - 训练中断后可以继续，不浪费计算资源
    - 可以调整超参数后继续训练
    - 可以保存多个检查点，选择最佳模型
    """
    os.makedirs(args.save_dir, exist_ok=True)  # 创建模型保存目录
    
    # 创建模型配置对象
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # 如果启用断点续训，尝试加载检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度训练 ==========
    """
    【混合精度训练原理】
    混合精度（Mixed Precision）训练使用FP16/BF16进行前向计算，FP32进行反向传播：
    
    1. 前向传播：使用FP16/BF16，显存减半，速度提升
    2. 反向传播：关键操作（如梯度计算）使用FP32，保证精度
    
    【FP16 vs BF16】
    - FP16（float16）：范围较小（最大65504），容易溢出和下溢
    - BF16（bfloat16）：范围与FP32相同（最大3.4e38），更稳定，推荐使用
    
    【Autocast原理】
    autocast是一个上下文管理器，自动选择合适的数据类型：
    - 大部分操作使用FP16/BF16（节省显存和计算）
    - 特定操作自动使用FP32（如softmax、loss计算，保证精度）
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16  # 选择数据类型
    
    # 创建autocast上下文：CPU上不使用（无效果），GPU上启用混合精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验跟踪工具（wandb） ==========
    """
    【实验跟踪的作用】
    wandb等工具用于记录和可视化训练过程：
    - 实时监控损失、学习率等指标
    - 可视化训练曲线
    - 记录超参数，便于实验管理
    - 支持断点续训时恢复wandb运行ID
    """
    wandb = None
    if args.use_wandb and is_main_process():  # 只在主进程初始化，避免重复记录
        import swanlab as wandb  # 这里使用swanlab作为wandb的替代（API兼容）
        
        # 如果有检查点，恢复wandb的运行ID（用于续训时保持同一个运行记录）
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 如果有ID则必须恢复
        
        # 构建运行名称（包含关键超参数，便于识别）
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        
        # 初始化wandb运行
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型、数据加载器、优化器 ==========
    """
    【模型初始化】
    init_model函数会：
    1. 加载tokenizer（用于文本编码和解码）
    2. 创建模型实例（根据配置）
    3. 如果指定了from_weight，加载预训练权重
    4. 将模型移动到指定设备
    
    【数据加载器】
    PretrainDataset用于加载和预处理预训练数据：
    - 读取jsonl文件，每行是一个文本样本
    - 使用tokenizer将文本转换为token IDs
    - 添加特殊token（如BOS、EOS）
    - 截断或padding到指定长度
    
    【分布式采样器】
    DistributedSampler确保每个GPU处理不同的数据：
    - 将数据集分成N份（N=GPU数量）
    - 每个GPU只处理属于自己的那份数据
    - 每个epoch会打乱数据顺序
    
    【梯度缩放器（GradScaler）】
    用于混合精度训练：
    - FP16训练时，梯度可能下溢（变为0）
    - GradScaler会将损失放大，反向传播后再缩小
    - BF16范围大，通常不需要缩放（但代码中为兼容性仍然使用）
    
    【优化器（AdamW）】
    AdamW是Adam的改进版，用于参数更新：
    - 自适应学习率（根据梯度历史调整）
    - 权重衰减（weight decay）解耦，更稳定
    - 公式：
        m_t = β1*m_{t-1} + (1-β1)*g_t      # 一阶矩（梯度均值）
        v_t = β2*v_{t-1} + (1-β2)*g_t²     # 二阶矩（梯度方差）
        m̂_t = m_t / (1-β1^t)               # 偏差校正
        v̂_t = v_t / (1-β2^t)
        θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ*θ_{t-1})  # 参数更新（含权重衰减λ）
    """
    # 初始化模型和tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 创建数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 创建分布式采样器（DDP模式）或None（单GPU模式）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建梯度缩放器（仅FP16时需要，BF16通常不需要但保留以兼容）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 创建优化器（AdamW，推荐用于大模型训练）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从检查点恢复训练状态 ==========
    """
    【断点续训原理】
    如果找到检查点，需要恢复：
    1. 模型权重：model.load_state_dict()
    2. 优化器状态：包含动量、二阶矩等，保证优化器行为连续
    3. 梯度缩放器状态：混合精度训练的scaler状态
    4. 训练进度：epoch和step，从上次中断的地方继续
    
    这样训练过程就像从未中断一样，保证训练的连续性。
    """
    start_epoch, start_step = 0, 0  # 默认从头开始
    if ckp_data:  # 如果找到检查点
        model.load_state_dict(ckp_data['model'])  # 恢复模型权重
        optimizer.load_state_dict(ckp_data['optimizer'])  # 恢复优化器状态
        scaler.load_state_dict(ckp_data['scaler'])  # 恢复scaler状态
        start_epoch = ckp_data['epoch']  # 恢复epoch
        start_step = ckp_data.get('step', 0)  # 恢复step（如果存在）
    
    # ========== 7. 使用DDP包装模型（多GPU训练） ==========
    """
    【DDP（DistributedDataParallel）原理】
    DDP是PyTorch的分布式训练包装器：
    
    1. 每个GPU维护模型副本，处理不同的数据batch
    2. 反向传播时，所有GPU的梯度自动同步（all-reduce）
    3. 所有GPU使用相同的梯度更新，保证模型一致性
    4. 比DataParallel更高效（无GIL锁，通信优化）
    
    【参数忽略】
    _ddp_params_and_buffers_to_ignore指定不需要同步的参数：
    - freqs_cos, freqs_sin：RoPE位置编码的缓存，每个GPU计算相同，无需同步
    - 减少不必要的通信，提升效率
    """
    if dist.is_initialized():  # 如果是分布式训练
        # 设置DDP需要忽略的参数（这些参数每个GPU都相同，不需要同步）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        
        # 使用DDP包装模型，指定该进程使用的GPU
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练循环 ==========
    """
    【训练循环结构】
    外层循环：遍历每个epoch
    内层循环：在train_epoch函数中，遍历每个batch
    
    【Epoch vs Step】
    - Epoch：完整遍历一次数据集
    - Step：处理一个batch
    - 关系：1 epoch = len(dataset) / batch_size steps
    
    【断点续训处理】
    如果从检查点恢复，需要：
    1. 跳过已经训练过的step
    2. 使用SkipBatchSampler跳过前面的batch
    3. 从上次中断的step继续训练
    """
    for epoch in range(start_epoch, args.epochs):  # 从恢复的epoch开始
        # DDP模式下，每个epoch需要设置sampler的epoch，保证数据打乱
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0: 
            # 情况1：第一个epoch且存在检查点，需要跳过已训练的step
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), 
                args.batch_size, 
                start_step + 1  # 跳过的batch数
            )
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True  # pin_memory加速GPU传输
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else: 
            # 情况2：正常训练，从头开始
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None),  # 单GPU时shuffle，DDP时用sampler
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布式进程 ==========
    """
    【进程清理】
    训练结束后，需要清理分布式进程组：
    - 释放通信资源
    - 关闭进程间通信
    - 避免资源泄漏
    """
    if dist.is_initialized(): 
        dist.destroy_process_group()  # 销毁进程组，释放资源