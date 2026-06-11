"""
液态神经网络核心模型
====================
基于 MIT CSAIL Liquid Neural Network 实现记忆门控。

核心公式: dh/dt = (-h/τ + tanh(A·h + B·x)) / τ
参数量: ~700
推理延迟目标: <5ms (CPU)
"""

import math
import torch
import torch.nn as nn
from typing import Optional

# ============================================================
# Liquid ODE Cell (核心动力学层)
# ============================================================

class LiquidODECell(nn.Module):
    """
    连续时间液态神经网络单元
    
    实现: dh/dt = (-h + activation(A·h + B·x + bias)) / τ
    
    其中 τ 是可学习的时间常数，控制状态演化速率。
    τ 大 → 状态变化慢（适合稳定偏好）
    τ 小 → 状态变化快（适合快速漂移检测）
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 隐藏到隐藏权重 (A)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 输入到隐藏权重 (B)
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=False)
        # 可学习时间常数 τ (每个神经元独立)
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 2.0)
        # 偏置
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        # 激活
        self.activation = nn.Tanh()

    def forward(self, t: torch.Tensor, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        计算 dh/dt
        
        Args:
            t: 时间点 (标量)
            h: 当前隐藏状态 [batch, hidden_dim]
            x: 输入特征 [batch, input_dim]
            
        Returns:
            dh/dt [batch, hidden_dim]
        """
        tau = torch.clamp(self.tau.abs(), min=0.1, max=10.0)
        dhdt = (-h + self.activation(self.W_h(h) + self.W_x(x) + self.bias)) / tau
        return dhdt


class LiquidNetwork(nn.Module):
    """
    完整液态神经网络
    
    结构: Input → Encoder → LiquidODE → Decoder → Output
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 16,
                 output_dim: int = 3, use_adjoint: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_adjoint = use_adjoint

        # 输入编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # 液态ODE层
        self.liquid_cell = LiquidODECell(hidden_dim, hidden_dim)

        # 输出解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # 输出在[0,1]范围
        )

    def forward(self, x: torch.Tensor, t_span: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, input_dim]
            t_span: 时间跨度 (默认 [0, 1])
            
        Returns:
            output: [drift_score, retrieval_relevance, importance] [batch, 3]
        """
        from torchdiffeq import odeint

        batch_size = x.shape[0]

        # 编码输入
        encoded = self.encoder(x)  # [batch, hidden_dim]

        # ODE求解
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x.device)

        # 初始隐藏状态
        h0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # 定义ODE函数 (固定输入x)
        def ode_func(t, h):
            return self.liquid_cell(t, h, encoded)

        # 求解ODE: 取最终时间点的状态
        if self.use_adjoint:
            from torchdiffeq import odeint_adjoint as odeint_solver
        else:
            odeint_solver = odeint

        h_trajectory = odeint_solver(
            ode_func, h0, t_span,
            method='euler',  # Euler法最快
            options={'step_size': 0.25},  # 4步
        )
        # h_trajectory: [time_steps, batch, hidden_dim]
        h_final = h_trajectory[-1]  # [batch, hidden_dim]

        # 解码输出
        output = self.decoder(h_final)  # [batch, 3]

        return output

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        快速推理模式 (固定4步Euler，避免odeint开销)
        
        Args:
            x: 输入特征 [1, input_dim]
            
        Returns:
            output: [1, 3]
        """
        encoded = self.encoder(x)
        h = torch.zeros(1, self.hidden_dim, device=x.device)
        dt = 0.25

        # 手动4步Euler (比odeint快)
        for _ in range(4):
            dh = self.liquid_cell(torch.tensor(0.0), h, encoded)
            h = h + dt * dh

        return self.decoder(h)