"""
记忆门控引擎
============
决定是否需要检索、选择检索策略、评估信息重要性。

Phase 1: 规则门控（关键词匹配+阈值）
Phase 2: LNN门控（连续时间ODE预测）

接口设计原则:
- 所有门控实现必须继承 BaseGate
- Phase 2 只需新增 LNNGate 类，替换 factory 返回值
- 输入输出格式不变，确保热替换
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

class GateDecision(str, Enum):
    """门控决策"""
    RETRIEVE = "retrieve"        # 需要检索
    SKIP = "skip"                # 跳过检索
    PROMOTE = "promote"          # 记忆升级（短期→中期→长期）
    DEMOTE = "demote"            # 记忆降级
    MAINTAIN = "maintain"        # 维持当前状态


@dataclass
class GateInput:
    """门控输入"""
    query: str                          # 当前查询
    user_id: str                        # 用户ID
    context: Optional[str] = None       # 上下文信息
    timestamp: Optional[float] = None   # 时间戳
    recent_interactions: int = 0        # 最近交互次数
    preference_stability: float = 0.5   # 偏好稳定性 (0-1)
    session_length: int = 0             # 当前会话长度


@dataclass
class GateOutput:
    """门控输出"""
    decision: GateDecision
    confidence: float           # 决策置信度 (0-1)
    retrieval_strategy: str     # 检索策略: vector | keyword | graph | hybrid
    importance_score: float     # 信息重要性 (0-1)
    drift_score: float          # 偏好漂移分数 (0-1, Phase 2 LNN核心输出)
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# 抽象基类（Phase 2 LNN必须实现此接口）
# ============================================================

class BaseGate(ABC):
    """
    门控引擎抽象基类
    
    Phase 1: RuleGate (规则门控)
    Phase 2: LNNGate (LNN连续时间ODE门控)
    
    LNNGate 实现要点:
    - 输入: GateInput 的 32 维特征向量 (见方案第三章)
    - 输出: GateOutput，其中 drift_score 由 LNN ODE 计算
    - 参数量: ~700 (门控组件)
    - 推理延迟: 目标 <5ms
    """

    @abstractmethod
    def predict(self, gate_input: GateInput) -> GateOutput:
        """
        门控预测
        
        Args:
            gate_input: 门控输入
            
        Returns:
            GateOutput: 包含决策、置信度、检索策略等
        """
        ...

    @abstractmethod
    def update(self, gate_input: GateInput, actual_outcome: bool):
        """
        在线更新（用于偏好漂移检测）
        
        Args:
            gate_input: 原始输入
            actual_outcome: 实际结果（检索是否有用）
        """
        ...

    @abstractmethod
    def get_drift_score(self, user_id: str) -> float:
        """
        获取用户偏好漂移分数
        
        Phase 1: 基于规则计算
        Phase 2: 由 LNN ODE 连续时间状态输出
        """
        ...


# ============================================================
# Phase 1: 规则门控
# ============================================================

class RuleGate(BaseGate):
    """
    规则门控引擎 (Phase 1)
    
    决策逻辑:
    1. 关键词匹配 → RETRIEVE
    2. 查询长度 > 阈值 → RETRIEVE
    3. 最近交互频繁 → SKIP（缓存命中概率高）
    4. 偏好稳定性低 → RETRIEVE（需要更新偏好）
    """

    # 关键词列表（触发检索的信号）
    RETRIEVE_KEYWORDS = {
        "怎么", "如何", "什么是", "请问", "帮我", "找一下",
        "配置", "设置", "安装", "卸载", "启动", "停止",
        "错误", "问题", "故障", "修复", "解决",
        "之前", "上次", "历史", "记得", "忘记",
        "偏好", "喜欢", "习惯", "常用", "端口", "密码",
        "编辑器", "版本", "地址", "IP", "ssh", "nginx",
        "防火墙", "规则", "备份", "部署", "数据库",
        "editor", "prefer", "port", "config", "deploy",
        "之前", "用的", "改成", "设为", "用什么",
    }

    # 关键词（触发跳过的信号：明显无需检索的短查询）
    SKIP_KEYWORDS = {
        "hi", "hello", "ok", "好的", "嗯", "谢谢", "thanks",
        "bye", "再见", "晚安", "知道了", "明白",
    }

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self._drift_scores: Dict[str, float] = {}  # user_id → drift_score
        self._interaction_counts: Dict[str, int] = {}

    def predict(self, gate_input: GateInput) -> GateOutput:
        """规则预测"""
        query = gate_input.query.lower()
        user_id = gate_input.user_id

        # 规则0: 明确跳过（短文本 + 无实质内容）
        is_skip = any(kw == query or query.startswith(kw) for kw in self.SKIP_KEYWORDS)
        is_trivial = len(query.strip()) <= 3

        # 规则1: 关键词匹配
        keyword_match = any(kw in query for kw in self.RETRIEVE_KEYWORDS)

        # 规则2: 查询长度（长查询通常需要检索）
        is_long_query = len(query) > 10

        # 规则3: 偏好稳定性（不稳定时需要检索更新）
        unstable = gate_input.preference_stability < 0.5

        # 综合决策
        if is_skip or is_trivial:
            decision = GateDecision.SKIP
            confidence = 0.9
            strategy = "none"
        elif keyword_match or is_long_query or unstable:
            decision = GateDecision.RETRIEVE
            confidence = 0.8 if keyword_match else 0.6
            strategy = "hybrid" if is_long_query else "vector"
        else:
            decision = GateDecision.SKIP
            confidence = 0.5
            strategy = "none"

        # 漂移分数（Phase 1简单计算）
        drift = self._compute_drift(user_id, gate_input)

        # 重要性分数
        importance = 0.5
        if keyword_match:
            importance += 0.2
        if unstable:
            importance += 0.2
        importance = min(1.0, importance)

        return GateOutput(
            decision=decision,
            confidence=confidence,
            retrieval_strategy=strategy,
            importance_score=importance,
            drift_score=drift,
            metadata={"keyword_match": keyword_match, "is_long": is_long_query}
        )

    def update(self, gate_input: GateInput, actual_outcome: bool):
        """更新交互计数和漂移分数"""
        user_id = gate_input.user_id
        self._interaction_counts[user_id] = self._interaction_counts.get(user_id, 0) + 1

        # 如果检索结果不被采纳，增加漂移分数
        if not actual_outcome:
            current = self._drift_scores.get(user_id, 0.0)
            self._drift_scores[user_id] = min(1.0, current + 0.05)
        else:
            current = self._drift_scores.get(user_id, 0.0)
            self._drift_scores[user_id] = max(0.0, current - 0.02)

    def get_drift_score(self, user_id: str) -> float:
        return self._drift_scores.get(user_id, 0.0)

    def _compute_drift(self, user_id: str, gate_input: GateInput) -> float:
        """计算漂移分数（Phase 1: 基于交互频率的简单估计）"""
        base_drift = self._drift_scores.get(user_id, 0.0)

        # 会话长度越长，漂移可能性越高
        session_factor = min(1.0, gate_input.session_length / 100.0)

        return min(1.0, base_drift + session_factor * 0.1)


# ============================================================
# Phase 2: LNN门控（预留接口）
# ============================================================

class LNNGate(BaseGate):
    """
    LNN连续时间ODE门控引擎 (Phase 2 - 已实现)
    
    架构: Input(32) → Encoder(16) → LiquidODE(16) → Decoder(3)
    参数量: ~700
    推理延迟: <5ms (CPU, 4步Euler)
    
    输出:
    - drift_score: 偏好漂移分数 (0-1)
    - retrieval_relevance: 检索相关度 (0-1)
    - importance_score: 信息重要性 (0-1)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "./data/lnn_gate.pth"
        self._model = None
        self._loaded = False
        self._drift_scores: Dict[str, float] = {}
        self._encoder = None
        self._load_model()

    def _load_model(self):
        """加载预训练LNN模型"""
        try:
            import torch
            from .lnn_model import LiquidNetwork
            from .lnn_trainer import LNNTrainer, GateFeatureEncoder

            if os.path.exists(self.model_path):
                self._model = LNNTrainer.load(self.model_path)
                self._model.eval()
                self._loaded = True
                self._encoder = GateFeatureEncoder
                logger.info(f"LNNGate 模型已加载: {self.model_path} (参数量: {self._model.count_parameters()})")
            else:
                logger.warning(f"LNNGate 模型文件不存在: {self.model_path}, 降级为规则门控")
        except ImportError:
            logger.warning("torch/torchdiffeq 未安装, LNNGate 降级为规则门控")
        except Exception as e:
            logger.error(f"LNNGate 加载失败: {e}")

    def _encode_features(self, gate_input: GateInput) -> "torch.Tensor":
        """将GateInput编码为32维特征张量"""
        import torch
        import numpy as np
        from .lnn_trainer import GateFeatureEncoder

        features = GateFeatureEncoder.encode(
            query=gate_input.query,
            user_id=gate_input.user_id,
            context=gate_input.context or "",
            timestamp=gate_input.timestamp or 0.0,
            recent_interactions=gate_input.recent_interactions,
            preference_stability=gate_input.preference_stability,
            session_length=gate_input.session_length,
        )
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, 32]

    def predict(self, gate_input: GateInput) -> GateOutput:
        """LNN预测"""
        if not self._loaded:
            # 降级: 使用规则门控
            fallback = RuleGate()
            result = fallback.predict(gate_input)
            result.metadata["fallback"] = True
            return result

        import torch
        import time

        start = time.time()
        features = self._encode_features(gate_input)

        with torch.no_grad():
            output = self._model.forward_inference(features)

        inference_ms = (time.time() - start) * 1000

        drift_score = output[0, 0].item()
        retrieval_relevance = output[0, 1].item()
        importance = output[0, 2].item()

        # 决策逻辑
        if drift_score > 0.6:
            decision = GateDecision.RETRIEVE
            strategy = "hybrid"
            confidence = 0.85
        elif retrieval_relevance > 0.5:
            decision = GateDecision.RETRIEVE
            strategy = "vector"
            confidence = 0.75
        elif importance > 0.6:
            decision = GateDecision.PROMOTE
            strategy = "vector"
            confidence = 0.7
        else:
            decision = GateDecision.SKIP
            strategy = "none"
            confidence = 0.8

        # 更新漂移缓存
        self._drift_scores[gate_input.user_id] = drift_score

        return GateOutput(
            decision=decision,
            confidence=confidence,
            retrieval_strategy=strategy,
            importance_score=importance,
            drift_score=drift_score,
            metadata={
                "retrieval_relevance": round(retrieval_relevance, 4),
                "inference_ms": round(inference_ms, 2),
                "model": "lnn",
            }
        )

    def update(self, gate_input: GateInput, actual_outcome: bool):
        """在线更新漂移分数"""
        user_id = gate_input.user_id
        current = self._drift_scores.get(user_id, 0.5)
        if not actual_outcome:
            self._drift_scores[user_id] = min(1.0, current + 0.03)
        else:
            self._drift_scores[user_id] = max(0.0, current - 0.01)

    def get_drift_score(self, user_id: str) -> float:
        """获取用户漂移分数"""
        return self._drift_scores.get(user_id, 0.0)


# ============================================================
# 工厂函数
# ============================================================

def create_gate(provider: str = "rule", **kwargs) -> BaseGate:
    """
    门控引擎工厂
    
    Phase 1: provider="rule" → RuleGate
    Phase 2: provider="lnn" → LNNGate
    """
    gates = {
        "rule": RuleGate,
        "lnn": LNNGate,
    }
    cls = gates.get(provider)
    if cls is None:
        raise ValueError(f"不支持的门控类型: {provider}")
    return cls(**kwargs)

