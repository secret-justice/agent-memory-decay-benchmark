"""
遗忘曲线衰减引擎
================
借鉴仿生记忆(Anamnesis)的遗忘曲线思想，融入我们的三级记忆管理:

核心公式: R(t) = e^(-t/S)
  R(t) = 记忆在时间 t 的保留强度 (0~1)
  t = 距离上次访问的时间 (秒)
  S = 稳定性常数 (越大遗忘越慢)

三层差异化衰减:
  短期: S=3600     (1小时半衰期)
  中期: S=2592000  (30天半衰期)
  长期: S=31536000 (365天半衰期)

增强机制:
  - 重复访问: S *= 1.5 (间隔效应)
  - 高重要性: S *= importance_boost
  - 被引用: S *= 1.2 (社会证明)

与比赛要求的映射:
  第6项 "短中长期记忆流转" - 衰减驱动自然流转
  第5项 "精准遗忘" - 低于阈值自动标记可遗忘
"""

import math
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class DecayConfig:
    """衰减配置"""
    # 基础稳定性常数 (秒)
    short_term_stability: float = 3600.0        # 1小时
    mid_term_stability: float = 2592000.0       # 30天
    long_term_stability: float = 31536000.0     # 365天

    # 衰减阈值
    forget_threshold: float = 0.2       # R < 此值标记为"可遗忘"
    promote_threshold: float = 0.7      # R > 此值可晋升
    demote_threshold: float = 0.3       # R < 此值应降级

    # 增强系数
    revisit_boost: float = 1.5          # 重复访问稳定性倍增
    importance_boost_base: float = 2.0  # 高重要性最大倍增
    reference_boost: float = 1.2        # 被其他记忆引用的倍增

    # 上限
    max_stability: float = 315360000.0  # 10年上限


# ============================================================
# 记忆强度追踪
# ============================================================

@dataclass
class MemoryStrength:
    """单条记忆的强度状态"""
    memory_id: str
    tier: str                           # short | mid | long
    stability: float = 0.0              # 当前稳定性常数 S
    last_access_ts: float = 0.0         # 最后访问时间戳
    access_count: int = 0               # 访问次数
    reference_count: int = 0            # 被引用次数
    importance: float = 0.5             # 原始重要性
    created_at_ts: float = 0.0          # 创建时间戳

    @property
    def age_seconds(self) -> float:
        """记忆年龄 (秒)"""
        return time.time() - self.created_at_ts

    @property
    def idle_seconds(self) -> float:
        """距离上次访问的空闲时间 (秒)"""
        return time.time() - self.last_access_ts


class ForgettingCurveEngine:
    """
    遗忘曲线衰减引擎

    核心功能:
    1. 计算任意记忆的当前保留强度 R(t)
    2. 记忆访问时更新稳定性 (间隔效应)
    3. 批量扫描需要降级/升级/遗忘的记忆
    4. 提供衰减统计和可观测指标
    """

    def __init__(self, config: DecayConfig = None):
        self._config = config or DecayConfig()
        self._memories: Dict[str, MemoryStrength] = {}

    # ============================================================
    # 核心公式
    # ============================================================

    def retention(self, memory: MemoryStrength) -> float:
        """
        计算记忆保留强度 R(t) = e^(-t/S)

        Returns:
            0.0 ~ 1.0 的保留强度
        """
        if memory.stability <= 0:
            return 0.0

        t = memory.idle_seconds
        S = memory.stability

        # R(t) = e^(-t/S)
        R = math.exp(-t / S)

        return max(0.0, min(1.0, R))

    def effective_stability(self, memory: MemoryStrength) -> float:
        """
        计算有效稳定性 (考虑各种增强因子)

        S_eff = S_base * revisit_factor * importance_factor * reference_factor
        """
        S = memory.stability

        # 访问次数增强 (间隔效应)
        if memory.access_count > 1:
            revisit_factor = self._config.revisit_boost ** min(memory.access_count - 1, 5)
            S *= revisit_factor

        # 重要性增强
        importance_factor = 1.0 + (self._config.importance_boost_base - 1.0) * memory.importance
        S *= importance_factor

        # 引用增强
        if memory.reference_count > 0:
            ref_factor = self._config.reference_boost ** min(memory.reference_count, 3)
            S *= ref_factor

        # 上限
        return min(S, self._config.max_stability)

    # ============================================================
    # 记忆管理
    # ============================================================

    def register(self, memory_id: str, tier: str, importance: float = 0.5,
                 created_at: float = None) -> MemoryStrength:
        """注册新记忆"""
        now = time.time()

        # 根据层级确定初始稳定性
        if tier == "short":
            base_S = self._config.short_term_stability
        elif tier == "mid":
            base_S = self._config.mid_term_stability
        else:
            base_S = self._config.long_term_stability

        strength = MemoryStrength(
            memory_id=memory_id,
            tier=tier,
            stability=base_S,
            last_access_ts=now,
            access_count=1,
            importance=importance,
            created_at_ts=created_at or now,
        )
        self._memories[memory_id] = strength
        return strength

    def access(self, memory_id: str) -> Optional[float]:
        """
        记忆被访问: 更新稳定性并返回当前 R(t)

        间隔效应: 每次访问稳定性 ×1.5
        """
        strength = self._memories.get(memory_id)
        if not strength:
            return None

        # 先计算当前 R (访问前)
        R_before = self.retention(strength)

        # 更新稳定性 (间隔效应)
        strength.stability = self.effective_stability(strength)
        strength.last_access_ts = time.time()
        strength.access_count += 1

        # 访问后 R 恢复到 1.0 (刚被访问)
        return 1.0

    def add_reference(self, memory_id: str) -> bool:
        """增加被引用计数"""
        strength = self._memories.get(memory_id)
        if not strength:
            return False
        strength.reference_count += 1
        # 重新计算稳定性
        strength.stability = self.effective_stability(strength)
        return True

    # ============================================================
    # 批量扫描
    # ============================================================

    def scan_actions(self) -> Dict[str, list]:
        """
        批量扫描所有记忆，返回需要执行的动作

        Returns:
            {
                "forget": [memory_ids...],      # R < forget_threshold
                "demote": [memory_ids...],      # R < demote_threshold
                "promote": [memory_ids...],     # R > promote_threshold
                "refresh": [memory_ids...],     # R 接近阈值，需要刷新
            }
        """
        cfg = self._config
        actions = {"forget": [], "demote": [], "promote": [], "refresh": []}

        for mid, strength in self._memories.items():
            R = self.retention(strength)

            if R < cfg.forget_threshold:
                actions["forget"].append(mid)
            elif R < cfg.demote_threshold:
                actions["demote"].append(mid)
            elif R > cfg.promote_threshold:
                actions["promote"].append(mid)
            elif abs(R - cfg.demote_threshold) < 0.1:
                actions["refresh"].append(mid)

        return actions

    def get_strength(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单条记忆的强度详情"""
        strength = self._memories.get(memory_id)
        if not strength:
            return None

        R = self.retention(strength)
        S_eff = self.effective_stability(strength)

        return {
            "memory_id": strength.memory_id,
            "tier": strength.tier,
            "retention": round(R, 4),
            "stability": round(strength.stability, 2),
            "effective_stability": round(S_eff, 2),
            "access_count": strength.access_count,
            "reference_count": strength.reference_count,
            "importance": strength.importance,
            "idle_hours": round(strength.idle_seconds / 3600, 2),
            "age_hours": round(strength.age_seconds / 3600, 2),
            "status": self._classify(R),
        }

    def _classify(self, R: float) -> str:
        """分类记忆状态"""
        if R < self._config.forget_threshold:
            return "forgettable"
        elif R < self._config.demote_threshold:
            return "fading"
        elif R > self._config.promote_threshold:
            return "strong"
        else:
            return "stable"

    # ============================================================
    # 统计
    # ============================================================

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        if not self._memories:
            return {"total": 0}

        retentions = [self.retention(s) for s in self._memories.values()]
        actions = self.scan_actions()

        return {
            "total": len(self._memories),
            "avg_retention": round(sum(retentions) / len(retentions), 4),
            "min_retention": round(min(retentions), 4),
            "max_retention": round(max(retentions), 4),
            "forgettable": len(actions["forget"]),
            "fading": len(actions["demote"]),
            "strong": len(actions["promote"]),
            "by_tier": {
                tier: sum(1 for s in self._memories.values() if s.tier == tier)
                for tier in ("short", "mid", "long")
            },
        }

    def half_life(self, tier: str) -> float:
        """计算某层级的半衰期 (秒)"""
        if tier == "short":
            S = self._config.short_term_stability
        elif tier == "mid":
            S = self._config.mid_term_stability
        else:
            S = self._config.long_term_stability
        # R(t) = 0.5 => t = S * ln(2)
        return S * math.log(2)


# ============================================================
# 时间感知检索增强
# ============================================================

def time_aware_boost(base_score: float, retention: float,
                     recency_weight: float = 0.15) -> float:
    """
    时间感知检索增强: 在基础分数上叠加时间衰减信号

    final_score = base_score * (1 - w) + retention * w

    Args:
        base_score: 原始检索分数 (BM25/向量/RRF)
        retention: 遗忘曲线保留强度 R(t)
        recency_weight: 时间信号权重 (0~1)

    Returns:
        增强后的分数
    """
    return base_score * (1 - recency_weight) + retention * recency_weight
