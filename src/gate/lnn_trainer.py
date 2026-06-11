"""
LNN训练管线 + 合成数据生成器
=============================
为记忆门控LNN生成偏好漂移训练数据并训练模型。

训练数据设计:
- 32维输入特征 → [drift_score, retrieval_relevance, importance] (3维标签)
- 模拟4种场景: 稳定偏好、缓慢漂移、快速漂移、新用户
"""

import os
import json
import math
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
from pathlib import Path

from .lnn_model import LiquidNetwork

logger = logging.getLogger(__name__)


# ============================================================
# 32维特征编码器 (GateInput → Feature Vector)
# ============================================================

class GateFeatureEncoder:
    """
    将 GateInput 编码为32维特征向量
    
    特征分布:
    [0:8]   - 查询统计 (长度、关键词匹配数、特殊字符比例等)
    [8:12]  - 时间特征 (小时sin/cos、工作日、距上次交互间隔)
    [12:20] - 上下文特征 (场景编码、对话轮次、主题变化)
    [20:24] - 用户状态 (历史交互数、偏好数量、漂移历史)
    [24:32] - 历史统计 (近期检索频率、命中率、偏好变化率)
    """

    # 关键词集 (与RuleGate保持一致)
    RETRIEVE_KEYWORDS = {
        "怎么", "如何", "什么是", "请问", "帮我", "找一下",
        "配置", "设置", "安装", "卸载", "启动", "停止",
        "错误", "问题", "故障", "修复", "解决",
        "之前", "上次", "历史", "记得", "忘记",
    }

    @staticmethod
    def encode(query: str, user_id: str = "", context: str = "",
               timestamp: float = 0.0, recent_interactions: int = 0,
               preference_stability: float = 0.5,
               session_length: int = 0,
               retrieval_hit_rate: float = 0.5,
               preference_change_rate: float = 0.0,
               scene: str = "default") -> np.ndarray:
        """
        编码为32维特征向量
        """
        features = np.zeros(32, dtype=np.float32)

        # [0:8] 查询统计
        features[0] = min(len(query) / 100.0, 1.0)
        kw_count = sum(1 for kw in GateFeatureEncoder.RETRIEVE_KEYWORDS if kw in query)
        features[1] = min(kw_count / 5.0, 1.0)
        features[2] = sum(1 for c in query if not c.isalnum() and not c.isspace()) / max(len(query), 1)
        features[3] = sum(1 for c in query if c.isascii()) / max(len(query), 1)
        features[4] = 1.0 if '?' in query or '？' in query else 0.0
        features[5] = len(set(query)) / max(len(query), 1)  # 字符多样性
        features[6] = min(query.count(' ') / 10.0, 1.0)
        features[7] = 1.0 if any(c.isdigit() for c in query) else 0.0

        # [8:12] 时间特征
        if timestamp > 0:
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            hour = dt.hour + dt.minute / 60.0
            features[8] = math.sin(2 * math.pi * hour / 24.0)
            features[9] = math.cos(2 * math.pi * hour / 24.0)
            features[10] = 1.0 if dt.weekday() < 5 else 0.0  # 工作日
            features[11] = hour / 24.0

        # [12:20] 上下文特征
        scene_map = {"default": 0, "coding": 1, "writing": 2, "sysadmin": 3, "research": 4}
        scene_idx = scene_map.get(scene, 0) / 4.0
        features[12] = scene_idx
        features[13] = min(len(context) / 200.0, 1.0)
        features[14] = min(session_length / 50.0, 1.0)
        features[15] = 1.0 if session_length > 10 else 0.0
        features[16] = len(context.split()) / max(len(context), 1) if context else 0.0
        features[17] = 1.0 if 'error' in context.lower() or '错误' in context else 0.0
        features[18] = 1.0 if 'config' in context.lower() or '配置' in context else 0.0
        features[19] = 0.5  # 预留

        # [20:24] 用户状态
        features[20] = min(recent_interactions / 100.0, 1.0)
        features[21] = preference_stability
        features[22] = min(abs(hash(user_id)) % 100 / 100.0, 1.0)  # 用户ID哈希
        features[23] = 1.0 if recent_interactions < 5 else 0.0  # 新用户

        # [24:32] 历史统计
        features[24] = retrieval_hit_rate
        features[25] = preference_change_rate
        features[26] = 1.0 - preference_stability  # 漂移倾向
        features[27] = min(retrieval_hit_rate * recent_interactions / 10.0, 1.0)
        features[28] = 0.5 if recent_interactions == 0 else min(
            preference_change_rate / max(recent_interactions, 1), 1.0)
        features[29] = 1.0 if preference_change_rate > 0.3 else 0.0
        features[30] = features[21] * features[24]  # stability × hit_rate
        features[31] = features[25] * features[20]  # change_rate × interaction_freq

        return features


# ============================================================
# 合成数据生成器
# ============================================================

class DriftScenarioGenerator:
    """
    生成4种偏好漂移场景的训练数据
    
    场景1: 稳定偏好 (drift=low, retrieval=low, importance=low)
    场景2: 缓慢漂移 (drift=med, retrieval=med, importance=med)
    场景3: 快速漂移 (drift=high, retrieval=high, importance=high)
    场景4: 新用户探索 (drift=med, retrieval=high, importance=high)
    """

    SAMPLE_QUERIES = [
        "帮我配置nginx", "如何安装Python", "之前用的编辑器是什么",
        "代码风格遵循pep8", "把输出改成JSON格式", "SSH端口改一下",
        "启动Docker服务", "git提交出错了", "文件权限怎么设置",
        "我记得上次用的是vim", "设置默认编译器", "网络连接失败",
        "数据库备份怎么做", "帮我找之前的配置", "加密级别调高一点",
        "清除我的搜索记录", "忘记我的工具偏好", "怎么优化性能",
    ]

    @staticmethod
    def generate_stable(n: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
        """场景1: 稳定偏好"""
        data = []
        for _ in range(n):
            query = random.choice(DriftScenarioGenerator.SAMPLE_QUERIES)
            features = GateFeatureEncoder.encode(
                query=query,
                user_id=f"user_{random.randint(1, 20)}",
                recent_interactions=random.randint(20, 100),
                preference_stability=random.uniform(0.7, 0.95),
                retrieval_hit_rate=random.uniform(0.6, 0.9),
                preference_change_rate=random.uniform(0.0, 0.1),
                session_length=random.randint(5, 30),
            )
            label = np.array([
                random.uniform(0.0, 0.2),  # drift: low
                random.uniform(0.0, 0.3),  # retrieval: low
                random.uniform(0.1, 0.4),  # importance: low
            ], dtype=np.float32)
            data.append((features, label))
        return data

    @staticmethod
    def generate_slow_drift(n: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
        """场景2: 缓慢漂移"""
        data = []
        for _ in range(n):
            query = random.choice(DriftScenarioGenerator.SAMPLE_QUERIES)
            features = GateFeatureEncoder.encode(
                query=query,
                user_id=f"user_{random.randint(1, 20)}",
                recent_interactions=random.randint(10, 50),
                preference_stability=random.uniform(0.3, 0.6),
                retrieval_hit_rate=random.uniform(0.4, 0.7),
                preference_change_rate=random.uniform(0.1, 0.3),
                session_length=random.randint(10, 50),
            )
            label = np.array([
                random.uniform(0.3, 0.6),  # drift: medium
                random.uniform(0.4, 0.7),  # retrieval: medium
                random.uniform(0.4, 0.7),  # importance: medium
            ], dtype=np.float32)
            data.append((features, label))
        return data

    @staticmethod
    def generate_fast_drift(n: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
        """场景3: 快速漂移"""
        data = []
        for _ in range(n):
            query = random.choice(DriftScenarioGenerator.SAMPLE_QUERIES)
            features = GateFeatureEncoder.encode(
                query=query,
                user_id=f"user_{random.randint(1, 20)}",
                recent_interactions=random.randint(5, 30),
                preference_stability=random.uniform(0.0, 0.3),
                retrieval_hit_rate=random.uniform(0.2, 0.5),
                preference_change_rate=random.uniform(0.3, 0.8),
                session_length=random.randint(20, 100),
            )
            label = np.array([
                random.uniform(0.7, 1.0),  # drift: high
                random.uniform(0.7, 1.0),  # retrieval: high
                random.uniform(0.6, 1.0),  # importance: high
            ], dtype=np.float32)
            data.append((features, label))
        return data

    @staticmethod
    def generate_new_user(n: int = 150) -> List[Tuple[np.ndarray, np.ndarray]]:
        """场景4: 新用户探索"""
        data = []
        for _ in range(n):
            query = random.choice(DriftScenarioGenerator.SAMPLE_QUERIES)
            features = GateFeatureEncoder.encode(
                query=query,
                user_id=f"new_user_{random.randint(100, 200)}",
                recent_interactions=random.randint(0, 5),
                preference_stability=random.uniform(0.1, 0.4),
                retrieval_hit_rate=random.uniform(0.0, 0.3),
                preference_change_rate=random.uniform(0.2, 0.6),
                session_length=random.randint(1, 10),
            )
            label = np.array([
                random.uniform(0.4, 0.8),  # drift: medium-high
                random.uniform(0.6, 1.0),  # retrieval: high (需要探索)
                random.uniform(0.5, 0.9),  # importance: medium-high
            ], dtype=np.float32)
            data.append((features, label))
        return data

    @staticmethod
    def generate_all(n_per_scenario: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成所有场景数据"""
        data = []
        data.extend(DriftScenarioGenerator.generate_stable(n_per_scenario))
        data.extend(DriftScenarioGenerator.generate_slow_drift(n_per_scenario))
        data.extend(DriftScenarioGenerator.generate_fast_drift(n_per_scenario))
        data.extend(DriftScenarioGenerator.generate_new_user(int(n_per_scenario * 0.75)))
        random.shuffle(data)
        return data


# ============================================================
# PyTorch Dataset
# ============================================================

class DriftDataset(Dataset):
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]]):
        self.features = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
        self.labels = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# 训练器
# ============================================================

class LNNTrainer:
    """LNN训练管线"""

    def __init__(self, model: LiquidNetwork, lr: float = 1e-3,
                 device: str = None):
        # CPU更快(模型仅1427参数), CUDA的ODE求解器有额外内存开销
        self.device = device or "cpu"
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        self.loss_fn = nn.MSELoss()
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.loss_fn(predictions, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0.0
        drift_errors = []
        retrieval_errors = []
        importance_errors = []
        n = 0

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(features)
                loss = self.loss_fn(predictions, labels)
                total_loss += loss.item()

                # 分维度误差
                drift_errors.append((predictions[:, 0] - labels[:, 0]).abs().mean().item())
                retrieval_errors.append((predictions[:, 1] - labels[:, 1]).abs().mean().item())
                importance_errors.append((predictions[:, 2] - labels[:, 2]).abs().mean().item())
                n += 1

        return {
            "loss": total_loss / max(n, 1),
            "drift_mae": sum(drift_errors) / max(n, 1),
            "retrieval_mae": sum(retrieval_errors) / max(n, 1),
            "importance_mae": sum(importance_errors) / max(n, 1),
        }

    def train(self, train_data: List[Tuple[np.ndarray, np.ndarray]],
              val_data: List[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 50, batch_size: int = 64) -> Dict[str, Any]:
        """
        完整训练流程
        
        Returns:
            训练结果字典
        """
        train_dataset = DriftDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data:
            val_dataset = DriftDataset(val_data)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        logger.info(f"开始训练 LNN | 设备: {self.device} | 参数量: {self.model.count_parameters()}")
        logger.info(f"训练集: {len(train_data)} | 验证集: {len(val_data) if val_data else 0}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.scheduler.step()

            metrics = {"epoch": epoch, "train_loss": train_loss}

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"早停: epoch {epoch}, val_loss={val_metrics['loss']:.4f}")
                    break

            self.history.append(metrics)

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | loss={train_loss:.4f}"
                if val_loader:
                    msg += f" | val_loss={metrics.get('val_loss', 0):.4f}"
                logger.info(msg)

        # 最终评估
        final_metrics = {}
        if val_loader:
            final_metrics = self.evaluate(val_loader)

        return {
            "epochs_trained": len(self.history),
            "best_val_loss": best_val_loss,
            "final_metrics": final_metrics,
            "param_count": self.model.count_parameters(),
            "history": self.history,
        }

    def save(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "output_dim": self.model.output_dim,
            "param_count": self.model.count_parameters(),
            "history": self.history,
        }, path)
        logger.info(f"模型已保存: {path}")

    @staticmethod
    def load(path: str, device: str = None) -> LiquidNetwork:
        """加载模型"""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = LiquidNetwork(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"模型已加载: {path} (参数量: {checkpoint['param_count']})")
        return model