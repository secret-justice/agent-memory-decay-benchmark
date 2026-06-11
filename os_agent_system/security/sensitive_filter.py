"""
安全合规模块
============
实现敏感信息识别、脱敏、加密和自然语言精准遗忘。

赛题要求⑤: 集成敏感信息识别与过滤功能，支持自然语言指令驱动的精准遗忘操作
"""

import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SensitivityLevel(str, Enum):
    """敏感度级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SensitiveMatch:
    """敏感信息匹配结果"""
    original: str           # 原始文本片段
    masked: str             # 脱敏后文本
    category: str           # 类别: id_card | phone | email | bank_card | name
    level: SensitivityLevel
    start: int              # 起始位置
    end: int                # 结束位置


class SensitiveFilter:
    """
    敏感信息过滤器
    
    三级脱敏:
    Level 1 - 规则脱敏: 正则匹配身份证/手机/银行卡/邮箱
    Level 2 - 语义脱敏: 识别姓名/地址等（Phase 2增加NER）
    Level 3 - 场景脱敏: 根据上下文判断敏感度
    """

    # 默认敏感模式
    DEFAULT_PATTERNS = {
        "id_card": {
            "pattern": r'(?<!\d)\d{17}[\dXx](?!\d)',
            "level": SensitivityLevel.CRITICAL,
            "mask": lambda m: m[:6] + "********" + m[-3:],
        },
        "phone": {
            "pattern": r'(?<!\d)1[3-9]\d{9}(?!\d)',
            "level": SensitivityLevel.HIGH,
            "mask": lambda m: m[:3] + "****" + m[-4:],
        },
        "bank_card": {
            "pattern": r'(?<!\d)\d{16,19}(?!\d)',
            "level": SensitivityLevel.CRITICAL,
            "mask": lambda m: m[:4] + " **** **** " + m[-4:],
        },
        "email": {
            "pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "level": SensitivityLevel.MEDIUM,
            "mask": lambda m: m[0] + "***@" + m.split("@")[1] if "@" in m else "***",
        },
        "ip_address": {
            "pattern": r'(?<!\d)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)',
            "level": SensitivityLevel.LOW,
            "mask": lambda m: "xxx.xxx.xxx.xxx",
        },
        "password_text": {
            "pattern": r'(?:password|passwd|pwd|\u5bc6\u7801)\s*(?:is|=|:|\u662f)\s*(\S+)',
            "level": SensitivityLevel.HIGH,
            "mask": lambda m: "****",
        },
        "api_key": {
            "pattern": r'(?:api[_\-]?key|apikey|secret[_\-]?key|access[_\-]?key)\s*(?:is|=|:)?\s*(\S{16,})',
            "category": "api_key",
            "level": "high",
            "severity": "high",
            "mask": lambda m: m[:4] + "****" + m[-4:]
        },
        "sk_key": {
            "pattern": r'(?:sk\-|sk_live_|sk_test_|pk_live_|pk_test_)[A-Za-z0-9]{10,}',
            "category": "sk_key",
            "level": "high",
            "severity": "high",
            "mask": lambda m: m[:3] + "****" + m[-4:]
        },
        "bearer_token": {
            "pattern": r'(?:bearer|token|auth)\s*(?:is|=|:)?\s*(eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+|[A-Za-z0-9_\-]{32,})',
            "category": "bearer_token",
            "level": "high",
            "severity": "high",
            "mask": lambda m: m[:6] + "****" + m[-4:]
        },
        "private_key": {
            "pattern": r'\-\-\-\-\-BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY\-\-\-\-\-',
            "category": "private_key",
            "level": "critical",
            "severity": "critical",
            "mask": lambda m: "****PRIVATE_KEY****"
        },
        "github_token": {
            "pattern": r'(?:ghp_|gho_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{10,}',
            "category": "github_token",
            "level": "high",
            "severity": "high",
            "mask": lambda m: m[:4] + "****" + m[-4:]
        },
        "aws_key": {
            "pattern": r'(?:AKIA|ASIA)[A-Z0-9]{16}',
            "category": "aws_key",
            "level": "critical",
            "severity": "critical",
            "mask": lambda m: m[:4] + "****" + m[-4:]
        },
    }

    def __init__(self, custom_patterns: Dict = None):
        self.patterns = dict(self.DEFAULT_PATTERNS)
        if custom_patterns:
            self.patterns.update(custom_patterns)

    def scan(self, text: str) -> List[SensitiveMatch]:
        """扫描文本中的敏感信息"""
        matches = []
        for category, config in self.patterns.items():
            for match in re.finditer(config["pattern"], text):
                original = match.group()
                masked = config["mask"](original)
                matches.append(SensitiveMatch(
                    original=original,
                    masked=masked,
                    category=category,
                    level=config["level"],
                    start=match.start(),
                    end=match.end(),
                ))
        return matches

    def mask(self, text: str, level: SensitivityLevel = None) -> Tuple[str, List[SensitiveMatch]]:
        """
        对文本进行脱敏处理
        
        Args:
            text: 原始文本
            level: 最低脱敏级别（None=全部脱敏）
            
        Returns:
            (脱敏后文本, 匹配列表)
        """
        matches = self.scan(text)
        if level:
            level_order = [SensitivityLevel.LOW, SensitivityLevel.MEDIUM,
                          SensitivityLevel.HIGH, SensitivityLevel.CRITICAL]
            min_idx = level_order.index(level)
            matches = [m for m in matches if level_order.index(m.level) >= min_idx]

        # 从后往前替换（避免位置偏移）
        result = text
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            result = result[:match.start] + match.masked + result[match.end:]

        return result, matches

    def is_safe(self, text: str) -> bool:
        """检查文本是否包含敏感信息（任何级别）"""
        matches = self.scan(text)
        return len(matches) == 0


class ForgettingEngine:
    """
    自然语言精准遗忘引擎
    
    支持的遗忘指令:
    - "忘记我的所有偏好" → 清除偏好
    - "忘记关于XX的知识" → 清除匹配知识
    - "删除我的搜索历史" → 清除交互记录
    - "清除所有数据" → 全量清除
    """

    def __init__(self, preference_engine=None, knowledge_engine=None,
                 memory_manager=None):
        self.preference_engine = preference_engine
        self.knowledge_engine = knowledge_engine
        self.memory_manager = memory_manager

    def forget(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        执行精准遗忘
        
        Returns:
            遗忘结果统计
        """
        result = {
            "preferences_deleted": 0,
            "knowledge_deleted": 0,
            "memories_deleted": 0,
            "query": query,
        }

        # 遗忘偏好
        if self.preference_engine:
            result["preferences_deleted"] = self.preference_engine.forget(user_id, query)

        # 遗忘知识（模糊匹配）
        if self.knowledge_engine:
            # 提取关键词
            keyword = re.sub(r"忘记|忘掉|删除|清除|关于|的|知识|记忆", "", query).strip()
            if keyword:
                # 检索匹配的知识
                search_results = self.knowledge_engine.search(keyword, top_k=10)
                for knowledge, score in search_results:
                    if score > 0.7:
                        self.knowledge_engine.delete(knowledge.id)
                        result["knowledge_deleted"] += 1

        logger.info(f"用户 {user_id} 精准遗忘: {result}")
        return result


