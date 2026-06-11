"""
三级记忆分层系统
================
基于 deque / LRU / 持久化存储 实现短期 / 中期 / 长期记忆的分层管理。

分层策略:
  短期 (ShortTermMemory):  当前会话交互上下文，deque 容量 50，TTL 30 分钟
  中期 (MidTermMemory):    近期会话摘要与模式，LRU 容量 500，TTL 24 小时
  长期 (LongTermMemory):   用户偏好 / 知识 / 失败经验，向量数据库 + Markdown 持久化

流转规则 (由 memory_flow.MemoryFlowEngine 驱动):
  短期→中期: 访问 >= 3 次 且 存活 > 10 分钟
  中期→长期: 重要性 >= 0.7 且 访问 >= 5 次
  长期→遗忘: 重要性 < 0.2 且 30 天未访问
"""

import time
import uuid
import json
import logging
import sqlite3
import threading
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# MemoryItem 数据类
# =========================================================================
@dataclass
class MemoryItem:
    """
    记忆条目，贯穿三级记忆的统一数据结构。

    Attributes:
        id:          唯一标识
        content:     记忆内容文本
        category:    分类标签 (preference / knowledge / interaction / failure_experience)
        importance:  重要性评分 (0.0 - 1.0)
        created_at:  创建时间戳 (epoch 秒)
        accessed_at: 最后访问时间戳
        access_count: 累计访问次数
        tier:        当前所在层级 (short / mid / long)
        metadata:    附加元数据
    """

    id: str
    content: str
    category: str
    importance: float
    created_at: float
    accessed_at: float
    access_count: int
    tier: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # 序列化辅助
    # -----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """从字典反序列化。"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def age_seconds(self) -> float:
        """自创建以来经过的秒数。"""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """自上次访问以来经过的秒数。"""
        return time.time() - self.accessed_at

    def touch(self) -> None:
        """更新访问计数和时间戳。"""
        self.accessed_at = time.time()
        self.access_count += 1


# =========================================================================
# ShortTermMemory — 短期记忆
# =========================================================================
class ShortTermMemory:
    """
    短期记忆：基于 deque，存储当前会话的交互上下文。

    特性:
      - 固定容量 50 条，超出时自动淘汰最旧条目
      - TTL 30 分钟，过期条目由 expire_old() 清理
      - 线程安全 (内部加锁)
    """

    MAX_CAPACITY: int = 50
    TTL_SECONDS: int = 1800  # 30 分钟

    def __init__(self, capacity: int = MAX_CAPACITY, ttl: int = TTL_SECONDS):
        """
        初始化短期记忆。

        Args:
            capacity: 最大条目数
            ttl: 存活时间 (秒)
        """
        self._capacity: int = capacity
        self._ttl: int = ttl
        self._store: deque[MemoryItem] = deque(maxlen=capacity)
        self._lock: threading.Lock = threading.Lock()

    # -----------------------------------------------------------------
    # 公开接口
    # -----------------------------------------------------------------
    def store(self, item: MemoryItem) -> MemoryItem:
        """
        存入一条短期记忆。

        若已满则自动淘汰最旧条目。item.tier 会被强制设为 "short"。

        Args:
            item: 记忆条目

        Returns:
            存入后的 MemoryItem (已更新 tier)
        """
        with self._lock:
            item.tier = "short"
            # deque(maxlen=) 自动淘汰最旧
            self._store.append(item)
            logger.debug("短期记忆存入: %s (容量 %d/%d)", item.id[:8], len(self._store), self._capacity)
            return item

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """
        基于简单关键词匹配召回短期记忆。

        按访问次数 + 创建时间倒序排列，返回最相关的 top_k 条。

        Args:
            query: 查询文本 (关键词)
            top_k: 返回条数上限

        Returns:
            匹配的 MemoryItem 列表
        """
        with self._lock:
            query_lower = query.lower()
            scored: List[Tuple[float, MemoryItem]] = []
            for item in self._store:
                # 简单关键词命中评分
                if query_lower in item.content.lower():
                    score = item.access_count + 1.0
                else:
                    score = 0.0
                scored.append((score, item))

            # 优先按命中分 → 访问次数 → 创建时间
            scored.sort(key=lambda x: (x[0], x[1].access_count, x[1].created_at), reverse=True)

            results: List[MemoryItem] = []
            for _, item in scored[:top_k]:
                item.touch()
                results.append(item)
            return results

    def expire_old(self) -> int:
        """
        清理已超过 TTL 的条目。

        Returns:
            被清除的条目数
        """
        with self._lock:
            now = time.time()
            before = len(self._store)
            self._store = deque(
                (item for item in self._store if (now - item.created_at) < self._ttl),
                maxlen=self._capacity,
            )
            expired = before - len(self._store)
            if expired > 0:
                logger.debug("短期记忆过期清理: %d 条", expired)
            return expired

    def get_context_window(self, last_n: int = 10) -> List[MemoryItem]:
        """
        获取最近 N 条记忆作为上下文窗口。

        Args:
            last_n: 条目数

        Returns:
            最近的 MemoryItem 列表 (按时间正序)
        """
        with self._lock:
            items = list(self._store)[-last_n:]
            for item in items:
                item.touch()
            return items

    # -----------------------------------------------------------------
    # 辅助属性
    # -----------------------------------------------------------------
    @property
    def size(self) -> int:
        """当前条目数。"""
        return len(self._store)

    @property
    def is_full(self) -> bool:
        """是否已满。"""
        return len(self._store) >= self._capacity

    def get_all(self) -> List[MemoryItem]:
        """返回所有条目的浅拷贝。"""
        with self._lock:
            return list(self._store)

    def remove(self, item_id: str) -> bool:
        """
        按 ID 移除一条记忆。

        Args:
            item_id: 记忆 ID

        Returns:
            是否成功移除
        """
        with self._lock:
            for i, item in enumerate(self._store):
                if item.id == item_id:
                    del self._store[i]
                    return True
            return False


# =========================================================================
# MidTermMemory — 中期记忆
# =========================================================================
class MidTermMemory:
    """
    中期记忆：基于 LRU 缓存 (OrderedDict)，存储近期会话的摘要和模式。

    特性:
      - 固定容量 500 条，LRU 淘汰
      - TTL 24 小时，过期条目由 expire_old() 清理
      - consolidate() 将高频 / 高重要性条目标记为可晋升长期记忆
      - 线程安全
    """

    MAX_CAPACITY: int = 500
    TTL_SECONDS: int = 86400  # 24 小时

    def __init__(self, capacity: int = MAX_CAPACITY, ttl: int = TTL_SECONDS):
        """
        初始化中期记忆。

        Args:
            capacity: 最大条目数
            ttl: 存活时间 (秒)
        """
        self._capacity: int = capacity
        self._ttl: int = ttl
        self._store: OrderedDict[str, MemoryItem] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()

    # -----------------------------------------------------------------
    # 公开接口
    # -----------------------------------------------------------------
    def store(self, item: MemoryItem, importance_score: Optional[float] = None) -> MemoryItem:
        """
        存入一条中期记忆。

        若 importance_score 不为 None，则覆盖 item.importance。
        已存在相同 ID 的条目会被更新并移到最近使用位置。

        Args:
            item: 记忆条目
            importance_score: 可选的重要性覆盖评分

        Returns:
            存入后的 MemoryItem
        """
        with self._lock:
            item.tier = "mid"
            if importance_score is not None:
                item.importance = max(0.0, min(1.0, importance_score))

            if item.id in self._store:
                # 更新已有条目
                self._store[item.id] = item
                self._store.move_to_end(item.id)
            else:
                # 新条目，若满则淘汰最久未使用
                if len(self._store) >= self._capacity:
                    evicted_id, _ = self._store.popitem(last=False)
                    logger.debug("中期记忆 LRU 淘汰: %s", evicted_id[:8])
                self._store[item.id] = item

            logger.debug("中期记忆存入: %s (容量 %d/%d)", item.id[:8], len(self._store), self._capacity)
            return item

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """
        基于关键词匹配 + 重要性评分召回中期记忆。

        评分 = 关键词命中(2.0) + importance*3 + access_count*0.1

        Args:
            query: 查询文本
            top_k: 返回条数上限

        Returns:
            匹配的 MemoryItem 列表
        """
        with self._lock:
            query_lower = query.lower()
            scored: List[Tuple[float, MemoryItem]] = []
            for item in self._store.values():
                hit = 2.0 if query_lower in item.content.lower() else 0.0
                score = hit + item.importance * 3.0 + item.access_count * 0.1
                scored.append((score, item))

            scored.sort(key=lambda x: x[0], reverse=True)

            results: List[MemoryItem] = []
            for _, item in scored[:top_k]:
                item.touch()
                # LRU: 移到末尾
                self._store.move_to_end(item.id)
                results.append(item)
            return results

    def consolidate(self) -> List[MemoryItem]:
        """
        整合：筛选出满足晋升条件的记忆条目。

        条件: importance >= 0.7 且 access_count >= 5

        Returns:
            可晋升到长期记忆的 MemoryItem 列表 (不自动移除，由 FlowEngine 决定)
        """
        with self._lock:
            candidates: List[MemoryItem] = []
            for item in self._store.values():
                if item.importance >= 0.7 and item.access_count >= 5:
                    candidates.append(item)
            if candidates:
                logger.info("中期记忆整合: %d 条满足晋升条件", len(candidates))
            return candidates

    def expire_old(self) -> int:
        """
        清理超过 TTL 的条目。

        Returns:
            被清除的条目数
        """
        with self._lock:
            now = time.time()
            expired_ids: List[str] = []
            for item_id, item in self._store.items():
                if (now - item.created_at) >= self._ttl:
                    expired_ids.append(item_id)

            for item_id in expired_ids:
                del self._store[item_id]

            if expired_ids:
                logger.debug("中期记忆过期清理: %d 条", len(expired_ids))
            return len(expired_ids)

    # -----------------------------------------------------------------
    # 辅助属性
    # -----------------------------------------------------------------
    @property
    def size(self) -> int:
        """当前条目数。"""
        return len(self._store)

    @property
    def is_full(self) -> bool:
        """是否已满。"""
        return len(self._store) >= self._capacity

    def get_all(self) -> List[MemoryItem]:
        """返回所有条目的浅拷贝列表。"""
        with self._lock:
            return list(self._store.values())

    def remove(self, item_id: str) -> bool:
        """
        按 ID 移除一条记忆。

        Args:
            item_id: 记忆 ID

        Returns:
            是否成功移除
        """
        with self._lock:
            if item_id in self._store:
                del self._store[item_id]
                return True
            return False

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """
        按 ID 获取条目 (不更新 LRU 位置)。

        Args:
            item_id: 记忆 ID

        Returns:
            MemoryItem 或 None
        """
        with self._lock:
            return self._store.get(item_id)


# =========================================================================
# LongTermMemory — 长期记忆
# =========================================================================
class LongTermMemory:
    """
    长期记忆：基于 SQLite 持久化，支持向量数据库 + Markdown 双存储。

    特性:
      - 无容量硬限制 (磁盘约束)
      - 无 TTL (永久保留，除非 forget 主动遗忘)
      - get_profile() 聚合用户偏好 / 知识 / 经验画像
      - 线程安全
    """

    # 默认数据库路径
    DEFAULT_DB_PATH: str = "./data/long_term_memory.db"

    def __init__(self, db_path: str = DEFAULT_DB_PATH, md_dir: str = None):
        """
        初始化长期记忆。

        Args:
            db_path: SQLite 数据库路径
            md_dir:  Markdown 持久化目录，默认与 db_path 同目录下的 memories/
        """
        self.db_path: str = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._md_dir: Path = Path(md_dir) if md_dir else Path(db_path).parent / "memories"
        self._md_dir.mkdir(parents=True, exist_ok=True)

        self._lock: threading.Lock = threading.Lock()
        self._init_db()

    # -----------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------
    def _init_db(self) -> None:
        """创建持久化表结构。"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    tier TEXT DEFAULT 'long',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_memories(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_importance ON long_term_memories(importance)"
            )
            conn.commit()
            conn.close()

    # -----------------------------------------------------------------
    # 公开接口
    # -----------------------------------------------------------------
    def store(self, item: MemoryItem) -> MemoryItem:
        """
        持久化存入一条长期记忆。

        同时写入 SQLite 和 Markdown 文件。

        Args:
            item: 记忆条目

        Returns:
            存入后的 MemoryItem
        """
        item.tier = "long"
        item.touch()  # 更新访问时间

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT OR REPLACE INTO long_term_memories
                   (id, content, category, importance, created_at, accessed_at, access_count, tier, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.id,
                    item.content,
                    item.category,
                    item.importance,
                    item.created_at,
                    item.accessed_at,
                    item.access_count,
                    item.tier,
                    json.dumps(item.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()
            conn.close()

        # Markdown 持久化
        self._write_markdown(item)

        logger.debug("长期记忆存入: %s [%s]", item.id[:8], item.category)
        return item

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """
        检索长期记忆。

        使用关键词匹配 + 重要性 + 访问频率综合排序。

        Args:
            query: 查询文本
            top_k: 返回条数上限

        Returns:
            匹配的 MemoryItem 列表
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            query_lower = query.lower()

            # 拉取所有条目后在 Python 侧做模糊匹配 (SQLite 无 FTS 时的兼容方案)
            rows = conn.execute(
                "SELECT id, content, category, importance, created_at, accessed_at, access_count, tier, metadata FROM long_term_memories"
            ).fetchall()

            scored: List[Tuple[float, MemoryItem]] = []
            for row in rows:
                item = self._row_to_item(row)
                hit = 2.0 if query_lower in item.content.lower() else 0.0
                score = hit + item.importance * 3.0 + item.access_count * 0.1
                scored.append((score, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            results: List[MemoryItem] = []

            now = time.time()
            for _, item in scored[:top_k]:
                item.touch()
                results.append(item)
                # 回写访问统计
                conn.execute(
                    "UPDATE long_term_memories SET access_count = ?, accessed_at = ? WHERE id = ?",
                    (item.access_count, item.accessed_at, item.id),
                )

            conn.commit()
            conn.close()
            return results

    def forget(self, criteria: Dict[str, Any]) -> int:
        """
        按条件遗忘长期记忆。

        支持的 criteria 键:
          - importance_below: float  重要性低于该值
          - idle_days: int           未访问天数超过该值
          - category: str            指定分类
          - ids: List[str]           指定 ID 列表

        Args:
            criteria: 遗忘条件

        Returns:
            被删除的条目数
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conditions: List[str] = []
            params: List[Any] = []

            if "importance_below" in criteria:
                conditions.append("importance < ?")
                params.append(criteria["importance_below"])

            if "idle_days" in criteria:
                cutoff = time.time() - criteria["idle_days"] * 86400
                conditions.append("accessed_at < ?")
                params.append(cutoff)

            if "category" in criteria:
                conditions.append("category = ?")
                params.append(criteria["category"])

            if "ids" in criteria:
                placeholders = ",".join("?" for _ in criteria["ids"])
                conditions.append(f"id IN ({placeholders})")
                params.extend(criteria["ids"])

            if not conditions:
                conn.close()
                logger.warning("forget() 调用缺少有效条件，跳过")
                return 0

            where = " AND ".join(conditions)
            sql = f"DELETE FROM long_term_memories WHERE {where}"
            cursor = conn.execute(sql, params)
            count = cursor.rowcount
            conn.commit()
            conn.close()

            if count > 0:
                logger.info("长期记忆遗忘: %d 条 (条件: %s)", count, criteria)
            return count

    def get_profile(self) -> Dict[str, Any]:
        """
        聚合长期记忆画像。

        返回按分类统计的记忆分布、高重要性条目摘要、
        最近活跃条目等用户画像信息。

        Returns:
            用户画像字典
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)

            total = conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0]

            by_category = conn.execute(
                "SELECT category, COUNT(*) FROM long_term_memories GROUP BY category"
            ).fetchall()

            top_items = conn.execute(
                """SELECT id, content, category, importance, access_count
                   FROM long_term_memories
                   ORDER BY importance DESC, access_count DESC
                   LIMIT 10"""
            ).fetchall()

            recent_items = conn.execute(
                """SELECT id, content, category, accessed_at
                   FROM long_term_memories
                   ORDER BY accessed_at DESC
                   LIMIT 5"""
            ).fetchall()

            avg_importance = conn.execute(
                "SELECT AVG(importance) FROM long_term_memories"
            ).fetchone()[0] or 0.0

            conn.close()

            return {
                "total_memories": total,
                "by_category": {r[0]: r[1] for r in by_category},
                "avg_importance": round(avg_importance, 3),
                "top_memories": [
                    {"id": r[0][:8], "content": r[1][:80], "category": r[2], "importance": r[3]}
                    for r in top_items
                ],
                "recent_activity": [
                    {"id": r[0][:8], "content": r[1][:60], "category": r[2], "accessed_at": r[3]}
                    for r in recent_items
                ],
            }

    # -----------------------------------------------------------------
    # 辅助
    # -----------------------------------------------------------------
    def get_all(self) -> List[MemoryItem]:
        """返回全部长期记忆条目。"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT id, content, category, importance, created_at, accessed_at, access_count, tier, metadata FROM long_term_memories"
            ).fetchall()
            conn.close()
            return [self._row_to_item(r) for r in rows]

    def count(self) -> int:
        """条目总数。"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            n = conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0]
            conn.close()
            return n

    def _row_to_item(self, row: Tuple) -> MemoryItem:
        """将数据库行转换为 MemoryItem。"""
        return MemoryItem(
            id=row[0],
            content=row[1],
            category=row[2],
            importance=row[3],
            created_at=row[4],
            accessed_at=row[5],
            access_count=row[6],
            tier=row[7] or "long",
            metadata=json.loads(row[8]) if row[8] else {},
        )

    def _write_markdown(self, item: MemoryItem) -> None:
        """
        将记忆条目追加写入 Markdown 文件。

        按分类组织: memories/{category}.md
        """
        try:
            md_path = self._md_dir / f"{item.category}.md"
            ts = datetime.fromtimestamp(item.created_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            entry = (
                f"\n### {item.id[:8]} ({ts})\n"
                f"- **重要性**: {item.importance:.2f}\n"
                f"- **访问次数**: {item.access_count}\n"
                f"- **内容**: {item.content}\n"
            )
            if item.metadata:
                entry += f"- **元数据**: `{json.dumps(item.metadata, ensure_ascii=False)}`\n"

            with open(md_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as exc:
            logger.warning("写入 Markdown 失败: %s", exc)
