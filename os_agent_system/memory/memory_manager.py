"""
三级记忆管理器
==============
实现短期/中期/长期记忆的存储、流转和管理。

Phase 1: 固定阈值流转
Phase 2: LNN连续时间ODE门控流转

记忆流转机制:
  短期(会话级,1h) → 中期(30天) → 长期(永久)
  触发条件: 访问频率 + 置信度 + 时间衰减
"""

import time
import uuid
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from src.search.bm25_scorer import BM25Scorer

logger = logging.getLogger(__name__)


class MemoryLevel(str, Enum):
    """记忆层级"""
    SHORT = "short_term"     # 短期: 当前会话, 1小时TTL
    MID = "mid_term"         # 中期: 近期记忆, 30天TTL
    LONG = "long_term"       # 长期: 永久记忆


@dataclass
class MemoryItem:
    """记忆条目"""
    id: str
    content: str
    level: MemoryLevel
    category: str            # preference | knowledge | interaction
    importance: float        # 重要性 (0-1)
    access_count: int        # 访问次数
    created_at: float        # 创建时间戳
    last_accessed: float     # 最后访问时间
    ttl: int                 # TTL秒数 (-1=永久)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """
    三级记忆管理器
    
    核心功能:
    1. 写入: 根据重要性分配到对应层级
    2. 读取: 跨层级统一检索
    3. 流转: 短期→中期→长期 自动晋升
    4. 清理: 过期记忆自动清理
    """

    def __init__(self, db_path: str = "./data/memory.db",
                 promotion_threshold: float = 0.7,
                 gate_engine=None):
        self.db_path = db_path
        self.promotion_threshold = promotion_threshold
        self.gate_engine = gate_engine  # Phase 2: LNN门控
        self._bm25 = BM25Scorer()
        self._bm25_dirty = True
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._rebuild_bm25()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                ttl INTEGER DEFAULT -1,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_level ON memories(level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(category)")
        conn.commit()
        conn.close()

    def _rebuild_bm25(self):
        """Rebuild BM25 index from all memories in DB."""
        if not self._bm25_dirty:
            return
        self._bm25 = BM25Scorer()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, content FROM memories").fetchall()
        conn.close()
        for row_id, content in rows:
            self._bm25.add(row_id, content)
        self._bm25_dirty = False

    def store(self, content: str, category: str = "interaction",
              importance: float = 0.5, level: MemoryLevel = None,
              metadata: Dict = None) -> MemoryItem:
        """
        存储记忆
        
        如果未指定level，根据importance自动分配:
        - importance < 0.3 → short_term
        - 0.3 <= importance < 0.7 → mid_term
        - importance >= 0.7 → long_term
        """
        now = time.time()
        metadata = metadata or {}

        if level is None:
            if importance >= 0.7:
                level = MemoryLevel.LONG
            elif importance >= 0.3:
                level = MemoryLevel.MID
            else:
                level = MemoryLevel.SHORT

        # TTL设置
        ttl_map = {
            MemoryLevel.SHORT: 3600,       # 1小时
            MemoryLevel.MID: 2592000,      # 30天
            MemoryLevel.LONG: -1,          # 永久
        }
        ttl = ttl_map[level]

        mid = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO memories (id, content, level, category, importance, access_count, created_at, last_accessed, ttl, metadata) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
            (mid, content, level.value, category, importance, now, now, ttl, json.dumps(metadata))
        )
        conn.commit()
        conn.close()

        self._bm25_dirty = True
        logger.debug(f"存储记忆 [{level.value}]: {content[:50]}...")

        return MemoryItem(
            id=mid, content=content, level=level, category=category,
            importance=importance, access_count=0,
            created_at=now, last_accessed=now, ttl=ttl, metadata=metadata
        )

    def retrieve(self, query: str, top_k: int = 5,
                 level: MemoryLevel = None,
                 category: str = None) -> List[MemoryItem]:
        """
        BM25-first retrieval: text relevance drives candidate selection.

        v3.1: BM25 search on full index (not SQL pre-filtered subset).
        This ensures textually relevant entries with low importance are found.
        """
        # Rebuild BM25 index if dirty
        self._rebuild_bm25()

        conn = sqlite3.connect(self.db_path)

        # Phase 1: BM25 search on FULL index for textually relevant candidates
        # Search more broadly than top_k to allow importance/recency to re-rank
        bm25_search_k = max(top_k * 5, 50)
        bm25_results = []
        if query and query.strip():
            bm25_results = self._bm25.search(query, top_k=bm25_search_k)

        if bm25_results:
            # Use BM25 results as primary candidates
            bm25_map = {doc_id: score for doc_id, score in bm25_results}
            candidate_ids = list(bm25_map.keys())

            # Fetch these specific entries from SQL (with optional filters)
            placeholders = ",".join(["?"] * len(candidate_ids))
            sql = f"SELECT * FROM memories WHERE id IN ({placeholders})"
            params = list(candidate_ids)
            if level:
                sql += " AND level = ?"
                params.append(level.value)
            if category:
                sql += " AND category = ?"
                params.append(category)

            rows = conn.execute(sql, params).fetchall()

            # Combine BM25 score with importance and recency
            import time as _time
            now = _time.time()
            scored_rows = []
            for row in rows:
                rid = row[0]
                bm25 = bm25_map.get(rid, 0.0)
                importance = row[5] if len(row) > 5 else 0.5
                last_acc = row[8] if len(row) > 8 else now
                recency = 1.0 / (1.0 + (now - last_acc) / 86400)
                # BM25 relevance is dominant; importance/recency are tiebreakers
                combined = bm25 * 3.0 + importance * 0.3 + recency * 0.2
                scored_rows.append((combined, row))

            scored_rows.sort(key=lambda x: x[0], reverse=True)
            rows = [r[1] for r in scored_rows[:top_k]]
        else:
            # No BM25 results or no query: fall back to importance/recency
            sql = "SELECT * FROM memories WHERE 1=1"
            params = []
            if level:
                sql += " AND level = ?"
                params.append(level.value)
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
            params.append(top_k)
            rows = conn.execute(sql, params).fetchall()

        # Update access count
        now = time.time()
        for row in rows:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, row[0])
            )

        conn.commit()
        conn.close()

        return [self._row_to_item(r) for r in rows]

    def promote(self, item_id: str, target_level: MemoryLevel) -> bool:
        """手动晋升记忆层级"""
        conn = sqlite3.connect(self.db_path)
        ttl_map = {
            MemoryLevel.SHORT: 3600,
            MemoryLevel.MID: 2592000,
            MemoryLevel.LONG: -1,
        }
        conn.execute(
            "UPDATE memories SET level = ?, ttl = ? WHERE id = ?",
            (target_level.value, ttl_map[target_level], item_id)
        )
        conn.commit()
        conn.close()
        return True

    def auto_promote(self) -> int:
        """
        自动流转: 检查短期/中期记忆，符合条件的自动晋升
        
        流转条件 (Phase 1 固定阈值):
        - 访问次数 >= 3 且 重要性 >= promotion_threshold
        
        Phase 2: 使用LNN门控的drift_score和importance_score决定流转
        """
        conn = sqlite3.connect(self.db_path)
        promoted = 0
        now = time.time()

        # 短期 → 中期
        rows = conn.execute(
            "SELECT id, importance, access_count FROM memories WHERE level = 'short_term' AND access_count >= 3 AND importance >= ?",
            (self.promotion_threshold,)
        ).fetchall()

        for row in rows:
            conn.execute(
                "UPDATE memories SET level = 'mid_term', ttl = 2592000 WHERE id = ?",
                (row[0],)
            )
            promoted += 1
            logger.info(f"记忆晋升: {row[0][:8]}... short_term → mid_term")

        # 中期 → 长期
        rows = conn.execute(
            "SELECT id, importance, access_count FROM memories WHERE level = 'mid_term' AND access_count >= 10 AND importance >= ?",
            (self.promotion_threshold + 0.1,)
        ).fetchall()

        for row in rows:
            conn.execute(
                "UPDATE memories SET level = 'long_term', ttl = -1 WHERE id = ?",
                (row[0],)
            )
            promoted += 1
            logger.info(f"记忆晋升: {row[0][:8]}... mid_term → long_term")

        conn.commit()
        conn.close()

        if promoted > 0:
            logger.info(f"自动流转完成: {promoted} 条记忆晋升")

        return promoted

    def cleanup(self) -> int:
        """清理过期记忆"""
        now = time.time()
        conn = sqlite3.connect(self.db_path)

        # 删除TTL已过期的记忆（ttl=-1的永久记忆不会被删除）
        count = conn.execute(
            "DELETE FROM memories WHERE ttl > 0 AND (? - created_at) > ttl",
            (now,)
        ).rowcount

        conn.commit()
        conn.close()

        if count > 0:
            logger.info(f"清理过期记忆: {count} 条")

        return count

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_level = conn.execute(
            "SELECT level, COUNT(*) FROM memories GROUP BY level"
        ).fetchall()
        by_category = conn.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        ).fetchall()
        conn.close()

        return {
            "total_memories": total,
            "by_level": {r[0]: r[1] for r in by_level},
            "by_category": {r[0]: r[1] for r in by_category},
        }

    def _row_to_item(self, row) -> MemoryItem:
        return MemoryItem(
            id=row[0], content=row[1], level=MemoryLevel(row[2]),
            category=row[3], importance=row[4], access_count=row[5],
            created_at=row[6], last_accessed=row[7], ttl=row[8],
            metadata=json.loads(row[9]) if row[9] else {}
        )

    # ================================================================
    # L1 Core Memory Snapshot (借鉴 Hermes 五层架构 L1)
    # ================================================================
    # 将高置信度、高访问频率的长时记忆冻结为结构化快照文本，
    # 可直接注入 system prompt，确保核心事实永不丢失。

    def get_core_snapshot(self, user_id: str = "default",
                          max_tokens: int = 800,
                          min_confidence: float = 0.7,
                          min_access: int = 2) -> str:
        """生成 L1 核心记忆快照。

        从长时记忆中提取置信度最高、访问最频繁的事实，
        按类别组织为结构化文本，用于 system prompt 注入。

        Args:
            user_id: 用户 ID
            max_tokens: 快照最大 token 数（近似，1 中文字 ≈ 1.5 token）
            min_confidence: 最低重要性阈值
            min_access: 最低访问次数

        Returns:
            结构化快照文本，格式如:
            [Core Memory Snapshot - user:default]
            ## Preferences
            - editor: vim (confidence=0.95, accessed 12 times)
            ## Knowledge
            - SSH default port is 22 (confidence=0.90, accessed 8 times)
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT content, category, importance, access_count, id
               FROM memories
               WHERE level = 'long_term'
                 AND importance >= ?
                 AND access_count >= ?
               ORDER BY importance DESC, access_count DESC
               LIMIT 50""",
            (min_confidence, min_access)
        ).fetchall()
        conn.close()

        if not rows:
            return ""

        # 按类别分组
        by_category: Dict[str, list] = {}
        for row in rows:
            cat = row[1]
            by_category.setdefault(cat, []).append(row)

        # 构建快照文本，控制 token 预算
        lines = [f"[Core Memory Snapshot - user:{user_id}]"]
        char_budget = int(max_tokens / 1.5)  # 近似中文字数
        used = 0

        category_labels = {
            "preference": "Preferences",
            "knowledge": "Knowledge",
            "interaction": "Interaction Patterns",
            "fact": "Key Facts",
            "workflow": "Workflows",
        }

        for cat, items in by_category.items():
            label = category_labels.get(cat, cat.title())
            header = f"## {label}"
            if used + len(header) > char_budget:
                break
            lines.append(header)
            used += len(header)

            for content, _, importance, access_count, item_id in items:
                line = f"- {content} (conf={importance:.2f}, used {access_count}x)"
                if used + len(line) > char_budget:
                    break
                lines.append(line)
                used += len(line)

        snapshot = "\n".join(lines)
        logger.debug(f"L1 snapshot generated: {len(lines)-1} facts, ~{used} chars")
        return snapshot

    def get_core_snapshot_dict(self, user_id: str = "default",
                               top_k: int = 20) -> List[Dict[str, Any]]:
        """返回 L1 核心记忆的结构化字典列表（供 API 使用）。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT id, content, category, importance, access_count
               FROM memories
               WHERE level = 'long_term' AND importance >= 0.7 AND access_count >= 2
               ORDER BY importance DESC, access_count DESC
               LIMIT ?""",
            (top_k,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "content": r[1], "category": r[2],
             "importance": r[3], "access_count": r[4]}
            for r in rows
        ]

    # ================================================================
    # Memory Lineage / Provenance (借鉴 Hermes L5 血缘追溯)
    # ================================================================
    # 每条记忆记录完整血缘链：原始来源 → 提取方式 → 入库决策 → 演化历史

    def _ensure_provenance_table(self):
        """确保 provenance 表存在（幂等）。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail TEXT DEFAULT '',
                source TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prov_memory ON provenance(memory_id)"
        )
        conn.commit()
        conn.close()

    def record_provenance(self, memory_id: str, event_type: str,
                          detail: str = "", source: str = "") -> int:
        """记录一条血缘事件。

        Args:
            memory_id: 关联的记忆 ID
            event_type: 事件类型 (created | accessed | promoted | demoted
                        | decayed | evolved | conflict_resolved | deleted)
            detail: 事件描述
            source: 来源标识（如 conversation_id, user_action, auto_flow）

        Returns:
            事件 ID
        """
        self._ensure_provenance_table()
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "INSERT INTO provenance (memory_id, event_type, detail, source, timestamp) VALUES (?, ?, ?, ?, ?)",
            (memory_id, event_type, detail, source, time.time())
        )
        event_id = cur.lastrowid
        conn.commit()
        conn.close()
        return event_id

    def get_lineage(self, memory_id: str) -> List[Dict[str, Any]]:
        """获取某条记忆的完整血缘链。

        Returns:
            按时间排序的事件列表，每条包含:
            - event_type, detail, source, timestamp
        """
        self._ensure_provenance_table()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT event_type, detail, source, timestamp FROM provenance WHERE memory_id = ? ORDER BY timestamp",
            (memory_id,)
        ).fetchall()
        conn.close()
        return [
            {"event_type": r[0], "detail": r[1], "source": r[2], "timestamp": r[3]}
            for r in rows
        ]

    def store_with_provenance(self, content: str, category: str = "interaction",
                               importance: float = 0.5, level: MemoryLevel = None,
                               metadata: Dict = None,
                               source: str = "",
                               provenance_detail: str = "") -> MemoryItem:
        """存储记忆并自动记录血缘。

        与 store() 相同，但额外写入 provenance 表。
        """
        item = self.store(content=content, category=category,
                          importance=importance, level=level, metadata=metadata)
        self.record_provenance(
            memory_id=item.id,
            event_type="created",
            detail=provenance_detail or f"Stored as {item.level.value}, importance={importance}",
            source=source,
        )
        return item
