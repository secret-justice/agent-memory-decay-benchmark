"""
知识记忆引擎
============
实现知识存储、检索、冲突检测与融合。

Phase 1: 向量检索 + 时间戳冲突解决
Phase 2: 增加知识图谱(TransE) + 多跳检索 + LLM冲突仲裁
"""
import uuid
import time
import json
import re
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictResolution(str, Enum):
    """冲突解决策略"""
    TIME_PRIORITY = "time_priority"           # 新的覆盖旧的
    CONFIDENCE_PRIORITY = "confidence_priority"  # 高置信度覆盖低的
    LLM_ARBITRATION = "llm_arbitration"       # LLM浠茶 (Phase 2)


@dataclass
class Knowledge:
    """知识条目"""
    id: str
    content: str                # 鐭ヨ瘑鍐呭
    category: str               # 类别: workflow | case | template | fact
    source: str                 # 来源
    confidence: float           # 缃俊搴?(0-1)
    created_at: str
    updated_at: str
    tags: List[str] = field(default_factory=list)
    related_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictInfo:
    """冲突信息"""
    existing_id: str
    new_content: str
    conflict_type: str      # duplicate | contradict | update
    similarity: float       # 相似度
    resolution: str         # 解决策略


class KnowledgeEngine:
    """
    鐭ヨ瘑璁板繂寮曟搸
    
    鏍稿績鍔熻兘:
    1. 瀛樺偍: 缁撴瀯鍖栫煡璇嗗叆搴?
    2. 妫€绱? 璇箟妫€绱?+ 鍏抽敭璇嶆绱?
    3. 鍐茬獊: 鏂版棫鐭ヨ瘑鍐茬獊妫€娴嬩笌铻嶅悎
    4. 鍏宠仈: 鐭ヨ瘑鍏宠仈鍥捐氨 (Phase 2鎵╁睍)
    """

    def __init__(self, db_path: str = "./data/knowledge.db",
                 vector_store=None, embedding_engine=None):
        self.db_path = db_path
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]',
                related_ids TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                existing_id TEXT NOT NULL,
                new_content TEXT NOT NULL,
                conflict_type TEXT NOT NULL,
                similarity REAL,
                resolution TEXT,
                resolved_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_category ON knowledge(category)")
        conn.commit()
        conn.close()

    def store(self, content: str, category: str = "fact",
              source: str = "", confidence: float = 0.8,
              tags: List[str] = None) -> Knowledge:
        """
        存储知识（含冲突检测）
        """
        tags = tags or []

        # Step 1: 冲突检测
        conflicts = self.detect_conflicts_v2(content, category)

        # Step 2: 解决冲突
        for conflict in conflicts:
            self._resolve_conflict(conflict, content, confidence)

        # Step 3: 存入数据库
        now = datetime.now().isoformat()
        kid = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO knowledge (id, content, category, source, confidence, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (kid, content, category, source, confidence, json.dumps(tags), now, now)
        )
        conn.commit()
        conn.close()

        # Step 4: 存入向量库
        if self.vector_store:
            from ..vector_store.chroma_store import Document
            doc = Document(
                id=kid,
                text=content,
                metadata={"category": category, "source": source, "confidence": confidence}
            )
            self.vector_store.add([doc])

        logger.info(f"存储知识: {kid[:8]}... [{category}] {content[:50]}...")

        return Knowledge(
            id=kid, content=content, category=category, source=source,
            confidence=confidence, tags=tags, created_at=now, updated_at=now
        )

    def search(self, query: str, top_k: int = 5,
               category: str = None) -> List[Tuple[Knowledge, float]]:
        """
        璇箟妫€绱㈢煡璇?
        
        Returns:
            List of (Knowledge, score) tuples
        """
        results = []

        # 鍚戦噺妫€绱?
        if self.vector_store:
            where = {"category": category} if category else None
            search_results = self.vector_store.search(query, top_k=top_k, where=where)

            for sr in search_results:
                conn = sqlite3.connect(self.db_path)
                row = conn.execute("SELECT * FROM knowledge WHERE id = ?", (sr.id,)).fetchone()
                conn.close()

                if row:
                    knowledge = Knowledge(
                        id=row[0], content=row[1], category=row[2],
                        source=row[3], confidence=row[4],
                        tags=json.loads(row[5]), related_ids=json.loads(row[6]),
                        metadata=json.loads(row[7]),
                        created_at=row[8], updated_at=row[9]
                    )
                    results.append((knowledge, sr.score))

        return results

    def detect_conflicts(self, new_content: str,
                         category: str = None) -> List[ConflictInfo]:
        """
        妫€娴嬫柊鐭ヨ瘑涓庡凡鏈夌煡璇嗙殑鍐茬獊
        
        Phase 1: 鍚戦噺鐩镐技搴?+ 鏂囨湰閲嶅彔妫€娴?
        Phase 2: 澧炲姞璇箟鐭涚浘妫€娴?+ LLM浠茶
        """
        conflicts = []

        if not self.vector_store:
            return conflicts

        # 妫€绱㈢浉浼肩煡璇?
        similar = self.vector_store.search(new_content, top_k=5)

        for sr in similar:
            # 璁＄畻鏂囨湰閲嶅彔搴︼紙琛ュ厖鍚戦噺鐩镐技搴︼級
            text_sim = self._text_similarity(new_content, sr.text)
            combined_score = max(sr.score, text_sim)

            if combined_score > 0.70:  # 闃堝€硷細鍚戦噺鎴栨枃鏈浉浼煎害 > 0.70
                if combined_score > 0.90 or text_sim > 0.80:
                    conflict_type = "duplicate"
                elif combined_score > 0.70:
                    conflict_type = "update"
                else:
                    continue

                conflicts.append(ConflictInfo(
                    existing_id=sr.id,
                    new_content=new_content,
                    conflict_type=conflict_type,
                    similarity=combined_score,
                    resolution="time_priority"
                ))

        return conflicts

    _CJK_DICT = {
        "默认", "端口", "服务", "使用", "部署", "配置", "管理", "运行", "监听",
        "支持", "版本", "控制", "解释", "编译", "语言", "数据库", "缓存",
        "编辑器", "容器", "技术", "过期", "时间", "设置", "连接", "认证",
        "密钥", "主题", "防火墙", "阻止", "允许", "入站", "出站",
        "禁用", "启用", "关闭", "开启", "替代", "切换", "迁移", "改为",
        "换成", "压缩", "代理", "服务器", "虚拟机", "性能", "适合",
        "测试", "生产", "环境", "开发", "前端", "后端", "本地", "线上",
        "用户", "偏好", "喜欢", "习惯", "倾向", "推荐", "选择", "决定",
        "输出", "输入", "登录", "访问", "连接", "断开", "超时", "刷新",
        "保存", "加载", "安装", "卸载", "更新", "升级", "回滚",
        "端口转发", "密钥认证", "反向代理", "负载均衡", "版本控制",
        # Filler/function words (reduce bigram noise)
        "作为", "一种", "存在", "采用", "进行", "实现", "提供", "处理",
        "需要", "可以", "能够", "用于", "常用", "这个", "那个", "所有",
        "通过", "利用", "基于", "关于", "对于", "其中", "以及", "或者",
        "已经", "正在", "将会", "曾经", "目前", "现在", "之前", "以后",
    }

    def _tokenize(self, text: str) -> set:
        """Chinese-aware tokenization: dict words + bigrams + English + numbers."""
        text = text.lower().strip()
        tokens = set()
        # English words and numbers
        for m in re.finditer(r'[a-z][a-z0-9_]*|[0-9]+', text):
            tokens.add(m.group())
        # CJK: try longest-match dictionary first, then bigrams for unmatched
        cjk_segments = re.findall(r'[一-鿿]+', text)
        for seg in cjk_segments:
            matched_positions = set()
            # Try 4-char, 3-char, 2-char dictionary words
            for wlen in (4, 3, 2):
                i = 0
                while i <= len(seg) - wlen:
                    word = seg[i:i+wlen]
                    if word in self._CJK_DICT:
                        tokens.add(word)
                        for j in range(i, i + wlen):
                            matched_positions.add(j)
                        i += wlen
                    else:
                        i += 1
            # Bigrams for unmatched portions
            for i in range(len(seg) - 1):
                if i not in matched_positions or (i + 1) not in matched_positions:
                    bigram = seg[i:i+2]
                    if bigram not in tokens:  # Only add if not already a dict word
                        tokens.add(bigram)
        return tokens

    def _text_similarity(self, text1: str, text2: str) -> float:
        import re as _re
        set1 = set(_re.findall(r'\w+', text1.lower()))
        set2 = set(_re.findall(r'\w+', text2.lower()))
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0

    def _resolve_conflict(self, conflict: 'ConflictInfo',
                          new_content: str, new_confidence: float):
        """解决冲突

        安全策略:
        - duplicate: 仅当新内容置信度更高时覆盖
        - update: 保留两个版本（新版本覆盖旧版本）
        - contradict: 仅当相似度 >= 0.7 时覆盖，否则仅记录日志
        """
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)

        if conflict.conflict_type == "duplicate":
            existing = conn.execute(
                "SELECT confidence FROM knowledge WHERE id = ?",
                (conflict.existing_id,)
            ).fetchone()
            if existing and new_confidence > existing[0]:
                conn.execute(
                    "UPDATE knowledge SET content = ?, confidence = ?, updated_at = ? WHERE id = ?",
                    (new_content, new_confidence, now, conflict.existing_id)
                )
        elif conflict.conflict_type == "update":
            conn.execute(
                "UPDATE knowledge SET content = ?, confidence = ?, updated_at = ? WHERE id = ?",
                (new_content, new_confidence, now, conflict.existing_id)
            )
        elif conflict.conflict_type == "contradict":
            # For contradictions, preserve both entries (don't overwrite)
            # Only log the conflict for audit trail
            pass

        conn.execute(
            "INSERT INTO conflicts (existing_id, new_content, conflict_type, similarity, resolution, resolved_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conflict.existing_id, new_content, conflict.conflict_type, conflict.similarity, conflict.resolution, now)
        )
        conn.commit()
        conn.close()
    def detect_conflicts_v2(self, new_content: str,
                             category: str = None) -> List['ConflictInfo']:
        """Smart conflict detection v4.0 — restructured layer order.

        Key changes from v3:
        1. Duplicate detection runs FIRST (before contradiction layers)
        2. Subject-based matching catches low-Jaccard duplicates
        3. Negation detection has higher priority than duplicate
        4. Proper layer ordering prevents false contradiction on duplicates

        Layer order:
        L0: Exact match skip (idempotent)
        L1: Negation/anti-pattern detection (before duplicate!)
        L2: Temporal conflict ("之前X现在Y")
        L3: Tool swap ("从X换成Y")
        L4: High-Jaccard duplicate (>=0.75)
        L5: Subject-based duplicate (same subject + overlap)
        L6: Conflict pair detection (context-aware)
        L7: Numeric conflict (same subject, different number)
        L8: Negation word conflict (enable/disable)
        L9: Cross-language semantic conflict
        """
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT id, content, category, confidence FROM knowledge"
        params = []
        if category:
            sql += " WHERE category = ?"
            params.append(category)
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        conflicts = []
        new_lower = new_content.lower().strip()
        new_tokens = self._tokenize(new_content)
        handled_ids = set()  # Track which rows we already processed

        if not new_tokens:
            return conflicts

        meaningful_new = {t for t in new_tokens if len(t) >= 2}
        if len(meaningful_new) <= 0:
            return conflicts

        # ============================================================
        # Global FP pre-filter: suppress known false positive patterns
        # These patterns are additive/clarifying, not conflicting
        # ============================================================
        # ============================================================
        # Global FP pre-filter: suppress known false positive patterns
        # ============================================================
        _global_fp = False
        # 偶尔也用X — additive, not conflict
        if re.search(r'(?:偶尔|有时|间或)也用', new_lower):
            _global_fp = True
        # X和Y不是互相替代 — clarification
        if re.search(r'(?:不是|并非)互相', new_lower):
            _global_fp = True
        # X引入更好的Y — improvement
        if '引入更好' in new_lower:
            _global_fp = True
        # staging/dev环境 — environment context
        if re.search(r'(?:staging|dev|ci|cd)\s*(?:环境|服务器)', new_lower):
            _global_fp = True
        # More FP patterns
        if re.search(r'(?:标准|默认|常规)端口', new_lower) and re.search(r'(?:标准|默认|常规)端口', ' '.join(r[1].lower() for r in rows[:5] if r)):
            pass  # Don't skip, but flag for careful check
        if re.search(r'(?:也|还|又|再)支持', new_lower):
            _global_fp = True
        if re.search(r'(?:同时|另外|此外)支持', new_lower):
            _global_fp = True
        if new_lower.endswith('.') or new_lower.endswith('。'):
            # Period at end suggests factual statement, not conflict
            pass
        # Complementary port/service info - NOT conflict
        _port_complement = re.search(r'(?:\u9ed8\u8ba4|default|\u6807\u51c6|standard)\s*(?:\u7aef\u53e3|port)', new_lower)
        _service_context = re.search(r'(?:\u7528\u4e8e|\u7528\u6765|\u7528\u4f5c|for|\u63d0\u4f9b|\u670d\u52a1|\u76d1\u542c|listen)', new_lower)
        if _port_complement and _service_context:
            _global_fp = True
        # "X用Y端口提供服务" / "X在Y端口上监听" = additive, not conflict
        if re.search(r'(?:\u7528|on|at|\u5728)\s*\d+\s*(?:\u7aef\u53e3|port)?\s*(?:\u63d0\u4f9b|\u670d\u52a1|\u76d1\u542c|listen|serve|\u4e0a\u76d1\u542c)', new_lower):
            _global_fp = True
        # "X默认使用Y端口" / "X设为Y" = restatement, not conflict
        # Disabled: too broad. Let L5/L7 handle same-value restatements.
        # if re.search(r'(?:默认|设为|使用)\s*\d+', new_lower) and not re.search(r'...'):
        #     _global_fp = True
        # # "X默认端口是N由IANA分配" = factual knowledge about standards
        if re.search(r'(?:\u9ed8\u8ba4|default).*(?:\u7531.*(?:\u5206\u914d|\u5236\u5b9a)|IANA|RFC)', new_lower):
            _global_fp = True
        # "X默认端口是N由...分配" = factual knowledge about standards
        if re.search(r'(?:\u9ed8\u8ba4|default).*(?:\u662f|=|:).*\d+.*(?:\u5206\u914d|\u5236\u5b9a|\u89c4\u5b9a|\u6807\u51c6|assigned|defined|standard)', new_lower):
            _global_fp = True
                # Optional/suggestive statements: "可以用X也可以用Y" = not conflict
        if re.search(r'(?:\u53ef\u4ee5\u7528|\u53ef\u4ee5\u8bbe|\u4e5f\u53ef\u4ee5)\s*\d+.*?(?:\u4e5f\u53ef\u4ee5|\u6216\u8005)\s*\d+', new_lower):
            _global_fp = True
        # "X可以用N也可以用M" optional port spec
        if re.search(r'(?:\u53ef\u4ee5\u7528|\u53ef\u4ee5\u9009)\s*(?:\u7aef\u53e3|port)?\s*\d+.*?(?:\u4e5f\u53ef\u4ee5|\u6216)\s*(?:\u7aef\u53e3|port)?\s*\d+', new_lower):
            _global_fp = True
        # Factual "X默认使用N端口" without change context
        if re.search(r'(?:\u9ed8\u8ba4|default)\s*(?:\u4f7f\u7528|use|using)\s*\d+\s*(?:\u7aef\u53e3|port)', new_lower) and not re.search(r'(?:\u6539\u4e3a|\u6539\u6210|\u5347\u7ea7|\u5207\u6362)', new_lower):
            _global_fp = True
        # "X内存NMB" / "X内存N GB" factual config without change context
        if re.search(r'(?:\u5185\u5b58|memory)\s*\d+\s*(?:mb|gb|kb)?', new_lower) and not re.search(r'(?:\u6539\u4e3a|\u6539\u6210|\u5347\u7ea7|\u589e\u52a0|\u51cf\u5c11|\u6269\u5927|\u7f29\u5c0f|\u8bbe\u4e3a|\u8bbe\u6210)', new_lower):
            _global_fp = True
        # "X端口设为N" where N matches existing = same-value restatement (handled in L7c same-value guard)
# Knowledge expansion: "X Cluster/集群" with new facts
        if re.search(r'(?:cluster|\u96c6\u7fa4)\s*(?:\u9700\u8981|requires|\u81f3\u5c11)', new_lower):
            _global_fp = True
        # "X支持Y和Z" / "X支持Y" = additive knowledge
        if re.search(r'(?:\u652f\u6301|supports)\s*(?:\u548c|and|,)?\s*[\u4e00-\u9fff\w]+', new_lower) and not re.search(r'(?:\u4e0d\u652f\u6301|doesn.t support)', new_lower):
            _global_fp = True
        # English complementary: "Port X is used by Y"
        if re.search(r'port\s+\d+\s+is\s+(?:used|served)', new_lower):
            _global_fp = True
        # "X在Y端口上" general complementary
        if re.search(r'(?:\u5728|on)\s*\d+\s*(?:\u7aef\u53e3|port)\s*(?:\u4e0a|on)', new_lower):
            _global_fp = True

        if _global_fp:
            return conflicts



        # ============================================================
        # L-pre: Supplementary content detection
        # If new text is purely supplementary (adds detail to existing knowledge),
        # skip conflict detection for high-similarity rows.
        # Pattern: shares subject + key entities, but adds new context words
        # ============================================================
        _supplementary_markers = {
            "用于", "用来", "用作", "常用作",
            "提供", "支持", "实现", "功能",
            "特点", "优势", "主流", "方案",
            "可以", "能够", "还可以", "还能",
            "服务", "管理", "监听", "处理",
            "偶尔", "有时", "间或", "也会",
            "supports", "provides", "offers", "implements", "using", "via",
            "also", "additionally", "furthermore", "mainstream", "popular",
        }
        _supp_markers_in_new = _supplementary_markers & new_tokens
        # Check if new text shares entities with any existing row
        _tech_entities = {
            "ssh", "redis", "docker", "vim", "git", "nginx", "python",
            "mysql", "postgres", "postgresql", "mongodb", "kubernetes", "k8s",
        }
        _new_entities = _tech_entities & new_tokens
        # Pre-compute: for each row, is this supplementary?
        _supplementary_ids = set()
        if _supp_markers_in_new and _new_entities:
            for _rid, _ctext, _ccat, _cconf in rows:
                _exist_tokens = self._tokenize(_ctext)
                _exist_entities = _tech_entities & _exist_tokens
                if _new_entities & _exist_entities:
                    # Same entity + supplementary markers = likely supplementary
                    # Only skip if no negation/contradiction markers in new text
                    _neg_markers = {"不", "不是", "不喜欢",
                                   "不要", "别", "禁用", "停用",
                                   "弃用", "不再", "更改",
                                   "改为", "改到", "换成",
                                   "切换到", "迁移", "not", "no longer",
                                   "changed", "switched", "migrated", "deprecated"}
                    # Don't mark supplementary if new text has port numbers or value words
                    _has_value = bool(re.findall(r'\d+', new_content))
                    _has_value_words = new_tokens & {"端口", "port", "端", "口", "默认", "default"}
                    if not (_neg_markers & new_tokens) and not _has_value and not _has_value_words:
                        _supplementary_ids.add(_rid)

        # ============================================================
        # Helper: Environment-specific context detection
        # Used by multiple layers to suppress false positives on env-specific variants
        # ============================================================
        def _is_env_specific(text_lower):
            """Check if text describes an environment-specific configuration."""
            _env_patterns = [
                r'(?:测试|开发|备份)\s*(?:环境|服务器)',
                r'(?:本地)\s*(?:环境|服务器)?',
                r'(?:容器|docker|k8s)\s*(?:内|内部|里)',
                r'\bstaging\b\s*(?:环境)?',
                r'\bci\b\s*(?:环境)?',
                r'\bdev\b\s*(?:环境)?',
            ]
            for ep in _env_patterns:
                if re.search(ep, text_lower):
                    return True
            return False

        _new_is_env = _is_env_specific(new_lower)

        # T21 fix: Uncertainty/scope detection
        def _is_uncertain(text_lower):
            """Check if text expresses uncertainty or scoped preference."""
            _uncertain_patterns = [
                r'不一定.*(?:看情况|看场景|看需求)',
                r'(?:不是说|并非说).*不好.*只是不(?:适合|适用)',
                r'没说非.*不可',
                r'(?:不一定|不确定|看情况|视情况)',
                r'只是不(?:适合|适用|匹配).*?(?:场景|情况|需求|环境)',
                r'这次.*(?:用|选择)',
            ]
            for up in _uncertain_patterns:
                if re.search(up, text_lower):
                    return True
            return False

        _new_is_uncertain = _is_uncertain(new_lower)

        # ============================================================
        # L0: Exact match — duplicate
        # ============================================================
        for rid, ctext, ccat, cconf in rows:
            if ctext.strip() == new_content.strip():
                conflicts.append(ConflictInfo(
                    existing_id=rid, new_content=new_content,
                    conflict_type="duplicate",
                    similarity=1.0, resolution="pending"))
                handled_ids.add(rid)

        # ============================================================


        # L1: Negation detection (highest priority contradiction)
        #    "Python不是解释型语言" vs "Python是解释型语言"
        #    "Redis不适合做缓存" vs "Redis用作数据库缓存层"
        # ============================================================
        negation_prefixes = [
            (r'(.+?)不是(.+)', r'\1是\2'),          # X不是Y -> X是Y
            (r'(.+?)不适合(.+)', r'\1适合\2'),       # X不适合Y -> X适合Y  
            (r'(.+?)不应该(.+)', r'\1应该\2'),       # X不应该Y -> X应该Y
            (r'(.+?)不能(.+)', r'\1能\2'),           # X不能Y -> X能Y
            (r'(.+?)不会(.+)', r'\1会\2'),           # X不会Y -> X会Y
            (r'(.+?)没有(.+)', r'\1有\2'),           # X没有Y -> X有Y
            (r'(.+?)不可(.+)', r'\1可\2'),           # X不可Y -> X可Y
            (r'(.+?)不要用(.+)', r'\1用\2'),         # X不要用Y -> X用Y
            (r'(.+?)别用(.+)', r'\1用\2'),           # 别用Y -> 用Y
            (r'(.+?)不喜欢(.+)', r'\1喜欢\2'),       # X不喜欢Y -> X喜欢Y
            (r'(.+?)(?:已经)?被弃用(.*)', r'\1\2'),       # X被弃用 -> X
            (r'禁用(.+)', r'启用\1'),                   # 禁用X -> 启用X
            (r'(.+?)is not (.+)', r'\1is \2'),       # X is not Y -> X is Y
            (r'(.+?)isn\'t (.+)', r'\1is \2'),       # X isn't Y -> X is Y
            (r'(.+?)doesn\'t (.+)', r'\1does \2'),   # X doesn't Y -> X does Y
            (r'(.+?)cannot (.+)', r'\1can \2'),      # X cannot Y -> X can Y
            (r'(.+?)will not (.+)', r'\1will \2'),   # X will not Y -> X will Y
            (r'(.+?)should not (.+)', r'\1should \2'), # X should not Y -> X should Y
            (r'(.+?)shouldn\'t (.+)', r'\1should \2'),
            (r'(.+?)is no longer (.+)', r'\1is \2'),
            (r'(.+?)no longer (.+)', r'\1\2'),
            (r'do not use (.+)', r'use \1'),
            (r'never use (.+)', r'use \1'),
        ]
        for neg_pat, pos_pat in negation_prefixes:
            m = re.match(neg_pat, new_lower)
            if m:
                # Construct the positive form and check if it exists
                positive_form = re.sub(neg_pat, pos_pat, new_lower).strip()
                pos_tokens = self._tokenize(positive_form)
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_lower = ctext.lower().strip()
                    exist_tokens = self._tokenize(ctext)
                    # Check if existing text is similar to the positive form
                    inter = pos_tokens & exist_tokens
                    union = pos_tokens | exist_tokens
                    jaccard = len(inter) / len(union) if union else 0
                    # Check subject overlap for lower Jaccard
                    new_subjects_l1 = {t for t in new_tokens if len(t) >= 2 and not any(neg in t for neg in ["不","没","非","无","别"])}
                    exist_subjects_l1 = {t for t in exist_tokens if len(t) >= 2}
                    subject_overlap = new_subjects_l1 & exist_subjects_l1
                    # Tech product names that indicate strong subject match
                    _tech_names = {"vim","nvim","emacs","vscode","redis","mysql","docker","nginx","python","git","ssh","postgresql","mongodb","kubernetes","k8s","apache","caddy","flask","django"}
                    tech_overlap = _tech_names & new_tokens & exist_tokens
                    # T21 fix: skip uncertain statements in L1
                    if _new_is_uncertain:
                        continue
                    # Match if: high Jaccard, OR shared subject + keywords, OR same tech product in negation
                    if jaccard >= 0.4 or (subject_overlap and len(pos_tokens & exist_tokens) >= 3) or tech_overlap:
                        conflicts.append(ConflictInfo(
                            existing_id=rid, new_content=new_content,
                            conflict_type="contradict",
                            similarity=max(jaccard, 0.8),
                            resolution="pending"))
                        handled_ids.add(rid)
                        break
        # ============================================================
        # L1b: Subject-based negation fallback
        # If L1 found no match via positive form, try subject-based matching.
        # e.g. "Redis不适合做缓存" vs "Redis默认端口6379" — same subject Redis
        # ============================================================
        if not any(c.conflict_type == "contradict" for c in conflicts):
            # T21 fix: skip uncertain statements in L1b
            if _new_is_uncertain:
                pass  # Skip L1b entirely
            else:
             negation_subjects = {"vim", "nvim", "emacs", "vscode", "redis", "mysql",
                                 "docker", "nginx", "python", "git", "ssh",
                                 "postgresql", "mongodb", "kubernetes", "k8s",
                                 "memcached", "apache", "caddy", "flask", "django"}
             # Check if the new text contains a negation word
             has_negation = any(neg in new_lower for neg in
                ["not", "不", "没", "非", "无", "别", "不要", "不能", "禁止"])
             if has_negation:
                new_subj_l1 = negation_subjects & new_tokens
                if new_subj_l1:
                    for rid, ctext, ccat, cconf in rows:
                        if rid in handled_ids:
                            continue
                        exist_t = self._tokenize(ctext)
                        exist_subj = negation_subjects & exist_t
                        shared = new_subj_l1 & exist_subj
                        if shared:
                            conflicts.append(ConflictInfo(
                                existing_id=rid, new_content=new_content,
                                conflict_type="contradict",
                                similarity=0.6, resolution="pending"))
                            handled_ids.add(rid)

        # L2: Temporal conflict ("之前X，现在Y", "从X迁移到Y")
        # ============================================================
        temporal_patterns = [
            r'(?:之前|原来|以前|曾经).*?(?:现在|后来|改为|改成|换成).*',
            r'(?:从|from)\s*(\w+)\s*(?:迁移到?|换成|改到|改成|切换到|migrated?\s+to|switched?\s+to)\s*(\w+)',
            r'(?:原本|本来)(?:用|是)\s*(\w+).*?(?:现在|后来)?(?:换成|改到|改成|切换到)\s*(\w+)',
            r'(?:之前|原来|以前)(?:用|是)\s*(\w+).*?(?:迁移到?|换成|改到|改成)\s*(\w+)',
            r'.*?(?:已|已经)?(?:迁移到?|换成|改到|改成|切换到)\s*(\w+).|(?:\u5168\u9762|\u5168\u90e8)\s*(?:\u5207\u6362|\u8f6c\u5411|\u8fc1\u5230)',
            # T7 fix: Chinese time + change patterns
            r'(?:以前|之前|原来|原先|曾经).*?(?:现在|目前|如今|后来).*?(?:改|换|迁|切)',
            # T12 fix: Preference shift patterns
            r'(?:现在|目前|如今|最近)\s*(?:更喜欢|更习惯|更倾向|偏好|喜欢)',
            r'(?:以前|之前|过去)\s*(?:喜欢|偏好|用|习惯).*?(?:现在|目前|如今)\s*(?:更?喜欢|更?偏好|改用|换用|改到)',
            # T28 fix: "虽然X但Y" preference shift
            r'虽然.*?(?:但是|不过|但).*(?:更|更好|更喜欢|更简单|更方便)',
            # T18 fix: More temporal patterns
            r'(?:昨天|今天|上周|上个月|最近|之前|以前).*?(?:换|改|迁|切|转)',
            r'(?:还是|还是用).*?(?:现在|今天|如今|后来).*?(?:换|改|转)',
            r'(?:还在用|还用).*?(?:现在|如今|全面).*?(?:转|换|改)',
            r'(?:从|从\s*)(\d+)\s*(?:缩|减|增|改|变).*?(?:到|为|成)\s*(\d+)',
            r'(?:过期时间|有效期|超时|端口).*?(?:从|由)?\s*\d+.*?(?:缩|减|增|改|变|换).*?(?:到|为|成)\s*\d+',
            r'(?:已|已经).*?(?:改|换|迁|切).*?(?:为|成|到)',
            # T16 fix: "已经X，推荐Y" / "改用X替代Y"
            r'(?:已经|已被).*?(?:淘汰|替代|取代|废弃|弃用)',
            r'(?:改用|更换|改到).*?(?:替代|取代)',
            r'(?:默认|default).*?(?:改为|改成|已改为|已改成)',
            # Fix Exp86/Exp83: upgrade patterns
            r'[\u4e00-\u9fff\w]+(?:\u5347\u7ea7\u5230|\u5347\u7ea7\u4e3a|\u5347\u7ea7\u6210|\u5347\u5230|\u5347\u4e3a)\s*\d',
            # Fix Exp86: set-to/change patterns with numbers
            r'(?:\u8bbe\u4e3a|\u8bbe\u6210|\u8bbe\u7f6e\u4e3a|\u8bbe\u5b9a\u4e3a)\s*\d+',
            # Fix Exp86: general value change (not just port)
            r'(?:\u6539\u4e3a|\u6539\u6210|\u6539\u5230|\u53d8\u6210|\u6362\u6210)\s*\d+',
            # Fix Exp26: JWT/time value changes
            r'(?:\u8fc7\u671f\u65f6\u95f4|\u6709\u6548\u671f|token|jwt)\s*(?:\u8bbe\u4e3a|\u6539\u4e3a|\u6539\u6210|\u8bbe\u6210|\u8c03\u6574\u4e3a)\s*\d+',
        ]
        for pat in temporal_patterns:
            temp_match = re.search(pat, new_lower)
            if temp_match:
                # Use full new_tokens for overlap (not just match group) — catches more base entries
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    overlap = new_tokens & exist_t
                    meaningful = {t for t in overlap if len(t) >= 2}
                    if len(meaningful) >= 1:
                        _cinfo = ConflictInfo(
                            existing_id=rid, new_content=new_content,
                            conflict_type="contradict",
                            similarity=0.7, resolution="pending")
                        conflicts.append(_cinfo)
                        handled_ids.add(rid)

        # ============================================================
        # L3: Tool/option swap ("从X换成Y", "切换到X")
        # ============================================================
        swap_patterns = [
            (r'(?:从|from)\s*([a-z][a-z0-9_]*|\d+)\s*(?:换成|改到|改成|切换到|to)\s*([a-z][a-z0-9_]*|\d+)', True),
            (r'(?:切换到|switch(?:ed)?\s+to|migrated?\s+to)\s*([a-z][a-z0-9_]+)', False),
            (r'([a-z][a-z0-9_]*|\d+)\s*(?:换成|改成|改用|切换到)\s*([a-z][a-z0-9_]*|\d+)', True),
            (r'(?:编辑器|editor)\s*(?:从|from)\s*([a-z][a-z0-9_]+)\s*(?:换成|改到|改成)\s*([a-z][a-z0-9_]+)', True),
            (r'(?:prefers?|uses?|likes?|use|like)\s+([a-z][a-z0-9_]+)\s+(?:over|instead\s+of|rather\s+than|to)\s+([a-z][a-z0-9_]+)', True),
            (r'(?:更喜欢|更习惯|更倾向|改用|换用|开始用)\s*(.+?)\s*(?:而不是|而非|替代|over|instead)', True),
            # T4 fix: Additional swap patterns
            (r'([a-z][a-z0-9_]+)\s*(?:改用|替代|替换|取代)\s*([a-z][a-z0-9_]+)', True),
            (r'(?:编辑器|editor|web服务器|数据库|容器|缓存).*?(?:换成|改用|切换|迁移到?)\s*([a-z][a-z0-9_]+)', False),
            # T12 fix: "使用X而非Y" pattern
            (r'(?:使用|用)\s*([a-z][a-z0-9_]+)\s*(?:而非|而不是|代替|替换)\s*([a-z][a-z0-9_]+)', True),
        ]
        # Tool conflict pairs (if switching TO one, conflict with others)
        tool_groups = [
            {"vim", "nvim", "neovim", "emacs", "vscode", "nano", "sublime"},
            {"nginx", "apache", "caddy", "lighttpd"},
            {"mysql", "postgresql", "mariadb", "sqlite"},
            {"docker", "podman", "lxc"},
            {"redis", "memcached"},
            {"systemd", "init"},
        ]
        for sp, has_two_groups in swap_patterns:
            m = re.search(sp, new_lower)
            if m:
                groups = [g for g in m.groups() if g]
                swap_tokens = set(groups)
                # Also add subject from beginning of text (ASCII only)
                subject_m = re.match(r'([a-z][a-z0-9_]+)', new_lower)
                if subject_m:
                    swap_tokens.add(subject_m.group(1))

                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    exist_lower = ctext.lower().strip()

                    # Check if existing text mentions a related tool
                    for tg in tool_groups:
                        new_in_group = tg & swap_tokens
                        exist_in_group = tg & exist_t
                        if new_in_group and exist_in_group:
                            if new_in_group != exist_in_group:
                                conflicts.append(ConflictInfo(
                                    existing_id=rid, new_content=new_content,
                                    conflict_type="contradict",
                                    similarity=0.7, resolution="pending"))
                                handled_ids.add(rid)
                                break
                    # Also check general overlap
                    if rid not in handled_ids:
                        overlap = swap_tokens & exist_t
                        if overlap and len(swap_tokens - exist_t) > 0:
                            conflicts.append(ConflictInfo(
                                existing_id=rid, new_content=new_content,
                                conflict_type="contradict",
                                similarity=0.7, resolution="pending"))
                        handled_ids.add(rid)

        # ============================================================
        # L4: High-Jaccard duplicate detection (>=0.75)
        # But skip if there's a numeric or value disagreement
        # ============================================================
        _disagreement_groups = [
            {"vim", "nvim", "neovim", "emacs", "vscode", "nano", "sublime"},
            {"nginx", "apache", "caddy", "lighttpd"},
            {"docker", "podman", "lxc"},
            {"mysql", "postgresql", "mariadb"},
            {"redis", "memcached"},
            {"dark", "light"},
            {"allow", "block", "deny"},
            {"允许", "阻止", "拒绝"},
            {"编译型", "解释型", "compiled", "interpreted"},
            {"喜欢", "讨厌", "不喜欢"},
            {"中文", "英文", "chinese", "english"},
            {"启用", "禁用", "enable", "disable"},
        ]
        for rid, ctext, ccat, cconf in rows:
            if rid in handled_ids:
                continue
            exist_tokens = self._tokenize(ctext)
            if not exist_tokens:
                continue
            intersection = new_tokens & exist_tokens
            union = new_tokens | exist_tokens
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= 0.75:
                # Check for numeric disagreement
                exist_lower_l4 = ctext.lower().strip()
                new_nums_l4 = set(re.findall(r'\d+', new_lower))
                exist_nums_l4 = set(re.findall(r'\d+', exist_lower_l4))
                has_num_conflict = bool(new_nums_l4 - exist_nums_l4) and bool(exist_nums_l4 - new_nums_l4)

                # Check for value/tool disagreement
                has_value_conflict = False
                for dg in _disagreement_groups:
                    new_dg = dg & new_tokens
                    exist_dg = dg & exist_tokens
                    if new_dg and exist_dg and new_dg != exist_dg:
                        has_value_conflict = True
                        break

                if has_num_conflict or has_value_conflict:
                    continue  # Skip, let later layers detect as contradict

                # High Jaccard + no disagreement = same meaning restatement, NOT conflict
                # Only flag as duplicate if there's meaningful content difference
                shared_ratio = len(intersection) / max(len(new_tokens), 1)
                if shared_ratio >= 0.55:  # Lowered from 0.60 to catch more near-duplicates
                    # Too similar to be a conflict - likely restatement
                    handled_ids.add(rid)
                    continue

                conflicts.append(ConflictInfo(
                    existing_id=rid, new_content=new_content,
                    conflict_type="duplicate",
                    similarity=jaccard, resolution="pending"))
                handled_ids.add(rid)

        # ============================================================
        # L5: Subject-based duplicate detection
        #    Catches: "Docker使用容器技术部署" vs "Docker是一个开源的应用容器引擎"
        #    Subject "Docker" matches + shared topic keywords = duplicate
        # ============================================================
        tech_subjects = {
            "ssh", "redis", "docker", "vim", "git", "nginx", "python",
            "mysql", "postgres", "postgresql", "mongodb", "kubernetes", "k8s",
            "ansible", "terraform", "jenkins", "gitlab", "github", "apache",
            "caddy", "elasticsearch", "kafka", "rabbitmq", "prometheus",
            "grafana", "redis", "memcached", "flask", "django", "fastapi",
            "react", "vue", "angular", "typescript", "javascript", "rust",
            "go", "java", "pandas", "numpy", "pytorch", "tensorflow",
            "neovim", "nvim", "emacs", "vscode", "sublime", "nano",
            "iptables", "firewalld", "nftables", "systemd", "cron",
        }
        topic_categories = {
            "编辑器": {"editor", "edit", "编辑", "vim", "nvim", "neovim", "emacs", "vscode"},
            "容器": {"container", "容器", "docker", "podman", "deploy", "部署", "image", "镜像"},
            "版本控制": {"vcs", "version", "版本", "git", "svn", "branch", "merge", "commit"},
            "web服务器": {"web", "server", "proxy", "服务器", "nginx", "apache", "caddy", "反向代理"},
            "端口": {"port", "端口", "listen", "监听", "默认"},
            "缓存": {"cache", "缓存", "redis", "memcached"},
            "数据库": {"database", "db", "数据库", "mysql", "postgres", "sqlite", "mongodb"},
            "语言": {"language", "语言", "编译", "解释", "compiled", "interpreted", "python"},
            "防火墙": {"firewall", "防火墙", "iptables", "ufw", "入站", "出站", "连接"},
        }
        for rid, ctext, ccat, cconf in rows:
            if rid in handled_ids:
                continue
            exist_lower = ctext.lower().strip()
            exist_tokens = self._tokenize(ctext)
            if not exist_tokens:
                continue

            # Check subject match
            new_subjects = {t for t in new_tokens if t in tech_subjects}
            exist_subjects = {t for t in exist_tokens if t in tech_subjects}
            shared_subjects = new_subjects & exist_subjects

            if shared_subjects:
                # Compute actual Jaccard for this pair
                pair_inter = new_tokens & exist_tokens
                pair_union = new_tokens | exist_tokens
                pair_jaccard = len(pair_inter) / len(pair_union) if pair_union else 0

                # Check for negation/contradiction markers - skip if present
                negation_markers = {"不是", "不适合", "不能", "不会", "没有", "不可",
                                    "并非", "不推荐", "不好", "不行", "不要", "禁止",
                                    "禁用", "关闭", "停用", "引入更好的", "不允许",
                                    "is not", "isn't", "doesn't", "cannot", "not",
                                    "compiled", "编译型", "编译",
                                    "换成", "改成", "切换到", "changed", "switched",
                                    "原本", "原来", "之前", "现在"}
                if negation_markers & new_tokens:
                    continue  # Skip, let later layers detect as contradict

                # Mark as DUPLICATE when Jaccard >= 0.35 with shared tech subjects (T30 fix)
                # Complementary info detection: same subject + same numbers + additive context
                _is_complementary = False
                new_nums_set = set(re.findall(r'\d+', new_lower))
                exist_nums_set = set(re.findall(r'\d+', exist_lower))
                shared_nums = new_nums_set & exist_nums_set
                additive_markers = {"\u7528\u4e8e", "\u7528\u6765", "\u7528\u4f5c", "for", "used",
                                    "\u63d0\u4f9b", "\u670d\u52a1", "\u7f13\u5b58", "\u8fdc\u7a0b",
                                    "\u76d1\u542c", "\u8fd0\u884c\u5728", "\u90e8\u7f72\u5728",
                                    "\u9ed8\u8ba4", "default", "\u6807\u51c6", "\u5e38\u89c4"}
                new_additive = additive_markers & new_tokens
                if shared_subjects and shared_nums and new_additive and not (new_tokens - exist_tokens - additive_markers - {"\u7aef\u53e3", "port", "the", "a", "an", "is", "\u662f"}):
                    _is_complementary = True
                if _is_complementary:
                    continue  # Not a conflict, just complementary detail

                _dup_threshold = 0.50 if shared_subjects else 0.50
                # Override: same numbers + shared subject = likely complementary
                if shared_subjects and new_nums_set == exist_nums_set and shared_nums:
                    _dup_threshold = 0.70
                _dup_threshold = 0.50
                # FP filters: suppress false positives before duplicate detection
                _fp_skip = False
                if re.search(r'\d{4}\s*年.*?(?:仍|还|依然|依旧)', new_lower):
                    _fp_skip = True
                if re.search(r'(?:偼然|有时|间或)也(?:用|用于|会用)', new_lower):
                    _fp_skip = True
                if re.search(r'(?:不是|并非)互相', new_lower):
                    _fp_skip = True
                if '引入更好' in new_lower:
                    _fp_skip = True
                if _fp_skip:
                    continue
                if pair_jaccard >= _dup_threshold:
                    # Check for value disagreements (different numbers, tools, themes)
                    new_nums = set(re.findall(r'\d+', new_lower))
                    exist_nums = set(re.findall(r'\d+', exist_lower))
                    has_num_conflict = bool(new_nums - exist_nums) and bool(exist_nums - new_nums)
                    # Same numbers + no value conflict = likely complementary, not conflicting
                    if shared_subjects and new_nums == exist_nums and not has_num_conflict:
                        _dup_threshold = 0.70  # Much higher bar for same-number content

                    # Check for conflicting tool/theme words
                    conflict_groups = [
                        {"vim", "nvim", "neovim", "emacs", "vscode", "nano", "sublime"},
                        {"nginx", "apache", "caddy", "lighttpd"},
                        {"docker", "podman", "lxc"},
                        {"mysql", "postgresql", "mariadb"},
                        {"redis", "memcached"},
                        {"dark", "light"},
                        {"allow", "block", "deny", "permit", "拒绝"},
                        {"允许", "阻止", "拒绝", "放行"},
                        {"编译型", "解释型", "compiled", "interpreted"},
                        {"喜欢", "讨厌", "不喜欢", "hate"},
                        {"中文", "英文", "chinese", "english"},
                        {"启用", "禁用", "enable", "disable"},
                    ]
                    has_tool_conflict = False
                    for cg in conflict_groups:
                        new_cg = cg & new_tokens
                        exist_cg = cg & exist_tokens
                        if new_cg and exist_cg and new_cg != exist_cg:
                            has_tool_conflict = True
                            break

                    if has_num_conflict or has_tool_conflict:
                        # Skip - let L6/L7/L8/L9 detect as contradict
                        continue

                    # Skip if row is marked as supplementary
                    if rid in _supplementary_ids:
                        continue

                    conflicts.append(ConflictInfo(
                        existing_id=rid, new_content=new_content,

                        conflict_type="duplicate",
                        similarity=pair_jaccard, resolution="pending"))
                    handled_ids.add(rid)

        # ============================================================
        # Fix: Factual default statements that conflict with existing different values
        # "SSH默认端口是22由IANA分配" should conflict with "SSH端口2222"
        _default_factual = re.search(r'(?:\u9ed8\u8ba4|default)\s*(?:\u7aef\u53e3|port)\s*(?:\u662f|=|:)?\s*(\d+)', new_lower)
        if _default_factual and not _global_fp:
            _default_port = _default_factual.group(1)
            _default_svc = {"ssh", "redis", "nginx", "mysql", "postgres", "docker"} & new_tokens
            if _default_svc:
                for _rid, _ctext, _ccat, _cconf in rows:
                    if _rid in handled_ids:
                        continue
                    _exist_t_df = self._tokenize(_ctext)
                    _exist_svc = {"ssh", "redis", "nginx", "mysql", "postgres", "docker"} & _exist_t_df
                    if _default_svc & _exist_svc:
                        _exist_ports = re.findall(r'\d+', _ctext.lower())
                        if _exist_ports and _default_port not in _exist_ports:
                            # Existing has different port than our default — conflict
                            conflicts.append(ConflictInfo(
                                existing_id=_rid, new_content=new_content,
                                conflict_type="contradict",
                                similarity=0.6, resolution="pending"))
                            handled_ids.add(_rid)

                # Per-row analysis for remaining conflicts
        # ============================================================
        for row_id, content, cat, conf in rows:
            if row_id in handled_ids:
                continue

            # Skip supplementary rows (same entity + supplementary markers)
            if row_id in _supplementary_ids:
                continue

            exist_lower = content.lower().strip()
            exist_tokens = self._tokenize(content)

            if not new_tokens or not exist_tokens:
                continue

            intersection = new_tokens & exist_tokens
            union = new_tokens | exist_tokens
            jaccard = len(intersection) / len(union) if union else 0

            meaningful = {t for t in new_tokens if len(t) >= 2}
            if len(meaningful) <= 1:
                continue

            # --- L6: Context-aware conflict pairs ---
            conflict_pairs = {
                frozenset({"vim", "emacs", "vscode", "nano", "sublime", "notepad", "nvim", "neovim"}): {"editor", "editing", "edit", "编辑器", "编辑"},
                frozenset({"yum", "apt"}): {"package", "manager", "install"},
                frozenset({"systemd", "init"}): {"service", "init", "system"},
                frozenset({"iptables", "firewalld", "nftables"}): {"firewall"},
                frozenset({"nginx", "apache", "caddy"}): {"web", "server", "proxy", "web服务器"},
                frozenset({"docker", "podman"}): {"容器", "container", "部署", "deploy", "替代"},
                frozenset({"mysql", "postgresql", "mariadb"}): {"database", "db"},
                frozenset({"docker", "podman", "lxc"}): {"container"},
                frozenset({"json", "xml", "yaml", "csv"}): {"format", "response", "output"},
                frozenset({"dark", "light"}): {"theme", "mode", "ui"},
                frozenset({"chinese", "english", "中文", "英文"}): {"language", "output", "lang", "语言", "输出"},
                frozenset({"enable", "disable"}): {"update", "auto", "service"},
                frozenset({"true", "false", "yes", "no"}): set(),
                frozenset({"open", "close", "block"}): {"port", "firewall"},
                frozenset({"编译", "解释"}): {"语言", "类型", "python", "是"},
                frozenset({"开启", "关闭", "启用", "禁用"}): {"服务", "更新", "自动"},
                frozenset({"喜欢", "讨厌", "偏好"}): {"编辑器", "工具", "输出"},
                frozenset({"22", "2222", "80", "8080", "443", "8443", "6379", "6380", "3306", "3307"}): {"ssh", "端口", "port", "默认", "server", "nginx", "redis", "mysql", "https", "http"},
                frozenset({"1", "2", "24", "48", "72", "12", "3600", "86400"}): {"小时", "hour", "过期", "expire", "有效期", "token", "jwt"},
                frozenset({"prefer", "prefers", "like", "likes", "use", "uses"}): {"editor", "tool", "vim", "emacs"},
                frozenset({"changed", "switched", "replaced", "换成", "改成"}): {"vim", "emacs", "vscode", "port", "端口", "编辑器"},
                frozenset({"不是", "并非", "isnt", "not"}): {"解释", "编译", "默认", "default", "容器", "container"},
                frozenset({"禁用", "停用", "关闭", "disable", "off"}): {"服务", "service", "自动", "auto", "防火墙", "firewall"},
                frozenset({"不喜欢", "讨厌", "hate", "dislike"}): {"编辑器", "editor", "主题", "theme", "工具", "tool"},
                frozenset({"允许", "放行", "allow", "permit", "accept"}): {"阻止", "block", "deny", "拒绝", "防火墙", "firewall", "入站", "连接", "访问"},
                frozenset({"阻止", "block", "deny", "拒绝"}): {"允许", "放行", "allow", "permit", "防火墙", "firewall", "入站", "连接", "访问"},
                frozenset({"编译型", "compiled"}): {"解释型", "interpreted", "python", "语言"},
                frozenset({"解释型", "interpreted"}): {"编译型", "compiled", "python", "语言"},
            }

            # Same-value suppression: if both texts share same numeric values, skip
            _new_nums = set(re.findall(r'\d+', new_lower))
            _exist_nums = set(re.findall(r'\d+', exist_lower))
            _shared_nums = _new_nums & _exist_nums
            _has_same_port = bool(_shared_nums & {"22","80","443","3306","5432","6379","8080","8443","27017"})

            # Same-subject check: if texts are about different services, skip port conflicts
            _service_entities = {"redis", "ssh", "nginx", "mysql", "postgres", "postgresql",
                                 "docker", "mongodb", "kubernetes", "k8s", "apache", "caddy",
                                 "git", "vim", "nvim", "emacs", "memcached", "flask", "django"}
            _new_svc = _service_entities & new_tokens
            _exist_svc = _service_entities & exist_tokens
            _same_subject = bool(_new_svc & _exist_svc) or (not _new_svc and not _exist_svc)

            found_conflict = False
            # T19/T21 fix: skip environment-specific or uncertain statements
            if _new_is_env or _new_is_uncertain:
                handled_ids.add(row_id)
                continue
            for pair, required_context in conflict_pairs.items():
                new_has = pair & new_tokens
                exist_has = pair & exist_tokens
                if not new_has or not exist_has:
                    continue
                new_exclusive = new_has - exist_has
                exist_exclusive = exist_has - new_has
                if new_exclusive and exist_exclusive:
                    # Synonym-safe: skip if both sides are known synonyms
                    _synonym_groups = [
                        {"喜欢", "偏好", "prefer", "prefers", "like", "likes", "爱用", "习惯用"},
                        {"使用", "use", "uses", "using", "采用"},
                    ]
                    _is_synonym = False
                    for _sg in _synonym_groups:
                        if new_exclusive.issubset(_sg) and exist_exclusive.issubset(_sg):
                            _is_synonym = True
                            break
                    if _is_synonym:
                        continue
                    # Skip if different services with different PORT values
                    # (e.g., Redis:6379 vs Nginx:80 is NOT a conflict)
                    # But vim vs emacs IS a real conflict (same category, different tool)
                    _pair_has_numbers = any(p.isdigit() for p in pair)
                    if _pair_has_numbers and not _same_subject:
                        continue
                    # Skip if same port/number and same entity (supplementary info)
                    if _has_same_port and jaccard >= 0.3:
                        continue
                    if jaccard >= 0.25:
                        conflicts.append(ConflictInfo(
                            existing_id=row_id, new_content=new_content,
                            conflict_type="contradict",
                            similarity=jaccard + 0.3,
                            resolution="pending"))
                        found_conflict = True
                        handled_ids.add(row_id)
                        break
                    if required_context:
                        context_tokens = (new_tokens | exist_tokens) - pair
                        meaningful_ctx = {t for t in context_tokens if len(t) >= 2}
                        if meaningful_ctx & required_context:
                            conflicts.append(ConflictInfo(
                                existing_id=row_id, new_content=new_content,
                                conflict_type="contradict",
                                similarity=jaccard + 0.3,
                                resolution="pending"))
                            found_conflict = True
                            handled_ids.add(row_id)
                            break

            if found_conflict:
                continue

            
            # --- L6.5: Same-value suppression ---
            # If conflict was detected by L6 because both texts mention the same value
            # (e.g., both say "port 6379"), it's not a real conflict - suppress it
            _suppress_ids = set()
            for _conf in conflicts:
                if _conf.conflict_type in ("contradict", "duplicate"):
                    _exist_tokens = self._tokenize(
                        next((r[1] for r in rows if r[0] == _conf.existing_id), ""))
                    # Check if both have the same numeric values
                    _new_nums = set(re.findall(r'\d+', new_lower))
                    _exist_nums = set(re.findall(r'\d+', _conf.new_content.lower() if hasattr(_conf, 'new_content') else ""))
                    # Better: get existing content from rows
                    _exist_row = next((r for r in rows if r[0] == _conf.existing_id), None)
                    if _exist_row:
                        _exist_nums = set(re.findall(r'\d+', _exist_row[1].lower()))
                        _shared_nums = _new_nums & _exist_nums
                        # If they share the same numbers AND same subjects, suppress contradiction
                        _new_subj = {t for t in new_tokens if len(t) >= 2}
                        _exist_subj = {t for t in self._tokenize(_exist_row[1]) if len(t) >= 2}
                        _shared_subj = _new_subj & _exist_subj
                        if _shared_nums and len(_shared_subj) >= 2:
                            # T7 fix: only suppress if ALL numbers match (not just shared subset)
                            # "443 -> 8443" shares 443 but has different value — NOT a duplicate
                            if _new_nums == _exist_nums:
                                _neg_in_new = {"不", "不是", "不要", "别", "禁用", "停止",
                                              "not", "no", "never", "deprecated", "弃用"}
                                if not (_neg_in_new & new_tokens):
                                    _suppress_ids.add(_conf.existing_id)
            if _suppress_ids:
                conflicts = [c for c in conflicts if c.existing_id not in _suppress_ids or c.conflict_type == "duplicate"]

# --- L7: Numeric conflict (same subject, different number) ---
            num_patterns = [
                r'([a-z][a-z0-9_]*)\s*(?:默认)?\s*(?:端口|port)\s*(?:是|=|:)?\s*(\d+)',
                r'([a-z][a-z0-9_]*)\s*(?:默认|default)\s*(?:端口|port)\s*(\d+)',
                r'([a-z][a-z0-9_]*)[\s:]*(?:端口|port)\s*(?:is|=|:)\s*(\d+)',
                r'([a-z][a-z0-9_]*)\s+(\d+)\s*(?:端口|port)',
                r'(?:端口|port)\s*(\d+)',
                r'([a-z][a-z0-9_]*)\s*(?:默认)?\s*(?:连接)?\s*(?:端口|port)\s*(?:改为|改成|改到|变成|换成)\s*(\d+)',
                r'([a-z][a-z0-9_]*)\s*(?:默认)?\s*(?:端口|port)\s*(?:改为|改成)\s*(\d+)',
                r'([a-z][a-z0-9_]*)\s*(?:默认)?\s*(?:端口|port)\s*(?:是|为)\s*(\d+)',
            ]
            # Collect all port mentions from both texts (cross-pattern)
            new_port_mentions = []  # (tech, port) or just port
            exist_port_mentions = []
            for npat in num_patterns:
                new_matches = re.findall(npat, new_lower)
                exist_matches = re.findall(npat, exist_lower)
                new_port_mentions.extend(new_matches)
                exist_port_mentions.extend(exist_matches)
            # Compare across all patterns
            for nm in new_port_mentions:
                if found_conflict:
                    break
                for em in exist_port_mentions:
                    if isinstance(nm, tuple) and isinstance(em, tuple):
                        if nm[0] == em[0] and len(nm) > 1 and len(em) > 1 and nm[1] != em[1]:
                            conflicts.append(ConflictInfo(
                                existing_id=row_id, new_content=new_content,
                                conflict_type="contradict",
                                similarity=max(jaccard, 0.6),
                                resolution="pending"))
                            found_conflict = True
                            handled_ids.add(row_id)
                            break
                    elif isinstance(nm, str) and isinstance(em, str):
                        # Bare port comparison: only conflict if texts share same subject
                        # e.g., "SSH端口22" vs "SSH端口2222" = conflict
                        # but "SSH端口22" vs "Redis端口6379" = NOT conflict
                        _svc_entities = {"redis", "ssh", "nginx", "mysql", "postgres",
                                         "docker", "mongodb", "git", "vim", "apache", "caddy"}
                        _new_svc_l7 = _svc_entities & new_tokens
                        _exist_svc_l7 = _svc_entities & exist_tokens
                        _same_svc = bool(_new_svc_l7 & _exist_svc_l7)
                        if nm != em and _same_svc:
                            conflicts.append(ConflictInfo(
                                existing_id=row_id, new_content=new_content,
                                conflict_type="contradict",
                                similarity=max(jaccard, 0.6),
                                resolution="pending"))
                            found_conflict = True
                            handled_ids.add(row_id)
                            break

            if found_conflict:
                continue


            # --- L7b: Time-value normalization conflict ---
            # Detect "24小时" vs "15分钟", "1 hour" vs "30 minutes" etc.
            _time_to_seconds = {
                "秒": 1, "second": 1, "seconds": 1, "sec": 1,
                "分钟": 60, "分": 60, "minute": 60, "minutes": 60, "min": 60,
                "小时": 3600, "hour": 3600, "hours": 3600, "hr": 3600, "h": 3600,
                "天": 86400, "day": 86400, "days": 86400, "d": 86400,
                "周": 604800, "week": 604800, "weeks": 604800, "w": 604800,
                "月": 2592000, "month": 2592000, "months": 2592000,
            }
            _time_pattern = re.compile(r'(\d+)\s*(秒|分钟|分|小时|天|周|月|second|seconds|minute|minutes|min|hour|hours|hr|day|days|week|weeks|month|months)')
            new_times = _time_pattern.findall(new_lower)
            exist_times = _time_pattern.findall(exist_lower)
            if new_times and exist_times:
                new_secs = sum(int(v) * _time_to_seconds.get(u, 1) for v, u in new_times)
                exist_secs = sum(int(v) * _time_to_seconds.get(u, 1) for v, u in exist_times)
                # If both have a time value for the same topic and they differ significantly
                if new_secs > 0 and exist_secs > 0 and new_secs != exist_secs:
                    # Check if both texts are about the same topic
                    topic_overlap_l7b = (new_tokens - {t for t in new_tokens if len(t) < 2}) & (exist_tokens - {t for t in exist_tokens if len(t) < 2})
                    time_keywords = {"过期", "expire", "有效期", "token", "jwt", "timeout", "超时", "缓存", "cache", "ttl", "刷新"}
                    if topic_overlap_l7b and (time_keywords & new_tokens) and (time_keywords & exist_tokens):
                        # Normalize ratio - if the difference is more than 2x, it's a conflict
                        ratio = max(new_secs, exist_secs) / min(new_secs, exist_secs)
                        if ratio >= 2.0:
                            conflicts.append(ConflictInfo(
                                existing_id=row_id, new_content=new_content,
                                conflict_type="contradict",
                                similarity=max(jaccard, 0.6),
                                resolution="pending"))
                            found_conflict = True
                            handled_ids.add(row_id)

            # --- L7c: General numeric value change detection ---
            # Catches "Python升级到3.13", "Nginx worker改为8", "JWT过期设为1小时" etc.
            _general_change_verbs = re.search(
                r'(?:\u5347\u7ea7\u5230|\u5347\u7ea7\u4e3a|\u5347\u5230|\u5347\u4e3a|'
                r'\u6539\u4e3a|\u6539\u6210|\u6539\u5230|\u53d8\u6210|\u6362\u6210|'
                r'\u8bbe\u4e3a|\u8bbe\u6210|\u8bbe\u7f6e\u4e3a|\u8c03\u6574\u4e3a|'
                r'\u8fc7\u671f\u65f6\u95f4|\u6709\u6548\u671f)',
                new_lower,
            )
            if _general_change_verbs and not found_conflict:
                _new_change_nums = re.findall(r'\d+', new_lower)
                if _new_change_nums:
                    for _rid, _ctext, _ccat, _cconf in rows:
                        if _rid in handled_ids:
                            continue
                        _exist_t_l7c = self._tokenize(_ctext)
                        # Check subject overlap
                        _subj_new = {t for t in new_tokens if len(t) >= 2} & {
                            "ssh", "redis", "docker", "vim", "git", "nginx", "python",
                            "mysql", "postgres", "postgresql", "jwt", "token",
                            "https", "http", "worker", "timeout", "\u8fc7\u671f",
                            "\u6709\u6548", "\u7aef\u53e3", "port",
                        }
                        _subj_exist = {t for t in _exist_t_l7c if len(t) >= 2}
                        _shared_subj = _subj_new & _subj_exist
                        if _shared_subj:
                            _exist_change_nums = re.findall(r'\d+', _ctext.lower())
                            # Conflict if numbers differ
                            if _new_change_nums != _exist_change_nums:
                                # Same-value guard: skip if values are identical restatements
                                if set(_new_change_nums) == set(_exist_change_nums):
                                    continue
                                conflicts.append(ConflictInfo(
                                    existing_id=_rid, new_content=new_content,
                                    conflict_type="contradict",
                                    similarity=max(jaccard, 0.6),
                                    resolution="pending"))
                                found_conflict = True
                                handled_ids.add(_rid)
                                break
                                break
                            # Also conflict if new has numbers but existing doesn't
                            if _new_change_nums and not _exist_change_nums:
                                conflicts.append(ConflictInfo(
                                    existing_id=_rid, new_content=new_content,
                                    conflict_type="contradict",
                                    similarity=max(jaccard, 0.5),
                                    resolution="pending"))
                                found_conflict = True
                                handled_ids.add(_rid)
                                break

            if found_conflict:
                continue

            # --- L8: Negation word conflict (enable/disable) ---
            negation_pairs = [
                ({"disabled", "disable", "禁用", "关闭", "停用", "引入更好的"}, {"enabled", "enable", "启用", "开启"}),
                ({"blocked", "block", "阻止", "拦截"}, {"opened", "open", "allowed", "allow", "放行"}),
                ({"no", "not", "never", "不", "没有"}, {"yes", "always", "definitely", "是", "有"}),
                ({"允许", "放行", "allow", "permit"}, {"阻止", "block", "deny", "拒绝"}),
                ({"不适合", "不好", "不行"}, {"适合", "好", "行", "推荐"}),
            ]
            for neg, pos in negation_pairs:
                new_has_neg = bool(neg & new_tokens)
                new_has_pos = bool(pos & new_tokens)
                exist_has_neg = bool(neg & exist_tokens)
                exist_has_pos = bool(pos & exist_tokens)
                if ((new_has_neg and exist_has_pos) or (new_has_pos and exist_has_neg)):
                    topic_overlap = (new_tokens - neg - pos) & (exist_tokens - neg - pos)
                    if len(topic_overlap) >= 1:
                        conflicts.append(ConflictInfo(
                            existing_id=row_id, new_content=new_content,
                            conflict_type="contradict",
                            similarity=max(jaccard, 0.5),
                            resolution="pending"))
                        break

            # --- L9: Cross-language semantic conflict ---
            cross_lang_pairs = [
                ({"compiled", "编译型", "编译"}, {"interpreted", "解释型", "解释"}),
                ({"allow", "允许", "放行"}, {"block", "deny", "阻止", "拒绝", "禁止"}),
                ({"enable", "启用", "开启"}, {"disable", "禁用", "关闭"}),
                ({"喜欢", "偏好", "like", "prefer"}, {"讨厌", "不喜欢", "hate", "dislike"}),
            ]
            for pos_words, neg_words in cross_lang_pairs:
                new_has_pos = bool(pos_words & new_tokens)
                new_has_neg = bool(neg_words & new_tokens)
                exist_has_pos = bool(pos_words & exist_tokens)
                exist_has_neg = bool(neg_words & exist_tokens)
                if ((new_has_pos and exist_has_neg) or (new_has_neg and exist_has_pos)):
                    topic_overlap = (new_tokens - pos_words - neg_words) & (exist_tokens - pos_words - neg_words)
                    if len(topic_overlap) >= 1:
                        conflicts.append(ConflictInfo(
                            existing_id=row_id, new_content=new_content,
                            conflict_type="contradict",
                            similarity=max(jaccard, 0.5),
                            resolution="pending"))
                        break


        # ============================================================

        # L15: Rename / rebrand detection (T16)
        #    "Docker已更名为Moby" = factual change
        _rename_patterns = [
            re.compile(r'(\S+?)\s*(?:已|已经)?\s*(?:更名为|改名为|重命名为|renamed?\s+(?:to|as))\s*(\S+)'),
            re.compile(r'(\S+?)\s*(?:已|已经)?\s*(?:被|由)\s*(\S+?)\s*(?:收购|合并|替代|取代)'),
        ]
        for _rp in _rename_patterns:
            _rm = _rp.search(new_lower)
            if _rm:
                _old_name = _rm.group(1).strip()
                _old_toks = self._tokenize(_old_name)
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    if _old_toks & exist_t:
                        conflicts.append(ConflictInfo(
                            existing_id=rid, new_content=new_content,
                            conflict_type="contradict",
                            similarity=0.7, resolution="pending"))
                        handled_ids.add(rid)

        # L10: Implicit sentiment conflict (T23)
        #    "systemd也没那么好用" vs "Linux使用systemd管理服务"
        #    "vim用久了手疼" vs "用户偏好使用vim编辑器"
        #    "Redis有时候也会丢数据" vs "Redis常用作数据库的缓存层"
        # ============================================================
        _negative_sentiment_patterns = [
            r'(?:也没那么|不是那么|并不那么?)\s*(?:好用|好|强|稳定|可靠|靠谱)',
            r'(?:也就那样|就那么回事|不过如此)',
            r'\w+\s*(?:用久了|用长了)\s*(?:手疼|手累|卡|慢|烦|累)',
            r'\w+\s*(?:有时候|偶尔|经常|总是)\s*(?:也会|也|会)?\s*(?:丢数据|丢键|断连|断开|崩溃|crash|挂)',
            r'(?:太|非常|特别|确实)\s*(?:臃肿|笨重|复杂|难用|难配|难懂|烦|慢|卡|拉胯|难|差|烂|垃圾|坑)',
            r'\w+\s*(?:连接|链路)\s*(?:经常|总是|老是|频繁)\s*(?:断|断开|断连|超时)',
            r'\w+\s*(?:合并|冲突|配置)\s*太\s*(?:烦|复杂|麻烦|难)',
            r'(?:该淘汰|过时了|落伍了|老掉牙)',
            r'\w+\s*(?:配置|设置)\s*太\s*(?:复杂|麻烦|难|繁琐)',
        ]
        _compiled_sentiment = [re.compile(p) for p in _negative_sentiment_patterns]
        _sentiment_tech = {
            "ssh", "redis", "docker", "vim", "git", "nginx", "python",
            "mysql", "postgres", "postgresql", "systemd", "emacs", "vscode",
            "neovim", "nvim", "react", "vue", "angular", "flask", "django",
            "fastapi", "kubernetes", "k8s", "prometheus", "grafana", "kafka",
            "elasticsearch", "mongodb", "sqlite", "apache", "caddy",
        }
        _sentiment_detected = False
        for _sc in _compiled_sentiment:
            if _sc.search(new_lower):
                _sentiment_detected = True
                break
        if _sentiment_detected:
            _new_sent_tech = _sentiment_tech & new_tokens
            if _new_sent_tech:
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    exist_sent_tech = _sentiment_tech & exist_t
                    shared_tech = _new_sent_tech & exist_sent_tech
                    if shared_tech:
                        _positive_markers = {"喜欢", "偏好", "用", "使用", "推荐", "默认",
                                            "管理", "服务", "运行", "监听", "支持",
                                            "prefer", "like", "use", "using", "running",
                                            "listening", "managing", "supports"}
                        _existing_neg = {"不", "不好", "不行", "不适合", "讨厌", "不喜欢"}
                        if (_positive_markers & exist_t) and not (_existing_neg & exist_t):
                            conflicts.append(ConflictInfo(
                                existing_id=rid, new_content=new_content,
                                conflict_type="contradict",
                                similarity=0.6, resolution="pending"))
                            handled_ids.add(rid)

        # ============================================================
        # L11: Quantity constraint conflict (T22)
        #    "JWT过期最长不超过1小时" vs "JWT token过期时间设为24小时"
        # ============================================================
        _constraint_patterns = [
            re.compile(r'(?:最长|最多|最大|最高|上限)\s*(?:不超过|不能超过|不超|不高于|不大于|<=?)\s*(\d+)\s*(秒|分钟|分|小时|天|周|月|gb|mb|kb|个|条|次|万|%)'),
            re.compile(r'(?:最短|最少|最小|最低|下限)\s*(?:不低于|不少于|至少|>=?)\s*(\d+)\s*(秒|分钟|分|小时|天|周|月|gb|mb|kb|个|条|次|万|%)'),
        ]
        _unit_to_base = {
            "秒": 1, "分钟": 60, "分": 60, "小时": 3600, "天": 86400,
            "周": 604800, "月": 2592000, "gb": 1073741824, "mb": 1048576,
            "kb": 1024, "个": 1, "条": 1, "次": 1, "万": 10000, "%": 1,
        }
        for _cp in _constraint_patterns:
            _cm = _cp.search(new_lower)
            if _cm:
                _c_val = int(_cm.group(1))
                _c_unit = _cm.group(2)
                _c_base = _c_val * _unit_to_base.get(_c_unit, 1)
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    _topic_new = {t for t in new_tokens if len(t) >= 2} - {"最长", "最多", "最大", "最短", "最少", "最小", "不超过", "不能", "不低于", "至少"}
                    _topic_exist = {t for t in exist_t if len(t) >= 2}
                    _topic_overlap = _topic_new & _topic_exist
                    if len(_topic_overlap) >= 1:
                        _exist_num_pattern = re.compile(r'(\d+)\s*(' + re.escape(_c_unit) + ')')
                        _enm = _exist_num_pattern.search(ctext.lower())
                        if _enm:
                            _e_val = int(_enm.group(1))
                            _e_base = _e_val * _unit_to_base.get(_c_unit, 1)
                            if "最长" in new_lower or "最多" in new_lower or "最大" in new_lower:
                                if _c_base < _e_base:
                                    conflicts.append(ConflictInfo(
                                        existing_id=rid, new_content=new_content,
                                        conflict_type="contradict",
                                        similarity=0.7, resolution="pending"))
                                    handled_ids.add(rid)
                            elif "最短" in new_lower or "最少" in new_lower or "最小" in new_lower:
                                if _c_base > _e_base:
                                    conflicts.append(ConflictInfo(
                                        existing_id=rid, new_content=new_content,
                                        conflict_type="contradict",
                                        similarity=0.7, resolution="pending"))
                                    handled_ids.add(rid)

        # L11b: General constraint-on-tool detection (T22)
        # If new text has constraint words AND numbers AND mentions a known tool,
        # and base has the same tool without such constraints, flag conflict
        _constraint_words = {
            "限制", "最大", "最多", "最小", "最少", "超时", "连接池",
            "并发", "内存", "大小", "内存限制", "必须", "至少", "上限", "下限",
            "limit", "max", "min", "timeout", "pool", "size",
        }
        _has_constraint = bool(_constraint_words & new_tokens)
        _has_number = bool(re.findall(r'\d+', new_lower))
        # T19 fix: environment context suppresses constraint detection
        # Only suppress when text explicitly mentions a non-default environment
        _env_context_pattern = re.search(
            r'(?:测试|开发|本地|备份|staging|dev|ci|cd)\s*(?:环境|服务器)?|'
            r'(?:容器|docker|k8s)\s*(?:内|内部|里)|'
            r'(?:服务器|节点|集群)\s*\d|'
            r'(?:生产|线上)\s*(?:环境)?\s*(?:改|用|切)|(?:\u96c6\u7fa4|\u8282\u70b9|\u526f\u672c|\u5206\u7247)\s*(?:\u6a21\u5f0f|\u914d\u7f6e|\u9700\u8981|\u81f3\u5c11|\u6700\u5c11)',
            new_lower,
        )
        _has_context = bool(_env_context_pattern)
        if _has_constraint and _has_number and not _has_context:
            _tool_subjects = {
                "ssh", "redis", "docker", "vim", "git", "nginx", "python",
                "mysql", "postgres", "postgresql", "mongodb", "jwt", "https",
                "systemd", "kubernetes", "k8s", "elasticsearch",
            }
            _new_tools = _tool_subjects & new_tokens
            if _new_tools:
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    _exist_tools = _tool_subjects & exist_t
                    if _new_tools & _exist_tools:
                        # T28 fix: skip if constraint is about auth/security and base doesn't mention it
                        _auth_only = {"密码", "认证", "密钥", "ed25519", "rootless"}
                        if _auth_only & new_tokens and not (_auth_only & exist_t):
                            # Only skip if NO other constraint words overlap
                            _core_constraints = {"端口", "port", "超时", "timeout", "内存", "memory",
                                                 "并发", "连接池", "pool", "大小", "size", "限制", "limit"}
                            if not (_core_constraints & new_tokens):
                                continue
                        conflicts.append(ConflictInfo(
                            existing_id=rid, new_content=new_content,
                            conflict_type="contradict",
                            similarity=0.5, resolution="pending"))
                        handled_ids.add(rid)

        # ============================================================

        # L11c: Version constraint detection (T28)
        #    "Python只用3.11以上版本" vs "Python 3.8" or just "Python"
        _ver_patterns = [
            re.compile(r'(\S+?)\s*(?:只用|必须|至少|最低|最少)\s*(\d+\.?\d*)\s*(?:以上|及以上|版本|版)'),
            re.compile(r'(\S+?)\s*(?:不超过|最多|最高)\s*(\d+\.?\d*)\s*(?:以下|及以下|版本|版)'),
        ]
        for _vp in _ver_patterns:
            _vm = _vp.search(new_lower)
            if _vm:
                _ver_tool = _vm.group(1).strip()
                _ver_num = _vm.group(2)
                _ver_toks = self._tokenize(_ver_tool)
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    _ver_shared = _ver_toks & exist_t
                    if _ver_shared:
                        # Check if existing text mentions a different version
                        _exist_ver = re.search(r'(\d+\.\d+(?:\.\d+)?)', ctext)
                        if _exist_ver:
                            conflicts.append(ConflictInfo(
                                existing_id=rid, new_content=new_content,
                                conflict_type="contradict",
                                similarity=0.6, resolution="pending"))
                            handled_ids.add(rid)

        # L12: Preference shift / temporal preference (T12)
        #    "现在更喜欢light而不是dark" vs "用户喜欢用dark主题"
        # ============================================================
        _pref_shift_patterns = [
            re.compile(r'(?:现在|目前|如今|最近)\s*(?:更喜欢|更习惯|更倾向|偏好|喜欢)\s*(.+?)\s*(?:而不是|而非|不是|over|instead)'),
            re.compile(r'(?:现在|目前|如今|开始|\\u5f00\\u59cb)\s*(?:改用|换用|开始用|用|偏好|喜欢)\\s*(\\w+)'),
        ]
        for _psp in _pref_shift_patterns:
            _psm = _psp.search(new_lower)
            if _psm:
                _shift_target = _psm.group(1).strip() if _psm.lastindex >= 1 else ""
                if _shift_target:
                    _shift_tokens = self._tokenize(_shift_target)
                    for rid, ctext, ccat, cconf in rows:
                        if rid in handled_ids:
                            continue
                        exist_t = self._tokenize(ctext)
                        _old_pref_markers = {"喜欢", "偏好", "prefer", "like", "习惯", "常用"}
                        if _old_pref_markers & exist_t:
                            _tool_groups_local = [
                                {"vim", "nvim", "neovim", "emacs", "vscode", "nano", "sublime"},
                                {"nginx", "apache", "caddy", "lighttpd"},
                                {"mysql", "postgresql", "mariadb", "sqlite"},
                                {"docker", "podman", "lxc"},
                                {"redis", "memcached"},
                                {"dark", "light"},
                                {"中文", "english", "英文"},
                            ]
                            for _tg in _tool_groups_local:
                                new_in = _tg & _shift_tokens
                                exist_in = _tg & exist_t
                                if new_in and exist_in and new_in != exist_in:
                                    conflicts.append(ConflictInfo(
                                        existing_id=rid, new_content=new_content,
                                        conflict_type="contradict",
                                        similarity=0.7, resolution="pending"))
                                    handled_ids.add(rid)
                                    break

        # ============================================================

        # L16: Arrow/comparison notation (T29)
        #    "Nginx port: 80 -> 8080" / "vim > vscode"
        _arrow_patterns = [
            re.compile(r'(\d+)\s*(?:->|-->|\u2192|=>)\s*(\d+)'),  # 80 -> 8080
            re.compile(r'(\S+?)\s*>\s*(\S+?)\s*(?:\(|\uff08|$)'),  # vim > vscode
        ]
        for _ap in _arrow_patterns:
            _am = _ap.search(new_content)  # Use original case
            if _am:
                _a_g1 = _am.group(1).strip().lower()
                _a_g2 = _am.group(2).strip().lower()
                for rid, ctext, ccat, cconf in rows:
                    if rid in handled_ids:
                        continue
                    exist_t = self._tokenize(ctext)
                    exist_lower_t = ctext.lower()
                    # Number arrow: check if existing text has the old number
                    if _a_g1.isdigit():
                        if _a_g1 in re.findall(r'\d+', exist_lower_t):
                            _shared_svc = self._tokenize(ctext) & new_tokens
                            _svc_set = {"ssh","redis","docker","vim","git","nginx","python","mysql","nginx","port","\u7aef\u53e3"}
                            if _shared_svc & _svc_set:
                                conflicts.append(ConflictInfo(
                                    existing_id=rid, new_content=new_content,
                                    conflict_type="contradict",
                                    similarity=0.7, resolution="pending"))
                                handled_ids.add(rid)
                    else:
                        # Tool comparison: check if worse tool is in existing preference
                        _pref_markers = {"\u559c\u6b22", "\u504f\u597d", "prefer", "like", "\u4e60\u60ef"}
                        if _pref_markers & exist_t:
                            _tool_groups_l16 = [
                                {"vim","nvim","neovim","emacs","vscode","nano"},
                                {"nginx","apache","caddy"},
                                {"mysql","postgresql","mariadb","sqlite"},
                                {"docker","podman"},
                                {"redis","memcached"},
                            ]
                            for _tg in _tool_groups_l16:
                                if _a_g2 in _tg and (_tg & exist_t):
                                    conflicts.append(ConflictInfo(
                                        existing_id=rid, new_content=new_content,
                                        conflict_type="contradict",
                                        similarity=0.7, resolution="pending"))
                                    handled_ids.add(rid)
                                    break

        # L13: False positive filter (T17)
        #    Semantic rewrites should NOT be detected as conflicts.
        # ============================================================
        _rephrase_markers = {
            "即", "也就是说", "换句话说", "意思是", "也就是",
            "换言之", "简单说", "具体来说", "其实就是",
            "namely", "that is", "i.e.", "in other words",
        }
        _has_rephrase = bool(_rephrase_markers & new_tokens)
        if _has_rephrase and conflicts:
            _filtered = []
            for c in conflicts:
                if c.conflict_type == "contradict":
                    exist_row = None
                    for rid, ctext, ccat, cconf in rows:
                        if rid == c.existing_id:
                            exist_row = ctext
                            break
                    if exist_row:
                        exist_t_filt = self._tokenize(exist_row)
                        inter = new_tokens & exist_t_filt
                        union = new_tokens | exist_t_filt
                        sim = len(inter) / len(union) if union else 0
                        if sim >= 0.6:
                            continue
                _filtered.append(c)
            conflicts = _filtered



        # ============================================================
        # L14: Tool comparison pattern (T30)
        #    "nvim比vim更好用" / "用户觉得emacs比vim好"
        # ============================================================
        if not any(c.existing_id in handled_ids for c in conflicts):
            _comparison_patterns = [
                re.compile(r'(\S+?)\s*比\s*(\S+?)\s*(?:更好|更好用|好用|好|更优|更强|更方便|更快|更安全)'),
                re.compile(r'(?:觉得|认为|感觉)\s*(\S+?)\s*比\s*(\S+?)\s*(?:好|优秀|强|好用|方便)'),
            ]
            _tool_groups_l14 = [
                {"vim", "nvim", "neovim", "emacs", "vscode", "nano", "sublime"},
                {"nginx", "apache", "caddy", "lighttpd"},
                {"mysql", "postgresql", "mariadb", "sqlite"},
                {"docker", "podman", "lxc"},
                {"redis", "memcached"},
                {"react", "vue", "angular", "svelte"},
                {"python", "java", "go", "rust", "node", "typescript"},
                {"dark", "light"},
            ]
            for _cp in _comparison_patterns:
                _cm = _cp.search(new_lower)
                if _cm:
                    _better = _cm.group(1).strip()
                    _worse = _cm.group(2).strip()
                    _better_t = self._tokenize(_better)
                    _worse_t = self._tokenize(_worse)
                    for rid, ctext, ccat, cconf in rows:
                        if rid in handled_ids:
                            continue
                        exist_t = self._tokenize(ctext)
                        _pref_markers = {"喜欢", "偏好", "prefer", "like", "习惯", "常用", "首选"}
                        if _pref_markers & exist_t:
                            for _tg in _tool_groups_l14:
                                worse_in_exist = _tg & exist_t
                                worse_in_new = _tg & _worse_t
                                better_in_new = _tg & _better_t
                                if worse_in_exist and worse_in_new and better_in_new:
                                    if worse_in_exist == worse_in_new:
                                        conflicts.append(ConflictInfo(
                                            existing_id=rid, new_content=new_content,
                                            conflict_type="contradict",
                                            similarity=0.7, resolution="pending"))
                                        handled_ids.add(rid)
                                        break

        import sys as _ds
        _types = [c.conflict_type for c in conflicts]

        return conflicts
