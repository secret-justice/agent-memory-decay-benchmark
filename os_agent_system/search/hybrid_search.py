"""
混合检索引擎
=============
借鉴 EverOS 的 search manager 设计:

检索管线:
  1. BM25 关键词召回 (基于 rank_bm25)
  2. 向量 ANN 语义召回 (基于 ChromaDB)
  3. RRF (Reciprocal Rank Fusion) 融合排序
  4. (可选) Cross-Encoder Rerank 精排

与 EverOS 的差异:
- EverOS 使用 LanceDB (内置 BM25 + 向量 + 标量)
- 本项目使用 ChromaDB + 独立 BM25 层 (保持现有架构不变)

RRF 公式: score(d) = Σ 1/(k + rank_i(d))
  k = 60 (默认，与 EverOS 一致)
"""

import logging
import math
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class SearchResult:
    """检索结果"""
    id: str
    text: str
    score: float
    source: str            # "bm25" | "vector" | "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)
    bm25_rank: int = 0
    vector_rank: int = 0


@dataclass
class SearchConfig:
    """检索配置"""
    bm25_top_k: int = 20           # BM25 召回数量
    vector_top_k: int = 20         # 向量召回数量
    final_top_k: int = 5           # 最终返回数量
    rrf_k: int = 60                # RRF 常数 k
    enable_rerank: bool = False    # 是否启用 Rerank
    rerank_top_k: int = 10         # Rerank 取 top_k
    bm25_weight: float = 1.0       # BM25 权重
    vector_weight: float = 1.0     # 向量权重
    enable_closure: bool = False   # DAG closure retrieval
    closure_max_depth: int = 3     # max closure depth
    closure_max_nodes: int = 15    # max closure nodes
    recency_weight: float = 0.0     # time-aware boost weight (0=disabled)


# ============================================================
# BM25 实现
# ============================================================

class BM25Index:
    """
    轻量级 BM25 索引

    基于内存的 BM25 实现，适用于中小规模文档集
    k1=1.5, b=0.75 (经典参数，与 Lucene/Elasticsearch 默认一致)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: Dict[str, str] = {}           # doc_id → text
        self._doc_len: Dict[str, int] = {}         # doc_id → length
        self._avg_dl: float = 0.0                  # 平均文档长度
        self._df: Dict[str, int] = {}              # term → 文档频率
        self._tf: Dict[str, Dict[str, int]] = {}   # term → {doc_id: tf}
        self._N: int = 0                           # 文档总数

    def _tokenize(self, text: str) -> List[str]:
        """分词: 中英文混合分词"""
        # 简单分词: 英文按空格/标点，中文按字符 (bigram)
        text = text.lower()
        # 英文单词
        en_tokens = re.findall(r'[a-z0-9]+', text)
        # 中文 bigram
        cn_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        cn_tokens = []
        for seg in cn_chars:
            if len(seg) >= 2:
                cn_tokens.extend([seg[i:i+2] for i in range(len(seg)-1)])
            else:
                cn_tokens.append(seg)
        return en_tokens + cn_tokens

    def add(self, doc_id: str, text: str):
        """添加文档到索引"""
        if doc_id in self._docs:
            self.remove(doc_id)

        self._docs[doc_id] = text
        tokens = self._tokenize(text)
        self._doc_len[doc_id] = len(tokens)
        self._N = len(self._docs)

        # 更新 TF
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        self._tf[doc_id] = dict(tf)

        # 更新 DF
        for token in set(tokens):
            self._df[token] = self._df.get(token, 0) + 1

        # 更新平均文档长度
        self._avg_dl = sum(self._doc_len.values()) / self._N if self._N > 0 else 0

    def remove(self, doc_id: str):
        """从索引中移除文档"""
        if doc_id not in self._docs:
            return

        # 更新 DF
        if doc_id in self._tf:
            for token in self._tf[doc_id]:
                self._df[token] = max(0, self._df.get(token, 0) - 1)
                if self._df[token] == 0:
                    del self._df[token]

        del self._docs[doc_id]
        del self._doc_len[doc_id]
        if doc_id in self._tf:
            del self._tf[doc_id]

        self._N = len(self._docs)
        self._avg_dl = sum(self._doc_len.values()) / self._N if self._N > 0 else 0

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """BM25 检索，返回 [(doc_id, score), ...]"""
        if self._N == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in query_tokens:
            if token not in self._df:
                continue

            df = self._df[token]
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in self._tf.items():
                if token not in tf:
                    continue

                doc_len = self._doc_len[doc_id]
                tf_val = tf[token]

                # BM25 公式
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
                scores[doc_id] += idf * numerator / denominator

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @property
    def size(self) -> int:
        return self._N


# ============================================================
# RRF 融合排序
# ============================================================

def rrf_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: List[float] = None,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF)

    将多个排序列表融合为一个统一排序。

    公式: score(d) = Σ w_i / (k + rank_i(d))

    Args:
        rankings: 多个排序列表，每个是 [(doc_id, original_score), ...]
        k: RRF 常数 (默认 60，与 EverOS 一致)
        weights: 每个排序列表的权重

    Returns:
        融合后的排序列表 [(doc_id, rrf_score), ...]
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    rrf_scores = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf_scores[doc_id] += weight / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# 混合检索引擎
# ============================================================

class HybridSearchEngine:
    """
    混合检索引擎

    融合 BM25 关键词检索和向量语义检索，
    通过 RRF (Reciprocal Rank Fusion) 合并结果。

    架构:
      query
        │
        ├──→ BM25 Index ──→ sparse_candidates ──┐
        │                                        ├──→ RRF Fusion ──→ final_results
        └──→ Vector Store ──→ dense_candidates ──┘

    使用方式:
        engine = HybridSearchEngine(
            vector_store=chroma_store,
            embedding_fn=my_embedding,
        )
        engine.build_bm25_index(documents)  # 首次构建
        results = engine.search("查询文本", top_k=5)
    """

    def __init__(
        self,
        vector_store=None,
        embedding_fn: Callable = None,
        config: SearchConfig = None,
    ):
        self._vector_store = vector_store
        self._embedding_fn = embedding_fn
        self._config = config or SearchConfig()
        self._bm25 = BM25Index()
        self._doc_texts: Dict[str, str] = {}  # doc_id → text (用于 BM25 构建)
        self._dependency_map: Dict[str, List[str]] = {}  # DAG dependency edges
        self._forgetting_curve = None  # ForgettingCurveEngine (optional)

    def build_bm25_index(self, documents: List[Tuple[str, str, Dict]] = None):
        """
        构建 BM25 索引

        Args:
            documents: [(doc_id, text, metadata), ...]
        """
        if documents is None:
            # 从向量库全量重建
            if self._vector_store:
                try:
                    all_docs = self._vector_store.get_all()
                    if all_docs:
                        documents = [
                            (doc["id"], doc["text"], doc.get("metadata", {}))
                            for doc in all_docs
                        ]
                except Exception as e:
                    logger.warning(f"从向量库获取文档失败: {e}")
                    return

        if not documents:
            logger.warning("无文档可索引")
            return

        self._bm25 = BM25Index()
        for doc_id, text, _ in documents:
            self._bm25.add(doc_id, text)
            self._doc_texts[doc_id] = text

        logger.info(f"BM25 索引构建完成: {self._bm25.size} 篇文档")

    def add_to_index(self, doc_id: str, text: str):
        """增量添加文档到 BM25 索引"""
        self._bm25.add(doc_id, text)
        self._doc_texts[doc_id] = text

    def remove_from_index(self, doc_id: str):
        """从 BM25 索引中移除文档"""
        self._bm25.remove(doc_id)
        self._doc_texts.pop(doc_id, None)

    def search(
        self,
        query: str,
        top_k: int = None,
        where: Dict = None,
        method: str = "hybrid",
    ) -> List[SearchResult]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回数量 (默认使用 config.final_top_k)
            where: ChromaDB 过滤条件
            method: 检索方法 "hybrid" | "bm25" | "vector"

        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = self._config.final_top_k

        if method == "bm25":
            results = self._search_bm25(query, top_k)
        elif method == "vector":
            results = self._search_vector(query, top_k, where)
        else:
            results = self._search_hybrid(query, top_k, where)

        # DAG closure post-processing
        results = self._apply_closure(results)

        # Time-aware retrieval boost
        results = self._apply_time_boost(results)
        return results

    def _search_bm25(self, query: str, top_k: int) -> List[SearchResult]:
        """纯 BM25 检索"""
        raw = self._bm25.search(query, top_k=top_k)
        results = []
        for rank, (doc_id, score) in enumerate(raw, 1):
            results.append(SearchResult(
                id=doc_id,
                text=self._doc_texts.get(doc_id, ""),
                score=score,
                source="bm25",
                bm25_rank=rank,
            ))
        return results

    def _search_vector(self, query: str, top_k: int,
                       where: Dict = None) -> List[SearchResult]:
        """纯向量检索"""
        if not self._vector_store or not self._embedding_fn:
            logger.warning("向量检索需要配置 vector_store 和 embedding_fn")
            return []

        query_embedding = self._embedding_fn(query)
        raw = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        results = []
        for rank, item in enumerate(raw, 1):
            results.append(SearchResult(
                id=item.get("id", ""),
                text=item.get("text", ""),
                score=item.get("score", 0),
                source="vector",
                vector_rank=rank,
                metadata=item.get("metadata", {}),
            ))
        return results

    def _search_hybrid(self, query: str, top_k: int,
                       where: Dict = None) -> List[SearchResult]:
        """
        混合检索: BM25 + 向量 + RRF 融合

        管线:
        1. BM25 召回 sparse_candidates
        2. 向量 ANN 召回 dense_candidates
        3. RRF 融合排序
        4. 截取 top_k
        """
        # 1. BM25 召回
        sparse_raw = self._bm25.search(query, top_k=self._config.bm25_top_k)
        sparse_ranking = [(doc_id, score) for doc_id, score in sparse_raw]

        # 2. 向量召回
        dense_ranking = []
        if self._vector_store and self._embedding_fn:
            try:
                query_embedding = self._embedding_fn(query)
                dense_raw = self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=self._config.vector_top_k,
                    where=where,
                )
                dense_ranking = [
                    (item.get("id", ""), item.get("score", 0))
                    for item in dense_raw
                ]
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")

        # 3. 如果只有一个通道有结果，直接返回
        if not sparse_ranking:
            results = []
            for rank, (doc_id, score) in enumerate(dense_ranking[:top_k], 1):
                results.append(SearchResult(
                    id=doc_id, text=self._doc_texts.get(doc_id, ""),
                    score=score, source="vector", vector_rank=rank,
                ))
            return results

        if not dense_ranking:
            results = []
            for rank, (doc_id, score) in enumerate(sparse_ranking[:top_k], 1):
                results.append(SearchResult(
                    id=doc_id, text=self._doc_texts.get(doc_id, ""),
                    score=score, source="bm25", bm25_rank=rank,
                ))
            return results

        # 4. RRF 融合
        fused = rrf_fusion(
            [sparse_ranking, dense_ranking],
            k=self._config.rrf_k,
            weights=[self._config.bm25_weight, self._config.vector_weight],
        )

        # 5. 构建结果
        sparse_map = {doc_id: rank for rank, (doc_id, _) in enumerate(sparse_ranking, 1)}
        dense_map = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_ranking, 1)}

        results = []
        for doc_id, rrf_score in fused[:top_k]:
            results.append(SearchResult(
                id=doc_id,
                text=self._doc_texts.get(doc_id, ""),
                score=rrf_score,
                source="hybrid",
                bm25_rank=sparse_map.get(doc_id, 0),
                vector_rank=dense_map.get(doc_id, 0),
            ))

        return results

    def set_forgetting_curve(self, engine):
        """Set ForgettingCurveEngine for time-aware retrieval"""
        self._forgetting_curve = engine

    def set_dependency_map(self, dep_map: Dict[str, List[str]]):
        """Set DAG dependency map for closure retrieval"""
        self._dependency_map = dict(dep_map)

    def update_dependency(self, entry_id: str, depends_on: List[str]):
        """Incrementally update single dependency"""
        if depends_on:
            self._dependency_map[entry_id] = list(depends_on)

    def _apply_closure(self, results: "List[SearchResult]") -> "List[SearchResult]":
        """Apply DAG closure to search results"""
        if not self._config.enable_closure or not self._dependency_map:
            return results

        from .dag_closure import closure_retrieve

        seed_dicts = []
        for r in results:
            seed_dicts.append({
                "id": r.id,
                "text": r.text,
                "score": r.score,
                "source": r.source,
            })

        all_entries = {}
        for doc_id, text in self._doc_texts.items():
            entry = {"raw_body": text, "inline_fields": {}}
            if doc_id in self._dependency_map:
                entry["inline_fields"]["depends_on"] = ",".join(self._dependency_map[doc_id])
            all_entries[doc_id] = entry

        closure_results = closure_retrieve(
            seed_results=seed_dicts,
            all_entries=all_entries,
            max_depth=self._config.closure_max_depth,
            max_nodes=self._config.closure_max_nodes,
        )

        seen_ids = {r.id for r in results}
        merged = list(results)
        for cr in closure_results:
            cid = cr.get("id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(SearchResult(
                    id=cid,
                    text=cr.get("text", ""),
                    score=cr.get("score", 0.1),
                    source="closure_context",
                    metadata=cr.get("metadata", {}),
                ))

        logger.info(f"Closure retrieval: {len(results)} seeds -> {len(merged)} total")
        return merged

    def _apply_time_boost(self, results: "List[SearchResult]") -> "List[SearchResult]":
        """Apply time-aware boost using forgetting curve"""
        if not self._forgetting_curve or self._config.recency_weight <= 0:
            return results

        try:
            from memory.forgetting_curve import time_aware_boost
        except ImportError:
            return results

        boosted = []
        for r in results:
            strength = self._forgetting_curve.get_strength(r.id)
            if strength:
                retention = strength["retention"]
                r.score = time_aware_boost(r.score, retention, self._config.recency_weight)
            boosted.append(r)

        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        return {
            "bm25_docs": self._bm25.size,
            "has_vector_store": self._vector_store is not None,
            "has_embedding_fn": self._embedding_fn is not None,
            "dependency_edges": len(self._dependency_map),
            "closure_enabled": self._config.enable_closure,
            "config": {
                "bm25_top_k": self._config.bm25_top_k,
                "vector_top_k": self._config.vector_top_k,
                "final_top_k": self._config.final_top_k,
                "rrf_k": self._config.rrf_k,
                "closure_max_depth": self._config.closure_max_depth,
            },
        }
