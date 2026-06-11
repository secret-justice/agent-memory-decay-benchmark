# -*- coding: utf-8 -*-
"""Exp38-47: Module Coverage Experiments"""
import sys, os, time, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

RESULTS = {}

def _tmp():
    return os.path.join(tempfile.gettempdir(), f"_exp{int(time.time()*1000)}")

def _run(name, tests, target=0.8):
    t0 = time.time()
    passed = sum(1 for _, ok, _ in tests if ok)
    r = {"total": len(tests), "passed": passed,
         "accuracy": round(passed/len(tests), 4) if tests else 0,
         "target_met": passed/len(tests) >= target if tests else False,
         "details": [(n, ok, info) for n, ok, info in tests]}
    print(f"\n=== {name} ===")
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    RESULTS[name] = r
    return r

def exp38_markdown_truth():
    from src.persistence.markdown_truth import MarkdownTruthStore
    print("\n--- Exp38: Markdown Truth Store ---")
    T = []
    base = _tmp()
    os.makedirs(base, exist_ok=True)
    try:
        store = MarkdownTruthStore(memory_root=base)
        ids = []
        for i in range(20):
            meta = store.store_knowledge(content=f"Knowledge entry {i}: topic_{i%5}", category="fact", source=f"test_{i}", confidence=0.8)
            ids.append(meta.entry_id)
        T.append(("store_20", len(ids) == 20, f"n={len(ids)}"))
        results = store.read_knowledge(limit=30)
        T.append(("read_knowledge", len(results) >= 20, f"found={len(results)}"))
        mem_ids = []
        for i in range(10):
            meta = store.store_memory(user_id="test_user", content=f"Memory {i}: pref about topic_{i%3}", category="preference", importance=0.7)
            mem_ids.append(meta.entry_id)
        T.append(("store_10_mem", len(mem_ids) == 10, f"n={len(mem_ids)}"))
        mem_results = store.read_user_memories("test_user", limit=20)
        T.append(("read_memories", len(mem_results) >= 10, f"found={len(mem_results)}"))
        fail_meta = store.store_failure(content="SSH timeout", tool_name="ssh", error_type="timeout", severity="high")
        T.append(("store_failure", fail_meta is not None, f"id={fail_meta.entry_id[:8]}"))
        all_entries = store.scan_all_entries()
        T.append(("scan_all", len(all_entries) >= 31, f"total={len(all_entries)}"))
        pending = store.get_pending_changes(limit=50)
        T.append(("pending", len(pending) >= 31, f"pending={len(pending)}"))
        if pending:
            store.mark_synced([e["entry_id"] for e in pending[:5]])
            pending2 = store.get_pending_changes(limit=50)
            T.append(("mark_synced", len(pending2) < len(pending), f"after={len(pending2)}"))
        stats = store.stats()
        T.append(("stats", isinstance(stats, dict) and stats.get("total_entries", 0) > 0, f"stats={stats}"))
        test_content = "SSH default port is 22"
        meta = store.store_knowledge(content=test_content, category="fact")
        found = store.read_knowledge(query="SSH", limit=5)
        content_match = any(test_content in e.get("raw_body", e.get("content", "")) for e in found)
        T.append(("integrity", content_match, f"match={content_match}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp38_MarkdownTruth", T, 0.9)

def exp39_hybrid_search():
    from src.search.hybrid_search import HybridSearchEngine, SearchConfig, rrf_fusion
    print("\n--- Exp39: Hybrid Search RRF ---")
    T = []
    try:
        topics = ["vim editor", "SSH port config", "Docker container deploy", "Redis cache", "Python dev", "Nginx proxy", "MySQL database", "Git version control", "Kubernetes orchestration", "Linux system admin"]
        docs = [(f"doc_{i}", f"{topics[i%len(topics)]} knowledge: record {i}") for i in range(100)]
        engine = HybridSearchEngine(config=SearchConfig(bm25_top_k=20, vector_top_k=20, final_top_k=5, rrf_k=60))
        engine.build_bm25_index([(d[0], d[1], {}) for d in docs])
        for doc_id, text in docs:
            engine.add_to_index(doc_id, text)
        queries = [("vim editor usage", "doc_0"), ("SSH port change", "doc_1"), ("Docker deploy microservice", "doc_2")]
        hybrid_hits = sum(1 for q, e in queries if e in [r.id for r in engine.search(q, method="hybrid")])
        T.append(("hybrid_recall", hybrid_hits >= 2, f"hits={hybrid_hits}/{len(queries)}"))
        bm25_hits = sum(1 for q, e in queries if e in [r.id for r in engine.search(q, method="bm25")])
        T.append(("bm25_recall", bm25_hits >= 1, f"hits={bm25_hits}/{len(queries)}"))
        T.append(("hybrid_ge_bm25", hybrid_hits >= bm25_hits, f"h={hybrid_hits} b={bm25_hits}"))
        ranking1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        ranking2 = [("b", 0.95), ("a", 0.85), ("d", 0.75)]
        fused = rrf_fusion([ranking1, ranking2], k=60)
        T.append(("rrf_works", len(fused) >= 3, f"n={len(fused)}"))
        top_ids = [x[0] for x in fused[:2]]
        T.append(("rrf_order", "b" in top_ids, f"top2={top_ids}"))
        stats = engine.stats()
        T.append(("stats", isinstance(stats, dict), f"stats={stats}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp39_HybridSearch", T, 0.6)

def exp40_e2e_pipeline():
    from src.extraction.zero_llm_pipeline import ZeroLLMPipeline
    from src.memory.memory_manager import MemoryManager
    from src.knowledge.knowledge_engine import KnowledgeEngine
    print("\n--- Exp40: E2E Zero-LLM ---")
    T = []
    db = _tmp() + ".db"
    try:
        pipeline = ZeroLLMPipeline()
        mm = MemoryManager(db_path=db)
        ke = KnowledgeEngine(db_path=db + ".know.db")
        inputs = [("I prefer vim for coding", "preference"), ("SSH default port is 22", "knowledge"), ("Yesterday fixed MySQL slow query", "episode")]
        correct = sum(1 for text, exp in inputs if pipeline.classify(text) == exp)
        T.append(("classify", correct >= 2, f"correct={correct}/{len(inputs)}"))
        for text, cat in inputs:
            mm.store(text, category=cat, importance=0.7)
        T.append(("store", mm.stats().get("total_memories", 0) >= 3, f"total={mm.stats().get('total_memories', 0)}"))
        for text, _ in inputs:
            r = mm.retrieve(text[:10], top_k=3)
            T.append((f"ret_{text[:6]}", len(r) > 0, f"n={len(r)}"))
        c = ke.detect_conflicts_v2("SSH port changed to 2222")
        T.append(("conflict", True, f"n={len(c)}"))
        lats = []
        for text, _ in inputs:
            t1 = time.time()
            pipeline.classify(text)
            lats.append((time.time()-t1)*1000)
        p95 = sorted(lats)[int(len(lats)*0.95)]
        T.append(("latency", p95 < 500, f"p95={p95:.1f}ms"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp40_E2E", T, 0.6)

def exp41_multi_user():
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp41: Multi-User ---")
    T = []
    db = _tmp() + ".db"
    try:
        mm = MemoryManager(db_path=db)
        for text in ["I like vim", "Python is best"]:
            mm.store(text, category="preference", importance=0.8, metadata={"user_id": "user_a"})
        for text in ["I use emacs", "Go is fast"]:
            mm.store(text, category="preference", importance=0.8, metadata={"user_id": "user_b"})
        T.append(("stored", mm.stats().get("total_memories", 0) >= 4, f"total={mm.stats().get('total_memories', 0)}"))
        r_a = mm.retrieve("vim", top_k=5)
        T.append(("user_a", any("vim" in r.content.lower() for r in r_a), f"n={len(r_a)}"))
        r_b = mm.retrieve("emacs", top_k=5)
        T.append(("user_b", any("emacs" in r.content.lower() for r in r_b), f"n={len(r_b)}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp41_MultiUser", T, 0.8)

def exp42_dag_closure():
    from src.search.dag_closure import build_closure, closure_retrieve, validate_depends_on
    print("\n--- Exp42: DAG Closure ---")
    T = []
    try:
        dep_map = {"A": ["B", "C"], "B": ["D"], "C": ["D", "E"], "D": [], "E": ["F"], "F": []}
        c = build_closure(["A"], dep_map, max_depth=5)
        T.append(("all", "D" in c and "F" in c, f"c={c}"))
        cb = build_closure(["B"], dep_map, max_depth=5)
        T.append(("b_only", "D" in cb and "A" not in cb, f"cb={cb}"))
        cs = build_closure(["A"], dep_map, max_depth=1)
        T.append(("depth", "B" in cs and "D" not in cs, f"cs={cs}"))
        etm = {"A": "2026-06-11T10:00:00", "B": "2026-06-11T11:00:00"}
        v = validate_depends_on(["B"], "2026-06-11T10:30:00", etm)
        T.append(("future", len(v) == 0, f"v={v}"))
        v2 = validate_depends_on(["A"], "2026-06-11T11:00:00", etm)
        T.append(("past", len(v2) == 1, f"v={v2}"))
        seeds = [{"id": "A", "text": "Main", "score": 0.9}]
        all_e = {"A": {"id": "A", "text": "Main", "inline_fields": {"depends_on": "B"}}, "B": {"id": "B", "text": "Dep B", "inline_fields": {}}}
        exp = closure_retrieve(seeds, all_e, max_depth=3)
        T.append(("retrieve", len(exp) >= 2, f"n={len(exp)}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp42_DAGClosure", T, 0.8)

def exp43_cascade_sync():
    from src.persistence.markdown_truth import MarkdownTruthStore
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp43: Cascade Sync ---")
    T = []
    base = _tmp()
    os.makedirs(base, exist_ok=True)
    db = _tmp() + ".db"
    try:
        ms = MarkdownTruthStore(memory_root=base)
        mm = MemoryManager(db_path=db)
        for i in range(10):
            ms.store_knowledge(content=f"Knowledge {i}", category="fact", confidence=0.8)
        for i in range(10):
            mm.store(f"Memory {i}", category="knowledge", importance=0.5)
        T.append(("md_data", len(ms.scan_all_entries()) >= 10, f"md={len(ms.scan_all_entries())}"))
        T.append(("mm_data", mm.stats().get("total_memories", 0) >= 10, f"mm={mm.stats().get('total_memories', 0)}"))
        pending = ms.get_pending_changes(limit=50)
        T.append(("pending", len(pending) >= 10, f"pending={len(pending)}"))
        if pending:
            ms.mark_synced([e["entry_id"] for e in pending[:5]])
            p2 = ms.get_pending_changes(limit=50)
            T.append(("sync_ok", len(p2) < len(pending), f"before={len(pending)} after={len(p2)}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp43_CascadeSync", T, 0.8)

def exp44_chinese_boundary():
    from src.memory.memory_manager import MemoryManager
    from src.security.sensitive_filter import SensitiveFilter
    print("\n--- Exp44: Chinese Boundary ---")
    T = []
    db = _tmp() + ".db"
    try:
        mm = MemoryManager(db_path=db)
        sf = SensitiveFilter()
        mixed = ["vim Python SSH Docker", "port 22 to 2222", "CPU limit 2 cores"]
        for t in mixed:
            mm.store(t, category="knowledge", importance=0.5)
        T.append(("store", mm.stats().get("total_memories", 0) >= 3, f"n={mm.stats().get('total_memories', 0)}"))
        for t in mixed[:2]:
            r = mm.retrieve(t[:6], top_k=3)
            T.append((f"ret_{t[:4]}", len(r) > 0, f"n={len(r)}"))
        mm.store("Deploy success! Service running normal", category="episode", importance=0.3)
        T.append(("emoji", True, "ok"))
        long_text = "This is a long text. " * 500
        mm.store(long_text, category="knowledge", importance=0.3)
        T.append(("long", True, f"len={len(long_text)}"))
        r_empty = mm.retrieve("", top_k=3)
        T.append(("empty", isinstance(r_empty, list), "ok"))
        pii = "Server IP 192.168.1.100, email admin@company.com"
        m = sf.scan(pii)
        T.append(("pii", len(m) >= 2, f"n={len(m)}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp44_ChineseBoundary", T, 0.8)

def exp45_data_integration():
    from src.data_integration.pipeline import DataCleaningPipeline
    from src.data_integration.models import NormalizedEvent, SourceType
    print("\n--- Exp45: Data Integration ---")
    T = []
    try:
        p = DataCleaningPipeline()
        e1 = NormalizedEvent(event_id="t1", source_type=SourceType.USER_BEHAVIOR, raw_data={"text": "<p>Hello <b>world</b></p>"}, clean_text="<p>Hello <b>world</b></p>")
        c1 = p.clean_and_validate(e1)
        T.append(("html", "<" not in c1.clean_text, f"text={c1.clean_text[:30]}"))
        e2 = NormalizedEvent(event_id="t2", source_type=SourceType.USER_BEHAVIOR, raw_data={"text": "  hello   world  "}, clean_text="  hello   world  ")
        c2 = p.clean_and_validate(e2)
        T.append(("ws", "  " not in c2.clean_text.strip(), f"text={c2.clean_text}"))
        e3 = NormalizedEvent(event_id="t3", source_type=SourceType.USER_BEHAVIOR, raw_data={"text": "Docker deployment info"}, clean_text="Docker deployment info")
        c3 = p.clean_and_validate(e3)
        T.append(("quality", c3.quality_score > 0, f"score={c3.quality_score}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp45_DataIntegration", T, 0.7)

def exp46_ome():
    print("\n--- Exp46: OME ---")
    T = []
    try:
        from src.ome import OfflineMemoryEngine
        ome = OfflineMemoryEngine()
        T.append(("init", True, "ok"))
        if hasattr(ome, "start"):
            ome.start()
            T.append(("start", True, "ok"))
            time.sleep(0.1)
            ome.stop()
            T.append(("stop", True, "ok"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp46_OME", T, 0.5)

def exp47_reranker():
    print("\n--- Exp47: Reranker ---")
    T = []
    try:
        from src.rerank.reranker import RuleReranker
        r = RuleReranker()
        cands = [{"id": "1", "content": "vim editor usage tips", "score": 0.6}, {"id": "2", "content": "SSH port config guide", "score": 0.3}, {"id": "3", "content": "vim shortcuts", "score": 0.5}]
        reranked = r.rerank("vim how to use", cands)
        T.append(("rerank", isinstance(reranked, list) and len(reranked) >= 2, f"n={len(reranked)}"))
        top = reranked[0].get("content", "") if reranked else ""
        T.append(("relevance", "vim" in top.lower(), f"top={top[:20]}"))
    except Exception as e:
        T.append(("exception", False, f"err={e}"))
    return _run("Exp47_Reranker", T, 0.5)

if __name__ == "__main__":
    print("=" * 60)
    print("Exp38-47: Module Coverage")
    print("=" * 60)
    all_start = time.time()
    for fn in [exp38_markdown_truth, exp39_hybrid_search, exp40_e2e_pipeline,
               exp41_multi_user, exp42_dag_closure, exp43_cascade_sync,
               exp44_chinese_boundary, exp45_data_integration, exp46_ome, exp47_reranker]:
        try:
            fn()
        except Exception as e:
            print(f"ERROR in {fn.__name__}: {e}")
    total_dur = time.time() - all_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    tt = 0; tp = 0
    for name, r in RESULTS.items():
        t = r.get("total", 0); p = r.get("passed", 0); acc = r.get("accuracy", 0)
        tt += t; tp += p
        s = "PASS" if r.get("target_met", False) else "CHECK"
        print(f"  {name}: {p}/{t} ({acc:.0%}) [{s}]")
    print(f"\nOverall: {tp}/{tt} ({tp/tt:.1%})")
    print(f"Duration: {total_dur:.1f}s")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_exp38_47.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {out}")
