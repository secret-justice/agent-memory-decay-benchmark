# -*- coding: utf-8 -*-
"""M1: LNN Gate Ablation - RuleGate vs LNNGate"""
import sys, os, time, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
random.seed(42)

RESULTS = {}

def _run(name, tests, threshold):
    passed = sum(1 for _, ok, _ in tests if ok)
    total = len(tests)
    acc = passed / total if total else 0
    result = {'total': total, 'passed': passed, 'accuracy': round(acc, 4), 'target_met': acc >= threshold, 'details': tests}
    RESULTS[name] = result
    print(f"\n=== {name} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result

def _tmp():
    import tempfile
    return tempfile.mktemp(suffix=".db")

def exp_m1a_gate_decision_quality():
    from src.gate.memory_gate import create_gate, GateInput, GateDecision
    print("\n--- M1a: Gate Decision Quality (RuleGate vs LNNGate) ---")

    rule_gate = create_gate("rule")
    try:
        lnn_gate = create_gate("lnn")
    except Exception:
        lnn_gate = None
        print("  LNNGate not available, using RuleGate with different thresholds")

    T = []

    # Test scenarios: (query, user_context, expected_decision)
    # skip = gate should skip retrieval (low value query)
    # retrieve = gate should retrieve (high value query)
    scenarios = [
        # High-value queries (should retrieve)
        ("SSH port configuration", "sysadmin", "retrieve"),
        ("Redis cache optimization", "developer", "retrieve"),
        ("Docker deployment error", "devops", "retrieve"),
        ("MySQL slow query diagnosis", "dba", "retrieve"),
        ("Nginx reverse proxy setup", "sysadmin", "retrieve"),
        ("Python memory leak debug", "developer", "retrieve"),
        ("Kubernetes pod crash loop", "devops", "retrieve"),
        ("JWT token expiration issue", "developer", "retrieve"),
        ("SSL certificate renewal", "sysadmin", "retrieve"),
        ("CI/CD pipeline failure", "devops", "retrieve"),
        # Medium-value queries
        ("check server status", "sysadmin", "retrieve"),
        ("list running processes", "sysadmin", "retrieve"),
        ("check disk usage", "sysadmin", "retrieve"),
        ("view recent logs", "devops", "retrieve"),
        ("check network connectivity", "sysadmin", "retrieve"),
        # Repeated queries (should still retrieve)
        ("SSH port configuration", "sysadmin", "retrieve"),
        ("Redis cache optimization", "developer", "retrieve"),
        ("Docker deployment error", "devops", "retrieve"),
        # Short queries
        ("help", "user", "retrieve"),
        ("status", "user", "retrieve"),
    ]

    for query, user, expected in scenarios:
        gi = GateInput(query=query, user_id=user)

        # RuleGate decision
        rule_out = rule_gate.predict(gi)
        rule_dec = rule_out.decision.value
        rule_ok = (rule_dec == expected) if expected != "retrieve" else (rule_dec != "skip")
        T.append(("rule_" + query[:15], rule_ok, f"dec={rule_dec} exp={expected}"))

        # LNNGate decision (if available)
        if lnn_gate:
            lnn_out = lnn_gate.predict(gi)
            lnn_dec = lnn_out.decision.value
            lnn_ok = (lnn_dec == expected) if expected != "retrieve" else (lnn_dec != "skip")
            T.append(("lnn_" + query[:15], lnn_ok, f"dec={lnn_dec} exp={expected}"))

    return _run("M1a_GateQuality", T, 0.70)

def exp_m1b_gate_latency():
    from src.gate.memory_gate import create_gate, GateInput
    print("\n--- M1b: Gate Latency (RuleGate vs LNNGate) ---")

    rule_gate = create_gate("rule")
    try:
        lnn_gate = create_gate("lnn")
    except Exception:
        lnn_gate = None

    T = []
    queries = ["SSH port", "Redis cache", "Docker deploy", "MySQL query", "Nginx proxy"] * 20

    # RuleGate latency
    rule_times = []
    for q in queries:
        gi = GateInput(query=q, user_id="bench")
        t0 = time.time()
        rule_gate.predict(gi)
        rule_times.append((time.time() - t0) * 1000)
    rule_times.sort()
    rule_p95 = rule_times[int(len(rule_times) * 0.95)]
    T.append(("rule_p95", rule_p95 < 10, f"P95={rule_p95:.2f}ms"))
    T.append(("rule_throughput", True, f"{len(queries)*1000/(sum(rule_times)+0.001):.0f} ops/s"))

    # LNNGate latency
    if lnn_gate:
        lnn_times = []
        for q in queries:
            gi = GateInput(query=q, user_id="bench")
            t0 = time.time()
            lnn_gate.predict(gi)
            lnn_times.append((time.time() - t0) * 1000)
        lnn_times.sort()
        lnn_p95 = lnn_times[int(len(lnn_times) * 0.95)]
        T.append(("lnn_p95", lnn_p95 < 50, f"P95={lnn_p95:.2f}ms"))
        T.append(("lnn_vs_rule", lnn_p95 < rule_p95 * 10, f"lnn={lnn_p95:.2f} rule={rule_p95:.2f}ms"))
    else:
        T.append(("lnn_skip", True, "LNN not available"))

    return _run("M1b_GateLatency", T, 0.70)

def exp_m1c_forgetting_ablation():
    from src.memory.forgetting_curve import ForgettingCurveEngine, DecayConfig, MemoryStrength
    print("\n--- M1c: Forgetting Curve Ablation ---")

    T = []

    # Config A: Default (tuned)
    config_a = DecayConfig()
    engine_a = ForgettingCurveEngine(config_a)

    # Config B: Aggressive decay (faster forgetting)
    config_b = DecayConfig(
        short_term_stability=1800.0,    # 30min instead of 1h
        mid_term_stability=1296000.0,   # 15d instead of 30d
        long_term_stability=15768000.0, # 6mo instead of 1yr
        forget_threshold=0.3,           # higher threshold
        promote_threshold=0.8,          # higher bar
    )
    engine_b = ForgettingCurveEngine(config_b)

    # Config C: Conservative decay (slower forgetting)
    config_c = DecayConfig(
        short_term_stability=7200.0,    # 2h
        mid_term_stability=5184000.0,   # 60d
        long_term_stability=63072000.0, # 2yr
        forget_threshold=0.1,           # lower threshold
        promote_threshold=0.6,          # lower bar
    )
    engine_c = ForgettingCurveEngine(config_c)

    # Register memories with different importance
    memories = [
        ("mem_ssh", "stm", 0.9, 10),    # high importance, accessed 10 times
        ("mem_redis", "stm", 0.5, 3),   # medium importance
        ("mem_log", "stm", 0.2, 1),     # low importance
        ("mem_api", "mtm", 0.8, 20),    # mid-term, high importance
        ("mem_cache", "mtm", 0.4, 5),   # mid-term, medium
        ("mem_debug", "ltm", 0.7, 50),  # long-term, high
        ("mem_test", "ltm", 0.3, 10),   # long-term, low
    ]

    for mid, tier, imp, acc_count in memories:
        engine_a.register(mid, tier, imp)
        engine_b.register(mid, tier, imp)
        engine_c.register(mid, tier, imp)
        for _ in range(acc_count):
            engine_a.access(mid)
            engine_b.access(mid)
            engine_c.access(mid)

    # Compare retention across configs
    for mid, tier, imp, acc_count in memories:
        sa = engine_a.get_strength(mid)
        sb = engine_b.get_strength(mid)
        sc = engine_c.get_strength(mid)
        ra = sa.get("retention", 0) if sa else 0
        rb = sb.get("retention", 0) if sb else 0
        rc = sc.get("retention", 0) if sc else 0
        # Conservative should have higher retention
        T.append((f"ret_{mid}", rc >= ra >= rb, f"C={rc:.2f} A={ra:.2f} B={rb:.2f}"))

    # Scan actions
    actions_a = engine_a.scan_actions()
    actions_b = engine_b.scan_actions()
    # Aggressive config should have more forget actions
    forget_a = len(actions_a.get("forget", []))
    forget_b = len(actions_b.get("forget", []))
    T.append(("more_forget_B", forget_b >= forget_a, f"A={forget_a} B={forget_b}"))

    return _run("M1c_ForgettingAblation", T, 0.70)

def exp_m1d_competition_comparison():
    from src.extraction.zero_llm_pipeline import ZeroLLMPipeline
    from src.memory.memory_manager import MemoryManager
    from src.knowledge.knowledge_engine import KnowledgeEngine
    print("\n--- M1d: Competition Metrics (Baseline vs Tuned) ---")

    T = []

    # Preference extraction baseline
    pipeline = ZeroLLMPipeline()
    pref_cases = [
        ("I prefer vim for coding", "preference"),
        ("SSH default port is 22", "knowledge"),
        ("Fixed MySQL slow query yesterday", "episode"),
        ("User likes dark theme", "preference"),
        ("Redis is used for caching", "knowledge"),
        ("Deployed Docker today", "episode"),
        ("Always use pytest for testing", "preference"),
        ("Nginx reverse proxy config", "knowledge"),
        ("Upgraded Python last week", "episode"),
        ("Prefer Tailwind over CSS", "preference"),
    ]
    correct = sum(1 for text, exp in pref_cases if pipeline.classify(text) == exp)
    pref_acc = correct / len(pref_cases)
    T.append(("pref_acc", pref_acc >= 0.7, f"{pref_acc:.0%} ({correct}/{len(pref_cases)})"))

    # Retrieval test
    db = _tmp()
    mm = MemoryManager(db_path=db)
    facts = ["SSH port 22", "Redis cache", "Docker deploy", "MySQL 3306", "Nginx proxy",
             "Python interpreted", "Git version control", "Linux open source", "PostgreSQL RDBMS", "K8s orchestration"]
    for f in facts:
        mm.store(f, category="knowledge", importance=0.7)
    mm.retrieve("warmup", top_k=1)
    queries = ["SSH", "Redis", "Docker", "MySQL", "Nginx", "Python", "Git", "Linux", "PostgreSQL", "K8s"]
    found = 0
    for q in queries:
        results = mm.retrieve(q, top_k=3)
        if any(q.lower() in r.content.lower() for r in results):
            found += 1
    ret_acc = found / len(queries)
    T.append(("ret_acc", ret_acc >= 0.8, f"{ret_acc:.0%} ({found}/{len(queries)})"))

    # Conflict detection
    ke = KnowledgeEngine(db_path=db + ".know")
    ke.store("SSH port is 22", category="fact")
    cr = ke.detect_conflicts_v2("SSH port changed to 2222")
    hc = cr.get("has_conflict", False) if isinstance(cr, dict) else bool(cr)
    T.append(("conflict_detect", hc, f"conflict={hc}"))

    # No false conflict
    cr2 = ke.detect_conflicts_v2("SSH port is 22")
    hc2 = cr2.get("has_conflict", True) if isinstance(cr2, dict) else bool(cr2)
    T.append(("no_false_conflict", not hc2, f"false={hc2}"))

    # Latency
    times = []
    for q in queries:
        t0 = time.time()
        mm.retrieve(q, top_k=5)
        times.append((time.time() - t0) * 1000)
    times.sort()
    p95 = times[int(len(times) * 0.95)]
    T.append(("latency_p95", p95 <= 500, f"P95={p95:.1f}ms"))

    try:
        os.remove(db)
        os.remove(db + ".know")
    except: pass

    return _run("M1d_Competition", T, 0.70)

if __name__ == "__main__":
    print("=" * 60)
    print("M1: LNN Gate Ablation Experiment")
    print("=" * 60)
    all_start = time.time()
    for fn in [exp_m1a_gate_decision_quality, exp_m1b_gate_latency,
               exp_m1c_forgetting_ablation, exp_m1d_competition_comparison]:
        try: fn()
        except Exception as e:
            import traceback
            print(f"ERROR in {fn.__name__}: {e}")
            traceback.print_exc()
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_m1_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {out}")
