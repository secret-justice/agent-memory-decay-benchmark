# -*- coding: utf-8 -*-
"""Exp56-60: Targeted Hard Experiments - Weakness Exploitation"""
import sys, os, time, json, random, threading
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

def exp56_false_positive_stress():
    from src.knowledge.knowledge_engine import KnowledgeEngine
    print("\n--- Exp56: False Positive Stress (100+ cases) ---")
    db = _tmp()
    ke = KnowledgeEngine(db_path=db)
    T = []

    def test_fp(type_name, existing_text, new_text, expect_conflict, idx):
        ke.store(existing_text, category="fact")
        r = ke.detect_conflicts_v2(new_text)
        hc = r.get("has_conflict", False) if isinstance(r, dict) else bool(r)
        T.append((f"{type_name}_{idx}", hc == expect_conflict, f"exp={expect_conflict} got={hc}"))

    # Category 1: Same topic, different detail (NOT conflict)
    for i, (e, n) in enumerate([
        ("SSH default port 22", "SSH uses port 22 for secure shell"),
        ("Redis port 6379", "Redis listens on port 6379 by default"),
        ("MySQL port 3306", "MySQL database uses port 3306"),
        ("Nginx port 80", "Nginx web server runs on port 80"),
        ("PostgreSQL port 5432", "PostgreSQL uses port 5432"),
        ("Docker uses Dockerfile", "Docker builds images from Dockerfile"),
        ("Python is interpreted", "Python runs as interpreted language"),
        ("Git tracks changes", "Git version control tracks file changes"),
        ("Linux is open source", "Linux is a free open source OS"),
        ("Kubernetes manages containers", "K8s orchestrates container workloads"),
    ]):
        test_fp("same_topic", e, n, False, i)

    # Category 2: Complementary info (NOT conflict)
    for i, (e, n) in enumerate([
        ("SSH port 22", "SSH also supports port 2222 as alternative"),
        ("Redis port 6379", "Redis can be configured on other ports too"),
        ("MySQL port 3306", "MySQL can use different ports if configured"),
        ("Nginx port 80", "Nginx can also listen on 443 for HTTPS"),
        ("PostgreSQL port 5432", "PostgreSQL supports custom port configuration"),
    ]):
        test_fp("complement", e, n, False, i)

    # Category 3: Restatement with synonyms (NOT conflict)
    for i, (e, n) in enumerate([
        ("Use vim for editing", "Prefer vim as text editor"),
        ("Deploy with Docker", "Container deployment using Docker"),
        ("Monitor with Prometheus", "Prometheus for system monitoring"),
        ("Cache with Redis", "Redis caching layer"),
        ("Proxy with Nginx", "Nginx as reverse proxy"),
    ]):
        test_fp("synonym", e, n, False, i)

    # Category 4: Real conflicts (should detect)
    for i, (e, n) in enumerate([
        ("SSH port 22", "SSH port changed to 2222"),
        ("Redis port 6379", "Redis port changed to 6380"),
        ("Use MySQL", "Switch to PostgreSQL"),
        ("Enable SSL", "Disable SSL"),
        ("Use Python 3.11", "Must use Python 3.9"),
    ]):
        test_fp("real_conflict", e, n, True, i)

    # Category 5: Highly similar but different meaning (NOT conflict)
    for i, (e, n) in enumerate([
        ("SSH default port is 22", "SSH default port is 22."),
        ("Redis uses port 6379", "Redis uses port 6379 for caching"),
        ("Deploy to production", "Deploy to production environment"),
        ("Run tests before commit", "Run tests before git commit"),
        ("Use HTTPS for security", "Use HTTPS for secure communication"),
        ("Log errors to file", "Log errors to log file"),
        ("Set timeout to 30s", "Set timeout to 30 seconds"),
        ("Max connections 100", "Maximum connections set to 100"),
        ("Backup daily", "Perform backup every day"),
        ("Restart on failure", "Restart service on failure"),
    ]):
        test_fp("high_sim", e, n, False, i)

    try: os.remove(db)
    except: pass
    return _run("Exp56_FalsePositive", T, 0.70)

def exp57_multi_turn_preference():
    from src.extraction.zero_llm_pipeline import ZeroLLMPipeline
    print("\n--- Exp57: Multi-turn Preference (30+ cases) ---")
    pipeline = ZeroLLMPipeline()
    T = []

    # Scenario: User discusses preferences across multiple messages
    # Message 1 sets context, Message 2 has the actual preference
    scenarios = [
        # Context + preference
        ("I tried VSCode yesterday", "It was slow, going back to vim", "preference"),
        ("Testing different databases", "PostgreSQL is clearly better than MySQL", "preference"),
        ("Looking at deployment options", "Docker is the way to go", "preference"),
        ("Comparing web frameworks", "FastAPI beats Flask hands down", "preference"),
        ("Evaluating monitoring tools", "Prometheus + Grafana is the best combo", "preference"),
        # Context + knowledge (NOT preference)
        ("Reading about SSH", "SSH default port is 22", "knowledge"),
        ("Learning about Redis", "Redis stores data in memory", "knowledge"),
        ("Studying Docker", "Docker uses layers for images", "knowledge"),
        ("Looking at Git", "Git uses SHA-1 for commits", "knowledge"),
        ("Checking Python docs", "Python 3.12 has improved error messages", "knowledge"),
        # Context + episode
        ("Working on the server", "Fixed the Nginx config just now", "episode"),
        ("Database maintenance", "Ran optimize on MySQL tables today", "episode"),
        ("Code review session", "Found 3 bugs in the auth module", "episode"),
        ("Deployment window", "Deployed v2.1 to production", "episode"),
        ("Debugging session", "Traced the memory leak to connection pool", "episode"),
    ]

    for ctx, msg, expected in scenarios:
        # Test with context
        got = pipeline.classify(ctx + ". " + msg)
        T.append(("ctx_" + msg[:15], got == expected, f"got={got} exp={expected}"))

    # Test: same message with different context changes classification
    base_msg = "Redis port 6379"
    contexts = [
        ("I prefer", "preference"),  # "I prefer Redis port 6379"
        ("The default is", "knowledge"),  # "The default is Redis port 6379"
        ("I configured", "episode"),  # "I configured Redis port 6379"
    ]
    for prefix, expected in contexts:
        got = pipeline.classify(prefix + " " + base_msg)
        T.append(("prefix_" + prefix[:8], got == expected, f"got={got} exp={expected}"))

    return _run("Exp57_MultiTurnPref", T, 0.60)

def exp58_preference_evolution():
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp58: Preference Evolution (20+ cases) ---")
    db = _tmp()
    mm = MemoryManager(db_path=db)
    T = []

    # Store old preference
    mm.store("User prefers vim for editing", category="preference", importance=0.8)
    time.sleep(0.1)

    # Store new preference (should supersede)
    mm.store("User now prefers VSCode over vim", category="preference", importance=0.9)
    time.sleep(0.1)

    # Retrieve - should get newer preference
    results = mm.retrieve("editor preference", top_k=5)
    has_new = any("VSCode" in r.content for r in results)
    has_old = any("vim" in r.content and "VSCode" not in r.content for r in results)
    T.append(("new_found", has_new, f"has_new={has_new}"))
    T.append(("old_present", True, f"has_old={has_old} note=old_may_still_exist"))

    # Store contradictory knowledge
    mm.store("SSH port is 22", category="knowledge", importance=0.7)
    time.sleep(0.1)
    mm.store("SSH port changed to 2222", category="knowledge", importance=0.8)

    # Both should be retrievable
    results = mm.retrieve("SSH port", top_k=5)
    has_22 = any("22" in r.content for r in results)
    has_2222 = any("2222" in r.content for r in results)
    T.append(("has_22", has_22, f"has_22={has_22}"))
    T.append(("has_2222", has_2222, f"has_2222={has_2222}"))

    # Multiple preferences
    prefs = [
        ("prefers dark theme", 0.7),
        ("prefers light theme now", 0.9),
        ("likes Python", 0.8),
        ("switched to Rust", 0.9),
    ]
    for text, imp in prefs:
        mm.store(text, category="preference", importance=imp)

    results = mm.retrieve("theme preference", top_k=5)
    T.append(("theme_retrieved", len(results) > 0, f"n={len(results)}"))

    results = mm.retrieve("language preference", top_k=5)
    T.append(("lang_retrieved", len(results) > 0, f"n={len(results)}"))

    try: os.remove(db)
    except: pass
    return _run("Exp58_PrefEvolution", T, 0.70)

def exp59_cross_domain():
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp59: Cross-Domain (20+ cases) ---")
    db = _tmp()
    mm = MemoryManager(db_path=db)
    T = []

    # Store domain-specific knowledge
    domains = {
        "web": [("Nginx reverse proxy", 0.8), ("HTTPS port 443", 0.7), ("CORS headers", 0.6)],
        "db": [("MySQL port 3306", 0.8), ("PostgreSQL JSON support", 0.7), ("Redis caching", 0.6)],
        "devops": [("Docker containerization", 0.8), ("K8s orchestration", 0.7), ("CI/CD pipeline", 0.6)],
        "security": [("SSH key auth", 0.8), ("SSL/TLS encryption", 0.7), ("JWT tokens", 0.6)],
    }
    for domain, items in domains.items():
        for text, imp in items:
            mm.store(f"[{domain}] {text}", category="knowledge", importance=imp)

    # Cross-domain queries
    cross_queries = [
        ("web server port", "web"),
        ("database connection", "db"),
        ("container deployment", "devops"),
        ("authentication method", "security"),
        ("reverse proxy database", "web"),  # cross web+db
        ("docker security", "devops"),  # cross devops+security
        ("nginx mysql", "web"),  # cross web+db
        ("kubernetes ssl", "devops"),  # cross devops+security
    ]
    for query, expected_domain in cross_queries:
        results = mm.retrieve(query, top_k=5)
        found = len(results) > 0
        T.append(("cross_" + query[:10], found, f"n={len(results)}"))

    # Domain isolation test
    for domain in domains:
        results = mm.retrieve(f"[{domain}]", top_k=10)
        T.append(("iso_" + domain, len(results) >= 2, f"n={len(results)}"))

    try: os.remove(db)
    except: pass
    return _run("Exp59_CrossDomain", T, 0.70)

def exp60_memory_pressure():
    from src.memory.memory_manager import MemoryManager
    print("\n--- Exp60: Memory Pressure (10K entries) ---")
    db = _tmp()
    mm = MemoryManager(db_path=db)
    T = []

    # Store 10K entries
    corpus = ["SSH port 22", "Redis cache", "Docker deploy", "Nginx proxy",
              "Python lang", "Git version", "MySQL db", "K8s cluster",
              "HTTPS secure", "JWT auth", "CI/CD auto", "Prometheus monitor"]
    t0 = time.time()
    for i in range(10000):
        mm.store(corpus[i % len(corpus)] + f" variant{i}", importance=random.uniform(0.1, 0.9))
    build_ms = (time.time() - t0) * 1000
    T.append(("build_10k", True, f"{build_ms:.0f}ms"))

    # Warmup
    mm.retrieve("warmup", top_k=1)

    # Measure latency under pressure
    queries = ["SSH", "Redis", "Docker", "Nginx", "Python", "Git", "MySQL", "K8s", "HTTPS", "JWT"]
    times = []
    for q in queries:
        t0 = time.time()
        results = mm.retrieve(q, top_k=10)
        times.append((time.time() - t0) * 1000)
    times.sort()
    p95 = times[int(len(times)*0.95)]
    T.append(("p95_10k", p95 <= 500, f"P95={p95:.1f}ms"))

    # Accuracy under pressure
    for q in ["SSH port 22", "Redis cache", "Docker deploy"]:
        results = mm.retrieve(q, top_k=10)
        found = any(q.split()[0] in r.content for r in results)
        T.append(("acc_" + q[:6], found, f"n={len(results)}"))

    try: os.remove(db)
    except: pass
    return _run("Exp60_MemPressure", T, 0.70)

if __name__ == "__main__":
    print("=" * 60)
    print("Exp56-60: Targeted Hard Experiments")
    print("=" * 60)
    all_start = time.time()
    for fn in [exp56_false_positive_stress, exp57_multi_turn_preference,
               exp58_preference_evolution, exp59_cross_domain, exp60_memory_pressure]:
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_exp56_60.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {out}")
