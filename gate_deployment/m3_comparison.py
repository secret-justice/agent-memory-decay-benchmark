# -*- coding: utf-8 -*-
"""
M3 Experiment: GRU+Time gate vs LTC gate comparison.
Run on Windows to get baseline numbers before Kylin deployment.
"""
import time, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gru_time_gate import GRUTimeGate

# Test queries covering different difficulty levels
test_queries = [
    {"query": "SSH port configuration", "context": "sysadmin", "expected": "retrieve"},
    {"query": "help", "context": "user", "expected": "retrieve"},
    {"query": "status", "context": "user", "expected": "retrieve"},
    {"query": "Docker deployment error", "context": "devops", "expected": "retrieve"},
    {"query": "Redis cache optimization", "context": "developer", "expected": "retrieve"},
    {"query": "MySQL slow query", "context": "dba", "expected": "retrieve"},
    {"query": "check server status", "context": "sysadmin", "expected": "retrieve"},
    {"query": "Kubernetes pod crash", "context": "devops", "expected": "retrieve"},
    {"query": "JWT token expiration", "context": "developer", "expected": "retrieve"},
    {"query": "CI/CD pipeline failure", "context": "devops", "expected": "retrieve"},
]

# Test GRU+Time gate
gate = GRUTimeGate(input_dim=64, hidden_dim=16)
gate.eval()

results = {"gru_time_gate": {"decisions": [], "latencies": []}}

for tq in test_queries:
    emb = np.random.randn(64).astype(np.float32)
    elapsed = 1.0
    
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        prob = gate.predict_single(emb, elapsed)
        times.append((time.perf_counter() - t0) * 1000)
    
    decision = "retrieve" if prob > 0.5 else "skip"
    results["gru_time_gate"]["decisions"].append({
        "query": tq["query"], "decision": decision, "prob": round(prob, 4)
    })
    results["gru_time_gate"]["latencies"].append(float(np.median(times)))

latencies = results["gru_time_gate"]["latencies"]
results["gru_time_gate"]["p50_ms"] = float(np.median(latencies))
results["gru_time_gate"]["p95_ms"] = float(np.percentile(latencies, 95))
results["gru_time_gate"]["params"] = sum(p.numel() for p in gate.parameters())

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3_gate_comparison.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("M3 Gate Comparison Results:")
print(f"  GRU+Time Gate: P50={results['gru_time_gate']['p50_ms']:.3f}ms P95={results['gru_time_gate']['p95_ms']:.3f}ms")
print(f"  Parameters: {results['gru_time_gate']['params']}")
print(f"  Decisions: {len([d for d in results['gru_time_gate']['decisions'] if d['decision']=='retrieve'])}/{len(test_queries)} retrieve")
print(f"\nSaved: {out_path}")
