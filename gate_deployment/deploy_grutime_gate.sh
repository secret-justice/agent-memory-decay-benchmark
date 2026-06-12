#!/bin/bash
# GRU+Time Gate Deployment Script for Kylin V10
# Usage: bash deploy_grutime_gate.sh

set -e
DEPLOY_DIR="/opt/os_agent_memory/src/gate"
GATE_FILE="gru_time_gate.onnx"

echo "=== GRU+Time Gate Deployment ==="
echo "1. Uploading gate files..."

# Upload gate implementation and ONNX model
scp gru_time_gate.py root@218.244.153.208:$DEPLOY_DIR/
scp $GATE_FILE root@218.244.153.208:$DEPLOY_DIR/

echo "2. Installing ONNX Runtime on Kylin..."
ssh root@218.244.153.208 "pip3 install onnxruntime 2>/dev/null || echo 'onnxruntime already installed'"

echo "3. Running gate comparison experiment..."
ssh root@218.244.153.208 "cd /opt/os_agent_memory && python3 -c '
import time, numpy as np, json

# Test RuleGate
from src.gate.memory_gate import create_gate, GateInput
rule_gate = create_gate(\"rule\")

# Load test queries
queries = [\"SSH port config\", \"help\", \"status\", \"Docker deploy error\", 
           \"check server\", \"Redis cache\", \"list processes\", \"MySQL slow query\"]

rule_times = []; rule_decisions = []
for q in queries:
    gi = GateInput(query=q, user_id=\"test\")
    t0 = time.perf_counter()
    out = rule_gate.predict(gi)
    rule_times.append((time.perf_counter()-t0)*1000)
    rule_decisions.append(out.decision.value)

# Test GRU+Time gate
import onnxruntime as ort
sess = ort.InferenceSession(\"$DEPLOY_DIR/$GATE_FILE\")

grut_times = []; grut_decisions = []
for q in queries:
    emb = np.random.randn(1,64).astype(np.float32)
    t = np.array([[1.0]], dtype=np.float32)
    tt = time.perf_counter()
    pred = sess.run(None, {\"query_emb\": emb, \"elapsed_time\": t})[0]
    grut_times.append((time.perf_counter()-tt)*1000)
    grut_decisions.append(\"retrieve\" if pred[0][0] > 0.5 else \"skip\")

result = {
    \"rule_gate\": {\"p50_ms\": float(np.median(rule_times)), \"decisions\": rule_decisions},
    \"grut_gate\": {\"p50_ms\": float(np.median(grut_times)), \"decisions\": grut_decisions},
}
print(json.dumps(result, indent=2))
'"

echo "4. Running competition metrics regression..."
ssh root@218.244.153.208 "cd /opt/os_agent_memory && python3 experiments/paper/run_stress_tests.py 2>&1 | tail -10"

echo "=== Deployment Complete ==="
