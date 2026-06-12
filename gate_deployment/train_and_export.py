# -*- coding: utf-8 -*-
"""Train GRU+Time gate, export ONNX."""
import torch, torch.nn as nn, torch.optim as optim, json, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gru_time_gate import GRUTimeGate

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

with open(os.path.join(OD, "personachat_enhanced.json"), 'r', encoding='utf-8') as f:
    data = json.load(f)
seqs = []
for u in data:
    mems = sorted(u['memory_bank'], key=lambda m: m['timestamp'])
    if len(mems) < 5: continue
    X = torch.tensor([m['query'] for m in mems], dtype=torch.float32)
    ts = torch.tensor([m['timestamp'] for m in mems], dtype=torch.float32)
    seqs.append({'X': X, 'ts': ts})

tasks = []
for seq in seqs:
    X=seq['X'];ts=seq['ts'];n=len(X)
    for i in range(max(1,n-5),n):
        s=max(0,i-5)
        sim=torch.cosine_similarity(X[i:i+1],X[s:i],dim=1)
        kept=(sim.max()>0.3).float()
        elapsed=float(ts[i]-ts[max(0,i-1)])
        tasks.append({'query':X[i],'elapsed':elapsed,'target':kept})

np.random.shuffle(tasks); n=len(tasks)
tr=tasks[:int(n*.6)]; va=tasks[int(n*.6):int(n*.8)]

torch.manual_seed(42)
gate = GRUTimeGate(input_dim=64, hidden_dim=16)
opt = optim.Adam(gate.parameters(), lr=0.005)
crit = nn.BCELoss()

print("Training GRU+Time gate...", flush=True)
for ep in range(20):
    gate.train(); tl=0
    idx_list = list(np.random.permutation(len(tr)))
    for idx in idx_list:
        d = tr[idx]
        opt.zero_grad()
        q = d['query'].unsqueeze(0)
        t = torch.tensor([[d['elapsed']]])
        pred = gate(q, t)
        loss = crit(pred.squeeze(), d['target'])
        loss.backward(); opt.step()
        tl += loss.item()
    tl /= len(tr)
    
    gate.eval(); vl=0
    for idx in range(len(va)):
        d = va[idx]
        with torch.no_grad():
            q = d['query'].unsqueeze(0)
            t = torch.tensor([[d['elapsed']]])
            pred = gate(q, t)
            vl += crit(pred.squeeze(), d['target']).item()
    vl /= len(va)
    if ep % 5 == 0:
        print(f"  Epoch {ep}: train={tl:.4f} val={vl:.4f}", flush=True)

# Export ONNX
gate.eval()
dummy_q = torch.randn(1, 64)
dummy_t = torch.randn(1, 1)
onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gru_time_gate.onnx")
torch.onnx.export(
    gate, (dummy_q, dummy_t), onnx_path,
    input_names=['query_emb', 'elapsed_time'],
    output_names=['keep_prob'],
    dynamic_axes={'query_emb':{0:'batch'}, 'elapsed_time':{0:'batch'}, 'keep_prob':{0:'batch'}},
    opset_version=11
)
print(f"\nONNX: {onnx_path} ({os.path.getsize(onnx_path)} bytes)")

# Speed test
import time as time_mod
times = []
with torch.no_grad():
    for _ in range(100):
        t0 = time_mod.perf_counter()
        gate(dummy_q, dummy_t)
        times.append((time_mod.perf_counter()-t0)*1000)
print(f"Latency: P50={np.median(times):.3f}ms P95={np.percentile(times,95):.3f}ms")
print(f"Params: {sum(p.numel() for p in gate.parameters())}")
