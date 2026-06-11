# -*- coding: utf-8 -*-
"""Exp-3 Only: Per-Timestep Ranking on PersonaChat (memory-efficient)"""
import os, io, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_models import Retriever, INPUT_DIM

torch.manual_seed(42); np.random.seed(42)
OD = os.path.dirname(os.path.abspath(__file__))
HD = 16; SEEDS = [42, 123, 256]; EPOCHS = 15
VARIANTS = ['cct_lnn', 'ltc', 'gru', 'gru_time', 'neural_ode', 'cfc', 'param_expdecay', 'transformer']

def load_personachat():
    with open(os.path.join(OD, "personachat_enhanced.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    cat_list = sorted(set(m["category"] for u in data for m in u["memory_bank"]))
    cat_map = {c: i for i, c in enumerate(cat_list)}
    seqs = []
    for u in data:
        mems = sorted(u["memory_bank"], key=lambda m: m["timestamp"])
        if len(mems) < 5: continue
        X = torch.tensor([m["query"] for m in mems], dtype=torch.float32)
        ts = torch.tensor([m["timestamp"] for m in mems], dtype=torch.float32)
        cats = torch.tensor([cat_map[m["category"]] for m in mems], dtype=torch.long)
        seqs.append({"X": X, "ts": ts, "cats": cats})
    return seqs, cat_list

def build_ranking_tasks(seqs, window=8, max_per_user=4):
    tasks = []
    for seq in seqs:
        X, ts, cats = seq["X"], seq["ts"], seq["cats"]
        n = len(X); c = 0
        for i in range(max(1, n - window), n):
            if c >= max_per_user: break
            s = max(0, i - window)
            if i - s < 3: continue
            sims = torch.cosine_similarity(X[i:i+1], X[s:i], dim=1)
            dt = ts[i] - ts[s:i]
            decay = torch.exp(-0.01 * dt)
            relevance = torch.clamp(sims * decay, 0.0, 1.0)
            tasks.append({"X": X[s:i], "ts": ts[s:i] - ts[s], "cats": cats[s:i], "relevance": relevance})
            c += 1
    return tasks

def compute_ndcg(rel_np, pred_np, k):
    ideal = np.sort(rel_np)[::-1][:k]
    pred_order = np.argsort(pred_np)[::-1][:k]
    dcg = sum(max(float(rel_np[pred_order[i]]), 0.0) / np.log2(i + 2) for i in range(k))
    idcg = sum(max(float(ideal[i]), 0.0) / np.log2(i + 2) for i in range(k))
    return dcg / max(idcg, 1e-8)

def train_eval_ranking(variant, train, val, test, seed, n_cats):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Retriever(INPUT_DIM, HD, variant, n_cats)
    lr = 0.002 if variant == 'cct_lnn' else 0.005
    ep_count = EPOCHS if variant != 'cct_lnn' else 20
    opt = optim.Adam(model.parameters(), lr=lr)
    # Per-step scorer
    step_scorer = nn.Sequential(nn.Linear(HD, HD), nn.ReLU(), nn.Linear(HD, 1), nn.Sigmoid())
    opt2 = optim.Adam(step_scorer.parameters(), lr=lr)
    mse = nn.MSELoss()
    best_val = 1e9; best_state = None; best_scorer = None; patience = 5; ni = 0

    for ep in range(ep_count):
        model.train(); step_scorer.train()
        for idx in np.random.permutation(len(train)):
            d = train[idx]
            opt.zero_grad(); opt2.zero_grad()
            H, _ = model.cell.forward_seq(d["X"], d["ts"], d["cats"])
            preds = step_scorer(H).squeeze(-1)
            tgt = d["relevance"]
            loss = mse(preds, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(step_scorer.parameters(), 1.0)
            opt.step(); opt2.step()

        model.eval(); step_scorer.eval(); val_ndcg = []
        with torch.no_grad():
            for d in val:
                H, _ = model.cell.forward_seq(d["X"], d["ts"], d["cats"])
                preds = step_scorer(H).squeeze(-1).cpu().numpy()
                rel = d["relevance"].numpy()
                if len(preds) >= 3:
                    val_ndcg.append(compute_ndcg(rel, preds, min(5, len(preds))))
        vloss = 1.0 - np.mean(val_ndcg) if val_ndcg else 1.0
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_scorer = {k: v.clone() for k, v in step_scorer.state_dict().items()}
            ni = 0
        else:
            ni += 1
        if ni >= patience: break

    if best_state: model.load_state_dict(best_state)
    if best_scorer: step_scorer.load_state_dict(best_scorer)
    model.eval(); step_scorer.eval()
    ndcgs = []; mrrs = []; kendalls = []
    with torch.no_grad():
        for d in test:
            H, _ = model.cell.forward_seq(d["X"], d["ts"], d["cats"])
            preds = step_scorer(H).squeeze(-1).cpu().numpy()
            rel = d["relevance"].numpy()
            if len(preds) < 3: continue
            k = min(5, len(preds))
            ndcgs.append(compute_ndcg(rel, preds, k))
            pred_order = np.argsort(preds)[::-1]
            rel_order = np.argsort(rel)[::-1]
            for rank, idx in enumerate(pred_order):
                if idx == rel_order[0]:
                    mrrs.append(1.0 / (rank + 1)); break
            else:
                mrrs.append(0.0)
            tau, _ = stats.kendalltau(preds, rel)
            if not np.isnan(tau): kendalls.append(float(tau))
    return {"ndcg5": float(np.mean(ndcgs)) if ndcgs else 0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0,
            "kendall_tau": float(np.mean(kendalls)) if kendalls else 0}

print("Loading PersonaChat...", flush=True)
pc_seqs, pc_cats = load_personachat()
print(f"PersonaChat: {len(pc_seqs)} users, {len(pc_cats)} cats")

pc_rank = build_ranking_tasks(pc_seqs, window=8, max_per_user=4)
np.random.shuffle(pc_rank)
n = len(pc_rank)
pc_tr = pc_rank[:int(n*.6)]
pc_va = pc_rank[int(n*.6):int(n*.8)]
pc_te = pc_rank[int(n*.8):]
print(f"Ranking: {n} total (tr={len(pc_tr)} va={len(pc_va)} te={len(pc_te)})")

pc_rel = np.concatenate([d["relevance"].numpy() for d in pc_rank])
print(f"Relevance: mean={pc_rel.mean():.4f} std={pc_rel.std():.4f} min={pc_rel.min():.4f} max={pc_rel.max():.4f}")

print("\n" + "=" * 70)
print("EXP-3: PER-TIMESTEP RANKING (PersonaChat, 3 seeds)")
print("=" * 70)
e3_pc = {}
for v in VARIANTS:
    print(f"  {v}...", end="", flush=True); t = time.time()
    runs = [train_eval_ranking(v, pc_tr, pc_va, pc_te, s, len(pc_cats)) for s in SEEDS]
    e3_pc[v] = {}
    for k in ["ndcg5", "mrr", "kendall_tau"]:
        vals = [r[k] for r in runs]
        e3_pc[v][k] = float(np.mean(vals))
        e3_pc[v][k + "_std"] = float(np.std(vals))
    elapsed = time.time() - t
    print(f" NDCG5={e3_pc[v]['ndcg5']:.3f}({e3_pc[v]['ndcg5_std']:.3f}) MRR={e3_pc[v]['mrr']:.3f} Kendall={e3_pc[v]['kendall_tau']:.3f} ({elapsed:.0f}s)")
    gc.collect()

out = os.path.join(OD, "v21_c0_exp3_results.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(e3_pc, f, ensure_ascii=False, indent=2)
print(f"\nExp-3 results saved: {out}")
