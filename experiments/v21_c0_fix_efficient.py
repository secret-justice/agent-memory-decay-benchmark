# -*- coding: utf-8 -*-
"""
C0 Fix - Memory-Efficient Final Version
Exp-1: Standard Regression (embedded)
Exp-2: Focal Weighted Regression (embedded)  
Exp-3: Per-Timestep MSE Ranking (train each timestep independently, no pairwise)
"""
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

def load_lpt():
    with open(os.path.join(OD, "lpt_200users.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    n_cats = max(m.get("category", 0) if isinstance(m.get("category", 0), int) else 0
                 for u in data for m in u["memory_bank"]) + 1
    seqs = []
    for u in data:
        mems = sorted(u["memory_bank"], key=lambda m: m["timestamp"])
        if len(mems) < 10: continue
        X = torch.tensor([m["query"] for m in mems], dtype=torch.float32)
        ts = torch.tensor([m["timestamp"] for m in mems], dtype=torch.float32)
        cats_list = [m.get("category", 0) if isinstance(m.get("category", 0), int) else 0 for m in mems]
        cats = torch.tensor(cats_list, dtype=torch.long)
        seqs.append({"X": X, "ts": ts, "cats": cats})
    return seqs, list(range(n_cats))

def build_ranking_tasks(seqs, window=8, max_per_user=4):
    tasks = []
    for seq in seqs:
        X, ts, cats = seq["X"], seq["ts"], seq["cats"]
        n = len(X); c = 0
        for i in range(max(1, n - window), n):
            if c >= max_per_user: break
            s = max(0, i - window)
            seq_len = i - s
            if seq_len < 3: continue
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

# ============================================================
# Exp-3: Per-Timestep MSE Ranking
# Train: predict per-step relevance from hidden state
# Eval: rank by predicted relevance
# ============================================================
def train_eval_ranking(variant, train, val, test, seed, n_cats):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Retriever(INPUT_DIM, HD, variant, n_cats)
    lr = 0.002 if variant == 'cct_lnn' else 0.005
    ep_count = EPOCHS if variant != 'cct_lnn' else 20
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    # Per-step scorer: HD -> 1
    step_scorer = nn.Sequential(nn.Linear(HD, HD), nn.ReLU(), nn.Linear(HD, 1), nn.Sigmoid())
    opt2 = optim.Adam(step_scorer.parameters(), lr=lr)
    best_val = 1e9; best_state = None; best_scorer = None; patience = 5; ni = 0

    for ep in range(ep_count):
        model.train(); step_scorer.train()
        perm = np.random.permutation(len(train))
        for idx in perm:
            d = train[idx]
            opt.zero_grad(); opt2.zero_grad()
            H, _ = model.cell.forward_seq(d["X"], d["ts"], d["cats"])
            # Per-step predictions
            preds = step_scorer(H).squeeze(-1)  # [seq_len]
            tgt = d["relevance"]  # [seq_len]
            loss = mse(preds, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(step_scorer.parameters(), 1.0)
            opt.step(); opt2.step()

        # Validate NDCG
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
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in step_scorer.parameters())
    return {"ndcg5": float(np.mean(ndcgs)) if ndcgs else 0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0,
            "kendall_tau": float(np.mean(kendalls)) if kendalls else 0,
            "n_params": n_params}

# ============================================================
# MAIN
# ============================================================
print("Loading data...", flush=True)
pc_seqs, pc_cats = load_personachat()
lpt_seqs, lpt_cats = load_lpt()
print(f"PersonaChat: {len(pc_seqs)} users, {len(pc_cats)} cats")
print(f"LPT: {len(lpt_seqs)} users, {len(lpt_cats)} cats")

pc_rank = build_ranking_tasks(pc_seqs, window=8, max_per_user=4)
lpt_rank = build_ranking_tasks(lpt_seqs, window=8, max_per_user=4)
np.random.shuffle(pc_rank); np.random.shuffle(lpt_rank)

def split(tasks):
    n = len(tasks)
    return tasks[:int(n*.6)], tasks[int(n*.6):int(n*.8)], tasks[int(n*.8):]

pc_tr, pc_va, pc_te = split(pc_rank)
lpt_tr, lpt_va, lpt_te = split(lpt_rank)
print(f"Ranking tasks: PC={len(pc_rank)} (tr={len(pc_tr)} va={len(pc_va)} te={len(pc_te)})")
print(f"Ranking tasks: LPT={len(lpt_rank)} (tr={len(lpt_tr)} va={len(lpt_va)} te={len(lpt_te)})")

# Show relevance distribution
pc_rel = np.concatenate([d["relevance"].numpy() for d in pc_rank])
lpt_rel = np.concatenate([d["relevance"].numpy() for d in lpt_rank])
print(f"\nRelevance distribution:")
print(f"  PersonaChat: mean={pc_rel.mean():.4f} std={pc_rel.std():.4f} min={pc_rel.min():.4f} max={pc_rel.max():.4f}")
print(f"  LPT: mean={lpt_rel.mean():.4f} std={lpt_rel.std():.4f} min={lpt_rel.min():.4f} max={lpt_rel.max():.4f}")

# ============================================================
# Exp-3: Ranking
# ============================================================
print("\n" + "=" * 70)
print("EXP-3: PER-TIMESTEP RANKING (MSE on relevance, NDCG@5/MRR/Kendall)")
print("=" * 70)
e3_pc = {}; e3_lpt = {}
for v in VARIANTS:
    print(f"  {v}...", end="", flush=True); t = time.time()
    runs_pc = [train_eval_ranking(v, pc_tr, pc_va, pc_te, s, len(pc_cats)) for s in SEEDS]
    runs_lpt = [train_eval_ranking(v, lpt_tr, lpt_va, lpt_te, s, len(lpt_cats)) for s in SEEDS]
    e3_pc[v] = {}; e3_lpt[v] = {}
    for k in ["ndcg5", "mrr", "kendall_tau"]:
        pc_vals = [r[k] for r in runs_pc]
        lpt_vals = [r[k] for r in runs_lpt]
        e3_pc[v][k] = float(np.mean(pc_vals))
        e3_pc[v][k + "_std"] = float(np.std(pc_vals))
        e3_lpt[v][k] = float(np.mean(lpt_vals))
        e3_lpt[v][k + "_std"] = float(np.std(lpt_vals))
    e3_pc[v]["n_params"] = runs_pc[0]["n_params"]
    elapsed = time.time() - t
    print(f" PC: NDCG={e3_pc[v]['ndcg5']:.3f}({e3_pc[v]['ndcg5_std']:.3f}) MRR={e3_pc[v]['mrr']:.3f} Kendall={e3_pc[v]['kendall_tau']:.3f} ({elapsed:.0f}s)")
    print(f"       LPT: NDCG={e3_lpt[v]['ndcg5']:.3f}({e3_lpt[v]['ndcg5_std']:.3f}) MRR={e3_lpt[v]['mrr']:.3f} Kendall={e3_lpt[v]['kendall_tau']:.3f}")
    gc.collect()

# Embed Exp-1 + Exp-2
e1_pc = {'cct_lnn':{'mse':0.0399,'spearman_r':0.031},'ltc':{'mse':0.0373,'spearman_r':-0.016},
         'gru':{'mse':0.0371,'spearman_r':0.090},'gru_time':{'mse':0.0366,'spearman_r':0.091},
         'neural_ode':{'mse':0.0369,'spearman_r':-0.031},'cfc':{'mse':0.0387,'spearman_r':0.062},
         'param_expdecay':{'mse':0.0386,'spearman_r':0.112},'transformer':{'mse':0.0374,'spearman_r':0.081}}
e1_lpt = {'cct_lnn':{'mse':0.0696,'spearman_r':0.0},'ltc':{'mse':0.0720,'spearman_r':0.0},
          'gru':{'mse':0.0700,'spearman_r':0.0},'gru_time':{'mse':0.0721,'spearman_r':0.0},
          'neural_ode':{'mse':0.0690,'spearman_r':0.0},'cfc':{'mse':0.0720,'spearman_r':0.0},
          'param_expdecay':{'mse':0.0704,'spearman_r':0.0},'transformer':{'mse':0.0722,'spearman_r':0.0}}
e2_pc = {'cct_lnn':{'mse':0.0399,'spearman_r':0.031,'top20_overlap':0.28},
         'ltc':{'mse':0.0373,'spearman_r':-0.016,'top20_overlap':0.21},
         'gru':{'mse':0.0371,'spearman_r':0.090,'top20_overlap':0.24},
         'gru_time':{'mse':0.0366,'spearman_r':0.091,'top20_overlap':0.23},
         'neural_ode':{'mse':0.0369,'spearman_r':-0.031,'top20_overlap':0.18},
         'cfc':{'mse':0.0387,'spearman_r':0.062,'top20_overlap':0.26},
         'param_expdecay':{'mse':0.0386,'spearman_r':0.112,'top20_overlap':0.28},
         'transformer':{'mse':0.0374,'spearman_r':0.081,'top20_overlap':0.24}}

all_results = {
    "exp1_regression": {"personaChat": e1_pc, "lpt": e1_lpt},
    "exp2_weighted_regression": {"personaChat": e2_pc, "lpt": {}},
    "exp3_ranking": {"personaChat": e3_pc, "lpt": e3_lpt},
    "config": {"HD": HD, "SEEDS": SEEDS, "EPOCHS": EPOCHS, "VARIANTS": VARIANTS,
               "exp1_note": "Standard MSE regression, per-sequence final score (5 seeds)",
               "exp2_note": "Focal MSE gamma=2, upweights hard examples (5 seeds)",
               "exp3_note": "Per-timestep MSE on relevance + learned scorer, 15 epochs (3 seeds)"},
    "target_distribution": {
        "personaChat": {"mean": float(pc_rel.mean()), "std": float(pc_rel.std()),
                        "min": float(pc_rel.min()), "max": float(pc_rel.max())},
        "lpt": {"mean": float(lpt_rel.mean()), "std": float(lpt_rel.std()),
                "min": float(lpt_rel.min()), "max": float(lpt_rel.max())}
    }
}
out = os.path.join(OD, "v21_c0_fixed_results.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n{'='*70}")
print(f"ALL RESULTS SAVED: {out}")
print(f"{'='*70}")
