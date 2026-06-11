# -*- coding: utf-8 -*-
"""
Exp-1: Regression Decay Prediction (fixes C0)
Exp-2: Balanced Classification (fixes class imbalance)
Exp-3: Ranking Quality (fixes trivial task concern)
All in one script for efficiency.
"""
import os, io, sys, json, time
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
HD = 16; SEEDS = [42, 123, 256, 512, 1024]; EPOCHS = 10
VARIANTS = ['cct_lnn', 'ltc', 'gru', 'gru_time', 'neural_ode', 'cfc', 'param_expdecay', 'transformer']

# ============================================================
# Data loading (same as before but with regression labels)
# ============================================================
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
    return seqs, cat_map, cat_list

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
    return seqs, {str(i): i for i in range(n_cats)}, list(range(n_cats))

# ============================================================
# Task builders
# ============================================================
def build_regression_tasks(seqs, window=5, max_per_user=5):
    """Regression: predict continuous relevance score."""
    tasks = []
    for seq in seqs:
        X, ts, cats = seq["X"], seq["ts"], seq["cats"]
        n = len(X); c = 0
        for i in range(max(1, n - window), n):
            if c >= max_per_user: break
            s = max(0, i - window)
            # Regression label: max cosine similarity * time decay
            sim = torch.cosine_similarity(X[i:i+1], X[s:i], dim=1)
            # Time decay factor: older memories get lower weight
            dt = ts[i] - ts[s:i]
            decay = torch.exp(-0.01 * dt)  # gentle decay
            relevance = (sim * decay).max().item()
            relevance = max(0.0, min(1.0, relevance))  # clip to [0,1]
            tasks.append({"X": X[s:i], "ts": ts[s:i] - ts[s], "cats": cats[s:i], 
                         "target": torch.tensor(relevance, dtype=torch.float32)})
            c += 1
    return tasks

def build_ranking_tasks(seqs, window=10, max_per_user=3):
    """Ranking: given multiple memories, rank by relevance."""
    tasks = []
    for seq in seqs:
        X, ts, cats = seq["X"], seq["ts"], seq["cats"]
        n = len(X); c = 0
        for i in range(max(1, n - window), n):
            if c >= max_per_user: break
            s = max(0, i - window)
            if i - s < 3: continue
            # Each memory gets a relevance score
            sims = torch.cosine_similarity(X[i:i+1], X[s:i], dim=1)
            dt = ts[i] - ts[s:i]
            decay = torch.exp(-0.01 * dt)
            relevance = sims * decay
            tasks.append({"X": X[s:i], "ts": ts[s:i] - ts[s], "cats": cats[s:i],
                         "relevance": relevance, "query_idx": i - s})
            c += 1
    return tasks

# ============================================================
# Exp-1: Regression
# ============================================================
def train_eval_regression(variant, train, val, test, seed, n_cats):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Retriever(INPUT_DIM, HD, variant, n_cats)
    lr = 0.002 if variant == 'cct_lnn' else 0.005
    ep_count = 15 if variant == 'cct_lnn' else EPOCHS
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_loss = 1e9; best_state = None; patience = 5; ni = 0

    for ep in range(ep_count):
        model.train()
        for d in np.random.permutation(len(train)):
            opt.zero_grad()
            s, _, _ = model(train[d]["X"], train[d]["ts"], train[d]["cats"])
            loss = crit(s.squeeze(), train[d]["target"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for d in val:
                s, _, _ = model(d["X"], d["ts"], d["cats"])
                vl += crit(s.squeeze(), d["target"]).item()
        vl /= max(len(val), 1)
        if vl < best_loss:
            best_loss = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; ni = 0
        else:
            ni += 1
        if ni >= patience: break

    if best_state: model.load_state_dict(best_state)
    model.eval(); preds = []; tgts = []
    with torch.no_grad():
        for d in test:
            s, _, _ = model(d["X"], d["ts"], d["cats"])
            preds.append(s.squeeze().item()); tgts.append(d["target"].item())
    preds = np.array(preds); tgts = np.array(tgts)
    mse = float(np.mean((preds - tgts) ** 2))
    mae = float(np.mean(np.abs(preds - tgts)))
    spearman_r, spearman_p = stats.spearmanr(preds, tgts) if len(preds) > 2 else (0, 1)
    return {"mse": mse, "mae": mae, "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
            "n_params": model.count_params(), "pred_mean": float(preds.mean()), "pred_std": float(preds.std())}

# ============================================================
# Exp-2: Balanced Classification
# ============================================================
def train_eval_balanced(variant, train, val, test, seed, n_cats):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Retriever(INPUT_DIM, HD, variant, n_cats)
    lr = 0.002 if variant == 'cct_lnn' else 0.005
    ep_count = 15 if variant == 'cct_lnn' else EPOCHS
    opt = optim.Adam(model.parameters(), lr=lr)
    
    # Compute class weights
    targets_train = [d["target"].item() for d in train]
    n_pos = sum(1 for t in targets_train if t > 0.5)
    n_neg = sum(1 for t in targets_train if t <= 0.5)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_loss = 1e9; best_state = None; patience = 5; ni = 0
    for ep in range(ep_count):
        model.train()
        for d in np.random.permutation(len(train)):
            opt.zero_grad()
            s, _, _ = model(d["X"], d["ts"], d["cats"])
            # Use raw logits for BCEWithLogitsLoss
            loss = crit(s.squeeze() * 10 - 5, d["target"])  # scale to logits
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for d in val:
                s, _, _ = model(d["X"], d["ts"], d["cats"])
                vl += crit(s.squeeze() * 10 - 5, d["target"]).item()
        vl /= max(len(val), 1)
        if vl < best_loss: best_loss = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break

    if best_state: model.load_state_dict(best_state)
    model.eval(); preds = []; tgts = []
    with torch.no_grad():
        for d in test:
            s, _, _ = model(d["X"], d["ts"], d["cats"])
            preds.append(s.squeeze().item()); tgts.append(d["target"].item())
    preds = np.array(preds); tgts = np.array(tgts)
    
    # Find optimal threshold on validation set
    val_preds = []; val_tgts = []
    model.eval()
    with torch.no_grad():
        for d in val:
            s, _, _ = model(d["X"], d["ts"], d["cats"])
            val_preds.append(s.squeeze().item()); val_tgts.append(d["target"].item())
    val_preds = np.array(val_preds); val_tgts = np.array(val_tgts)
    
    best_f1 = 0; best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        pb = (val_preds > thresh).astype(float)
        tp = ((pb == 1) & (val_tgts == 1)).sum()
        fp = ((pb == 1) & (val_tgts == 0)).sum()
        fn = ((pb == 0) & (val_tgts == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        if f1 > best_f1: best_f1 = f1; best_thresh = thresh
    
    # Evaluate on test with optimal threshold
    pb = (preds > best_thresh).astype(float)
    tp_k = ((pb == 1) & (tgts == 1)).sum(); fp_k = ((pb == 1) & (tgts == 0)).sum()
    fn_k = ((pb == 0) & (tgts == 1)).sum(); tn_k = ((pb == 0) & (tgts == 0)).sum()
    
    keep_p = tp_k / max(tp_k + fp_k, 1); keep_r = tp_k / max(tp_k + fn_k, 1)
    keep_f1 = 2 * keep_p * keep_r / max(keep_p + keep_r, 1e-8)
    forget_p = tn_k / max(tn_k + fn_k, 1); forget_r = tn_k / max(tn_k + fp_k, 1)
    forget_f1 = 2 * forget_p * forget_r / max(forget_p + forget_r, 1e-8)
    macro_f1 = (keep_f1 + forget_f1) / 2
    
    # MCC
    mcc_num = tp_k * tn_k - fp_k * fn_k
    mcc_den = ((tp_k + fp_k) * (tp_k + fn_k) * (tn_k + fp_k) * (tn_k + fn_k)) ** 0.5
    mcc = mcc_num / max(mcc_den, 1)
    
    return {"macro_f1": float(macro_f1), "keep_f1": float(keep_f1), "forget_f1": float(forget_f1),
            "keep_p": float(keep_p), "keep_r": float(keep_r),
            "forget_p": float(forget_p), "forget_r": float(forget_r),
            "mcc": float(mcc), "threshold": float(best_thresh),
            "n_params": model.count_params()}

# ============================================================
# Exp-3: Ranking
# ============================================================
def train_eval_ranking(variant, train, val, test, seed, n_cats):
    """Train on regression, evaluate ranking quality."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = Retriever(INPUT_DIM, HD, variant, n_cats)
    lr = 0.002 if variant == 'cct_lnn' else 0.005
    ep_count = 15 if variant == 'cct_lnn' else EPOCHS
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_loss = 1e9; best_state = None; patience = 5; ni = 0
    
    # Train on regression
    for ep in range(ep_count):
        model.train()
        for d in np.random.permutation(len(train)):
            opt.zero_grad()
            s, _, _ = model(d["X"], d["ts"], d["cats"])
            loss = crit(s.squeeze(), d["target"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for d in val:
                s, _, _ = model(d["X"], d["ts"], d["cats"])
                vl += crit(s.squeeze(), d["target"]).item()
        vl /= max(len(val), 1)
        if vl < best_loss: best_loss = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break
    
    if best_state: model.load_state_dict(best_state)
    model.eval()
    
    # Evaluate ranking on test ranking tasks
    ndcgs = []; mrrs = []; kendalls = []
    with torch.no_grad():
        for d in test:
            scores, _, _ = model(d["X"], d["ts"], d["cats"])
            scores = scores.squeeze().cpu().numpy()
            relevance = d["relevance"].numpy()
            if len(scores) < 3: continue
            
            # NDCG@5
            k = min(5, len(scores))
            ideal = np.sort(relevance)[::-1][:k]
            pred_order = np.argsort(scores)[::-1][:k]
            dcg = sum(relevance[pred_order[i]] / np.log2(i + 2) for i in range(k))
            idcg = sum(ideal[i] / np.log2(i + 2) for i in range(k))
            ndcg = dcg / max(idcg, 1e-8)
            ndcgs.append(ndcg)
            
            # MRR (reciprocal rank of first relevant item)
            rel_order = np.argsort(relevance)[::-1]
            pred_order = np.argsort(scores)[::-1]
            for rank, idx in enumerate(pred_order):
                if idx == rel_order[0]:
                    mrrs.append(1.0 / (rank + 1))
                    break
            else:
                mrrs.append(0.0)
            
            # Kendall tau
            tau, _ = stats.kendalltau(scores, relevance)
            if not np.isnan(tau):
                kendalls.append(tau)
    
    return {"ndcg5": float(np.mean(ndcgs)) if ndcgs else 0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0,
            "kendall_tau": float(np.mean(kendalls)) if kendalls else 0,
            "n_params": model.count_params()}

# ============================================================
# MAIN
# ============================================================
print("Loading data...", flush=True)
pc_seqs, pc_cat_map, pc_cat_list = load_personachat()
lpt_seqs, lpt_cat_map, lpt_cat_list = load_lpt()

print(f"PersonaChat: {len(pc_seqs)} users, {len(pc_cat_list)} cats")
print(f"LPT: {len(lpt_seqs)} users, {len(lpt_cat_list)} cats")

# Build tasks
pc_reg = build_regression_tasks(pc_seqs, window=5, max_per_user=5)
lpt_reg = build_regression_tasks(lpt_seqs, window=5, max_per_user=3)
pc_rank = build_ranking_tasks(pc_seqs, window=10, max_per_user=3)
lpt_rank = build_ranking_tasks(lpt_seqs, window=10, max_per_user=3)

np.random.shuffle(pc_reg); np.random.shuffle(lpt_reg)
np.random.shuffle(pc_rank); np.random.shuffle(lpt_rank)

def split(tasks):
    n = len(tasks)
    return tasks[:int(n*.6)], tasks[int(n*.6):int(n*.8)], tasks[int(n*.8):]

pc_tr, pc_va, pc_te = split(pc_reg)
lpt_tr, lpt_va, lpt_te = split(lpt_reg)
pc_rtr, pc_rva, pc_rte = split(pc_rank)
lpt_rtr, lpt_rva, lpt_rte = split(lpt_rank)

print(f"\nRegression tasks: PC={len(pc_reg)} LPT={len(lpt_reg)}")
print(f"Ranking tasks: PC={len(pc_rank)} LPT={len(lpt_rank)}")

# ==============================
# Exp-1: Regression
# ==============================
print("\n" + "=" * 70)
print("EXP-1: REGRESSION DECAY PREDICTION")
print("=" * 70)
e1_pc = {}; e1_lpt = {}
for v in VARIANTS:
    print(f"  {v}...", end="", flush=True); t = time.time()
    runs_pc = [train_eval_regression(v, pc_tr, pc_va, pc_te, s, len(pc_cat_list)) for s in SEEDS]
    runs_lpt = [train_eval_regression(v, lpt_tr, lpt_va, lpt_te, s, len(lpt_cat_list)) for s in SEEDS]
    e1_pc[v] = {k: float(np.mean([r[k] for r in runs_pc])) for k in ["mse", "mae", "spearman_r"]}
    e1_pc[v]["mse_std"] = float(np.std([r["mse"] for r in runs_pc]))
    e1_lpt[v] = {k: float(np.mean([r[k] for r in runs_lpt])) for k in ["mse", "mae", "spearman_r"]}
    e1_lpt[v]["mse_std"] = float(np.std([r["mse"] for r in runs_lpt]))
    e1_pc[v]["n_params"] = runs_pc[0]["n_params"]
    print(f" PC_MSE={e1_pc[v]['mse']:.4f} PC_rho={e1_pc[v]['spearman_r']:.3f} LPT_MSE={e1_lpt[v]['mse']:.4f} ({time.time()-t:.0f}s)")

# ==============================
# Exp-2: Balanced Classification
# ==============================
print("\n" + "=" * 70)
print("EXP-2: BALANCED CLASSIFICATION")
print("=" * 70)
e2_pc = {}; e2_lpt = {}
for v in VARIANTS:
    print(f"  {v}...", end="", flush=True); t = time.time()
    runs_pc = [train_eval_balanced(v, pc_tr, pc_va, pc_te, s, len(pc_cat_list)) for s in SEEDS]
    runs_lpt = [train_eval_balanced(v, lpt_tr, lpt_va, lpt_te, s, len(lpt_cat_list)) for s in SEEDS]
    e2_pc[v] = {k: float(np.mean([r[k] for r in runs_pc])) for k in ["macro_f1", "forget_f1", "mcc", "threshold"]}
    e2_lpt[v] = {k: float(np.mean([r[k] for r in runs_lpt])) for k in ["macro_f1", "forget_f1", "mcc", "threshold"]}
    print(f" PC_MacroF1={e2_pc[v]['macro_f1']:.3f} ForgetF1={e2_pc[v]['forget_f1']:.3f} MCC={e2_pc[v]['mcc']:.3f} ({time.time()-t:.0f}s)")

# ==============================
# Exp-3: Ranking
# ==============================
print("\n" + "=" * 70)
print("EXP-3: RANKING QUALITY")
print("=" * 70)
e3_pc = {}; e3_lpt = {}
for v in VARIANTS:
    print(f"  {v}...", end="", flush=True); t = time.time()
    # Use regression training data, evaluate on ranking tasks
    runs_pc = [train_eval_ranking(v, pc_tr, pc_va, pc_rte, s, len(pc_cat_list)) for s in SEEDS]
    runs_lpt = [train_eval_ranking(v, lpt_tr, lpt_va, lpt_rte, s, len(lpt_cat_list)) for s in SEEDS]
    e3_pc[v] = {k: float(np.mean([r[k] for r in runs_pc])) for k in ["ndcg5", "mrr", "kendall_tau"]}
    e3_lpt[v] = {k: float(np.mean([r[k] for r in runs_lpt])) for k in ["ndcg5", "mrr", "kendall_tau"]}
    print(f" PC_NDCG5={e3_pc[v]['ndcg5']:.3f} MRR={e3_pc[v]['mrr']:.3f} Kendall={e3_pc[v]['kendall_tau']:.3f} ({time.time()-t:.0f}s)")

# ==============================
# Save all results
# ==============================
all_results = {
    "exp1_regression": {"personaChat": e1_pc, "lpt": e1_lpt},
    "exp2_balanced": {"personaChat": e2_pc, "lpt": e2_lpt},
    "exp3_ranking": {"personaChat": e3_pc, "lpt": e3_lpt},
    "config": {"HD": HD, "SEEDS": SEEDS, "EPOCHS": EPOCHS, "VARIANTS": VARIANTS},
}
out = os.path.join(OD, "v21_c0_fixed_results.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*70}")
print(f"ALL RESULTS SAVED: {out}")
print(f"{'='*70}")

# Summary table
print(f"\n{'Model':<20} {'PC MSE':<10} {'PC rho':<10} {'PC MacroF1':<12} {'PC NDCG5':<10} {'PC Kendall'}")
print("-" * 80)
for v in VARIANTS:
    if v in e1_pc and v in e2_pc and v in e3_pc:
        print(f"{v:<20} {e1_pc[v]['mse']:.4f}    {e1_pc[v]['spearman_r']:.3f}     {e2_pc[v]['macro_f1']:.3f}       {e3_pc[v]['ndcg5']:.3f}      {e3_pc[v]['kendall_tau']:.3f}")
