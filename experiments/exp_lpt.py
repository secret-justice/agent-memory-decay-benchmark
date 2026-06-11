# -*- coding: utf-8 -*-
"""
v21 LPT Experiments: Long-term temporal dynamics.
LPT = 200 users, ~189 entries/user, 63-day span, 5 categories.
"""
import torch, torch.nn as nn, torch.optim as optim, json, numpy as np, os, sys, time
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_models import Retriever, INPUT_DIM

torch.manual_seed(42); np.random.seed(42)
OD = os.path.dirname(os.path.abspath(__file__))
HD = 16; SEEDS = [42, 123]; EPOCHS = 8
VARIANTS = ['cct_lnn', 'ltc', 'gru', 'gru_time', 'neural_ode', 'cfc', 'param_expdecay', 'transformer']

def load_lpt():
    with open(os.path.join(OD, "lpt_200users.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    n_cats = 0
    for u in data:
        for m in u['memory_bank']:
            c = m.get('category', 0)
            if isinstance(c, int): n_cats = max(n_cats, c+1)
            else: n_cats = max(n_cats, 1)
    
    seqs = []
    for u in data:
        mems = sorted(u['memory_bank'], key=lambda m: m['timestamp'])
        if len(mems) < 10: continue
        X = torch.tensor([m['query'] for m in mems], dtype=torch.float32)
        ts = torch.tensor([m['timestamp'] for m in mems], dtype=torch.float32)
        cats_list = []
        for m in mems:
            c = m.get('category', 0)
            cats_list.append(c if isinstance(c, int) else 0)
        cats = torch.tensor(cats_list, dtype=torch.long)
        seqs.append({'X': X, 'ts': ts, 'cats': cats})
    return seqs, n_cats

def build_tasks(seqs, window=5, max_per_user=3):
    tasks = []
    for seq in seqs:
        X=seq['X'];ts=seq['ts'];cats=seq['cats'];n=len(X);c=0
        for i in range(max(1,n-window),n):
            if c>=max_per_user: break
            s=max(0,i-window)
            sim=torch.cosine_similarity(X[i:i+1],X[s:i],dim=1)
            kept=(sim.max()>0.3).float()
            tasks.append({'X':X[s:i],'ts':ts[s:i]-ts[s],'cats':cats[s:i],'target':kept})
            c+=1
    return tasks

def build_temporal_tasks(seqs, window=5):
    """Build tasks categorized by temporal pattern."""
    bursty=[];regular=[];long_gap=[]
    for seq in seqs:
        ts=seq['ts']
        if len(ts)<10: continue
        gaps=[float(ts[i]-ts[i-1]) for i in range(1,len(ts))]
        if len(gaps)<3: continue
        cv=np.std(gaps)/max(np.mean(gaps),1e-8)
        max_gap=max(gaps)
        # Build tasks for this user
        X=seq['X'];cats=seq['cats'];n=len(X)
        user_tasks=[]
        for i in range(max(1,n-window),n):
            s=max(0,i-window)
            sim=torch.cosine_similarity(X[i:i+1],X[s:i],dim=1)
            kept=(sim.max()>0.3).float()
            user_tasks.append({'X':X[s:i],'ts':ts[s:i]-ts[s],'cats':cats[s:i],'target':kept})
        
        if cv>1.0 and len(user_tasks)>=3: bursty.extend(user_tasks)
        elif cv<0.5 and len(user_tasks)>=3: regular.extend(user_tasks)
        if max_gap>10 and len(user_tasks)>=3: long_gap.extend(user_tasks)
    
    return bursty, regular, long_gap

def train_eval(variant, train, val, test, seed, n_cats):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Retriever(INPUT_DIM, HD, variant, n_cats)
    ep_count = 15 if variant == 'cct_lnn' else EPOCHS
    lr = 0.002 if variant == 'cct_lnn' else 0.005
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    best_loss=1e9; best_state=None; patience=5; ni=0
    
    for ep in range(ep_count):
        model.train()
        for d in np.random.permutation(len(train)):
            opt.zero_grad()
            s,_,_ = model(train[d]['X'],train[d]['ts'],train[d]['cats'])
            loss = crit(s.squeeze(), train[d]['target'])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); vl=0
        with torch.no_grad():
            for d in val:
                s,_,_=model(d['X'],d['ts'],d['cats'])
                vl+=crit(s.squeeze(),d['target']).item()
        vl/=max(len(val),1)
        if vl<best_loss: best_loss=vl; best_state={k:v.clone() for k,v in model.state_dict().items()}; ni=0
        else: ni+=1
        if ni>=patience: break
    
    if best_state: model.load_state_dict(best_state)
    model.eval(); preds=[]; tgts=[]
    with torch.no_grad():
        for d in test:
            s,_,_=model(d['X'],d['ts'],d['cats'])
            preds.append(s.squeeze().item()); tgts.append(d['target'].item())
    preds=np.array(preds); tgts=np.array(tgts)
    mse=float(np.mean((preds-tgts)**2))
    pb=(preds>0.5).astype(float)
    tp=float(((pb==1)&(tgts==1)).sum()); fp=float(((pb==1)&(tgts==0)).sum())
    fn=float(((pb==0)&(tgts==1)).sum()); tn=float(((pb==0)&(tgts==0)).sum())
    acc=(tp+tn)/max(tp+fp+fn+tn,1); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    f1=2*prec*rec/max(prec+rec,1e-8)
    return {'mse':mse,'accuracy':acc,'f1':f1,'precision':prec,'recall':rec,'n_params':model.count_params()}

print("Loading LPT data...", flush=True)
seqs, n_cats = load_lpt()
print(f"LPT: {len(seqs)} users, {n_cats} categories")

tasks = build_tasks(seqs, window=5, max_per_user=3)
np.random.shuffle(tasks)
n=len(tasks); tr=tasks[:int(n*.6)]; va=tasks[int(n*.6):int(n*.8)]; te=tasks[int(n*.8):]
print(f"Tasks: {len(tasks)} (train/val/test={len(tr)}/{len(va)}/{len(te)})")

# E1+E2: Baselines
print("\n=== E1+E2: Baselines on LPT ===", flush=True)
e1e2={}
for v in VARIANTS:
    print(f"  {v}...", end="", flush=True); t=time.time()
    runs=[]
    for s in SEEDS:
        r=train_eval(v,tr,va,te,s,n_cats); runs.append(r)
        print(".",end="",flush=True)
    mses=[r['mse'] for r in runs]; f1s=[r['f1'] for r in runs]; accs=[r['accuracy'] for r in runs]
    e1e2[v]={'mse_mean':float(np.mean(mses)),'mse_std':float(np.std(mses)),
             'acc_mean':float(np.mean(accs)),'f1_mean':float(np.mean(f1s)),
             'f1_std':float(np.std(f1s)),'n_params':runs[0]['n_params']}
    print(f" MSE={e1e2[v]['mse_mean']:.4f} F1={e1e2[v]['f1_mean']:.3f} ({time.time()-t:.0f}s)")

# E3: Ablation
print("\n=== E3: Ablation on LPT ===", flush=True)
e3={}
for name,v in [('A1_fixed_tau','ltc_fixed'),('A2_input_dep_tau','ltc'),('A3_cct_full','cct_lnn')]:
    runs=[train_eval(v,tr,va,te,s,n_cats) for s in SEEDS]
    e3[name]={'mse_mean':float(np.mean([r['mse'] for r in runs])),
              'mse_std':float(np.std([r['mse'] for r in runs])),
              'f1_mean':float(np.mean([r['f1'] for r in runs]))}
    print(f"  {name}: MSE={e3[name]['mse_mean']:.4f}+/-{e3[name]['mse_std']:.4f}")

# E4: Tau
print("\n=== E4: Tau on LPT ===", flush=True)
torch.manual_seed(42); np.random.seed(42)
model=Retriever(INPUT_DIM,HD,'cct_lnn',n_cats)
opt=optim.Adam(model.parameters(),lr=0.002); crit=nn.BCELoss()
for ep in range(15):
    model.train()
    for d in np.random.permutation(len(tr)):
        opt.zero_grad(); s,_,_=model(tr[d]['X'],tr[d]['ts'],tr[d]['cats'])
        crit(s.squeeze(),tr[d]['target']).backward(); opt.step()

model.eval(); cat_tau={i:[] for i in range(n_cats)}
with torch.no_grad():
    for d in tasks[:150]:
        _,_,tl=model(d['X'],d['ts'],d['cats'])
        if tl is not None:
            for i,ci in enumerate(d['cats']): cat_tau[int(ci)].append(tl[i].mean().item())

ts2={c:{'mean':float(np.mean(v)),'std':float(np.std(v)),'n':len(v)} for c,v in cat_tau.items() if len(v)>3}
cats_s=sorted(ts2.keys())
taus=[ts2[c]['mean'] for c in cats_s]
freqs=[sum(1 for s in seqs for m in s['cats'] if int(m)==c) for c in cats_s]
if len(cats_s)>=3:
    r,p=stats.pearsonr(taus,freqs); rs,ps=stats.spearmanr(taus,freqs)
    z=np.arctanh(r);se=1/np.sqrt(len(cats_s)-3)
    rlo,rhi=np.tanh(z-1.96*se),np.tanh(z+1.96*se)
    tost=(rlo>-0.3)and(rhi<0.3)
else: r=p=rs=ps=rlo=rhi=float('nan');tost=False

e4={'n_cats':len(cats_s),'pearson_r':float(r),'pearson_p':float(p),
    'spearman_r':float(rs),'spearman_p':float(ps),
    'r_ci95':[float(rlo),float(rhi)],'tost_pass':bool(tost),'tau_stats':ts2}
print(f"  n={len(cats_s)}, r={r:.4f} p={p:.4f}, TOST={tost}")
for c in cats_s: print(f"    cat{c}: tau={ts2[c]['mean']:.3f}+/-{ts2[c]['std']:.3f} (n={ts2[c]['n']})")

# E7: Temporal patterns
print("\n=== E7: Temporal on LPT ===", flush=True)
bursty,regular,long_gap = build_temporal_tasks(seqs, window=5)
print(f"  Bursty:{len(bursty)} Regular:{len(regular)} LongGap:{len(long_gap)}")
e7={}
for sname,td in [('bursty',bursty),('regular',regular),('long_gap',long_gap)]:
    if len(td)<12: print(f"  Skip {sname}"); continue
    np.random.shuffle(td); nn2=len(td)
    tr2=td[:int(nn2*.6)];va2=td[int(nn2*.6):int(nn2*.8)];te2=td[int(nn2*.8):]
    e7[sname]={}
    for v in ['ltc','cct_lnn','gru','gru_time']:
        runs=[train_eval(v,tr2,va2,te2,s,n_cats) for s in SEEDS]
        e7[sname][v]={'mse_mean':float(np.mean([r['mse'] for r in runs])),
                      'f1_mean':float(np.mean([r['f1'] for r in runs]))}
    print(f"  {sname}: " + " | ".join(f"{v}={e7[sname][v]['mse_mean']:.4f}" for v in e7[sname]))

# Save
all_results = {
    'dataset': 'LPT',
    'e1e2_baselines': e1e2,
    'e3_ablation': e3,
    'e4_tau': e4,
    'e7_temporal': e7,
    'n_cats': n_cats,
    'config': {'HD': HD, 'SEEDS': SEEDS, 'EPOCHS': EPOCHS}
}
out = os.path.join(OD, "v21_lpt_results.json")
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Saved: {out}")
print(f"{'='*60}")
print(f"\n{'Model':<20} {'MSE':<16} {'F1':<8} {'Params'}")
print("-"*55)
for v in VARIANTS:
    if v in e1e2: r=e1e2[v]; print(f"{v:<20} {r['mse_mean']:.4f}+/-{r['mse_std']:.4f} {r['f1_mean']:.3f}  {r['n_params']}")

if 'cct_lnn' in e1e2 and 'ltc' in e1e2:
    imp=(e1e2['ltc']['mse_mean']-e1e2['cct_lnn']['mse_mean'])/e1e2['ltc']['mse_mean']*100
    print(f"\nCCT-LNN vs LTC: {imp:+.1f}% MSE {'improvement' if imp>0 else 'regression'}")
if 'cct_lnn' in e1e2 and 'gru' in e1e2:
    imp=(e1e2['gru']['mse_mean']-e1e2['cct_lnn']['mse_mean'])/e1e2['gru']['mse_mean']*100
    print(f"CCT-LNN vs GRU: {imp:+.1f}% MSE {'improvement' if imp>0 else 'regression'}")

