# -*- coding: utf-8 -*-
"""
v21 Quick Experiments - Streamlined for fast execution.
All files in experiment folder. 8 core models, 3 seeds, 10 epochs.
"""
import torch, torch.nn as nn, torch.optim as optim, json, numpy as np, os, sys, time
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_models import Retriever, INPUT_DIM

torch.manual_seed(42); np.random.seed(42)
OD = os.path.dirname(os.path.abspath(__file__))
HD = 16; SEEDS = [42, 123, 256]; EPOCHS = 10
VARIANTS = ['cct_lnn', 'ltc', 'gru', 'gru_time', 'neural_ode', 'cfc', 'param_expdecay', 'transformer']

def load_data():
    with open(os.path.join(OD, "personachat_enhanced.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    cat_set = set()
    for u in data:
        for m in u['memory_bank']: cat_set.add(m['category'])
    cat_list = sorted(cat_set)
    cat_map = {c:i for i,c in enumerate(cat_list)}
    seqs = []
    for u in data:
        mems = sorted(u['memory_bank'], key=lambda m: m['timestamp'])
        if len(mems) < 5: continue
        X = torch.tensor([m['query'] for m in mems], dtype=torch.float32)
        ts = torch.tensor([m['timestamp'] for m in mems], dtype=torch.float32)
        cats = torch.tensor([cat_map[m['category']] for m in mems], dtype=torch.long)
        seqs.append({'X': X, 'ts': ts, 'cats': cats})
    return seqs, cat_map, cat_list

def build_tasks(seqs, window=5, max_per_user=5):
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

def train_eval(variant, train, val, test, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n_cats = len(set(int(d['cats'][-1]) for d in train))
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

print("Loading data...", flush=True)
seqs, cat_map, cat_list = load_data()
tasks = build_tasks(seqs)
np.random.shuffle(tasks)
n=len(tasks); tr=tasks[:int(n*.6)]; va=tasks[int(n*.6):int(n*.8)]; te=tasks[int(n*.8):]
print(f"Data: {len(seqs)} users, {len(tasks)} tasks, {len(cat_list)} cats")
print(f"Train/Val/Test = {len(tr)}/{len(va)}/{len(te)}")

# ==============================
# E1+E2: Baseline comparison
# ==============================
print("\n=== E1+E2: Baseline Comparison ===", flush=True)
e1e2 = {}
for v in VARIANTS:
    print(f"  {v}...", end="", flush=True); t=time.time()
    runs=[]
    for s in SEEDS:
        r=train_eval(v,tr,va,te,s); runs.append(r)
        print(".",end="",flush=True)
    mses=[r['mse'] for r in runs]; f1s=[r['f1'] for r in runs]; accs=[r['accuracy'] for r in runs]
    e1e2[v]={'mse_mean':float(np.mean(mses)),'mse_std':float(np.std(mses)),
             'acc_mean':float(np.mean(accs)),'acc_std':float(np.std(accs)),
             'f1_mean':float(np.mean(f1s)),'f1_std':float(np.std(f1s)),
             'prec_mean':float(np.mean([r['precision'] for r in runs])),
             'rec_mean':float(np.mean([r['recall'] for r in runs])),
             'n_params':runs[0]['n_params']}
    elapsed=time.time()-t
    print(f" MSE={e1e2[v]['mse_mean']:.4f}+/-{e1e2[v]['mse_std']:.4f} F1={e1e2[v]['f1_mean']:.3f} P={e1e2[v]['n_params']} ({elapsed:.0f}s)")

# ==============================
# E3: Ablation
# ==============================
print("\n=== E3: Ablation ===", flush=True)
e3={}
for name,v in [('A1_fixed_tau','ltc_fixed'),('A2_input_dep_tau','ltc'),('A3_cct_full','cct_lnn')]:
    runs=[train_eval(v,tr,va,te,s) for s in SEEDS]
    mses=[r['mse'] for r in runs]; f1s=[r['f1'] for r in runs]
    e3[name]={'mse_mean':float(np.mean(mses)),'mse_std':float(np.std(mses)),
              'f1_mean':float(np.mean(f1s)),'n_params':runs[0]['n_params']}
    print(f"  {name}: MSE={e3[name]['mse_mean']:.4f}+/-{e3[name]['mse_std']:.4f} F1={e3[name]['f1_mean']:.3f}")

# ==============================
# E4: Tau analysis
# ==============================
print("\n=== E4: Tau Analysis ===", flush=True)
torch.manual_seed(42); np.random.seed(42)
model=Retriever(INPUT_DIM,HD,'cct_lnn',len(cat_list))
opt=optim.Adam(model.parameters(),lr=0.002); crit=nn.BCELoss()
for ep in range(15):
    model.train()
    for d in np.random.permutation(len(tr)):
        opt.zero_grad(); s,_,_=model(tr[d]['X'],tr[d]['ts'],tr[d]['cats'])
        crit(s.squeeze(),tr[d]['target']).backward(); opt.step()

model.eval(); cat_tau={c:[] for c in cat_list}
with torch.no_grad():
    for d in tasks[:200]:
        _,_,tl=model(d['X'],d['ts'],d['cats'])
        if tl is not None:
            for i,ci in enumerate(d['cats']): cat_tau[cat_list[int(ci)]].append(tl[i].mean().item())

cat_freq={}
for s in seqs:
    for ci in s['cats']:
        cat_freq[cat_list[int(ci)]]=cat_freq.get(cat_list[int(ci)],0)+1

ts2={c:{'mean':float(np.mean(v)),'std':float(np.std(v)),'n':len(v)} for c,v in cat_tau.items() if len(v)>3}
cats_s=sorted(ts2.keys(),key=lambda c:cat_freq.get(c,0))
taus=[ts2[c]['mean'] for c in cats_s]; freqs=[cat_freq.get(c,1) for c in cats_s]
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
for c in cats_s: print(f"    {c}: tau={ts2[c]['mean']:.3f}+/-{ts2[c]['std']:.3f} (n={ts2[c]['n']})")

# ==============================
# E7: Temporal patterns
# ==============================
print("\n=== E7: Temporal Patterns ===", flush=True)
bursty=[];regular=[];long_gap=[]
for s in seqs:
    ts=s['ts']
    if len(ts)<5: continue
    gaps=[float(ts[i]-ts[i-1]) for i in range(1,len(ts))]
    if len(gaps)<2: continue
    cv=np.std(gaps)/max(np.mean(gaps),1e-8)
    if cv>1.5: bursty.append(s)
    elif cv<0.5: regular.append(s)
    if max(gaps)>200: long_gap.append(s)

print(f"  Bursty:{len(bursty)} Regular:{len(regular)} LongGap:{len(long_gap)}")
e7={}
for sname,seqs_sub in [('bursty',bursty),('regular',regular),('long_gap',long_gap)]:
    if len(seqs_sub)<8: print(f"  Skip {sname} (too few)"); continue
    td=build_tasks(seqs_sub,window=5,max_per_user=3)
    if len(td)<8: continue
    np.random.shuffle(td); nn2=len(td)
    tr2=td[:int(nn2*.6)];va2=td[int(nn2*.6):int(nn2*.8)];te2=td[int(nn2*.8):]
    e7[sname]={}
    for v in ['ltc','cct_lnn','gru','gru_time']:
        runs=[train_eval(v,tr2,va2,te2,s) for s in SEEDS]
        e7[sname][v]={'mse_mean':float(np.mean([r['mse'] for r in runs])),
                      'f1_mean':float(np.mean([r['f1'] for r in runs]))}
    print(f"  {sname}: " + " | ".join(f"{v}={e7[sname][v]['mse_mean']:.4f}" for v in e7[sname]))

# ==============================
# Save all results
# ==============================
all_results = {
    'e1e2_baselines': e1e2,
    'e3_ablation': e3,
    'e4_tau': e4,
    'e7_temporal': e7,
    'categories': cat_list,
    'config': {'HD': HD, 'SEEDS': SEEDS, 'EPOCHS': EPOCHS, 'INPUT_DIM': INPUT_DIM}
}
out = os.path.join(OD, "v21_experimental_results.json")
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Results saved: {out}")
print(f"{'='*60}")
print(f"\n{'Model':<20} {'MSE':<16} {'F1':<8} {'Acc':<8} {'Params'}")
print("-"*60)
for v in VARIANTS:
    if v in e1e2:
        r=e1e2[v]
        print(f"{v:<20} {r['mse_mean']:.4f}+/-{r['mse_std']:.4f} {r['f1_mean']:.3f}  {r['acc_mean']:.3f}  {r['n_params']}")

# CCT vs LTC comparison
if 'cct_lnn' in e1e2 and 'ltc' in e1e2:
    cct=e1e2['cct_lnn']; ltc=e1e2['ltc']
    improvement=(ltc['mse_mean']-cct['mse_mean'])/ltc['mse_mean']*100
    print(f"\nCCT-LNN vs LTC: MSE improvement = {improvement:+.1f}%")
