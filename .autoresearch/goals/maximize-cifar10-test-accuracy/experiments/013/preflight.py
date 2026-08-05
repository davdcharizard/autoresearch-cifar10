import importlib, os, statistics, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
import prepare

class Stub:
    def __init__(self): pass
    def evaluate(self, *args): raise AssertionError("real evaluation forbidden")
prepare.Eval = Stub
sys.modules.pop("train", None)
t = importlib.import_module("train")
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA H20"
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

torch.manual_seed(1)
m = t.WideResNet(2, 2, 10).to(dev)
keys = tuple(m.state_dict())
ids = {k: id(v) for k, v in m.state_dict(keep_vars=True).items()}
ema = t.ModelEMA(m, .999)
assert tuple(m.state_dict()) == keys and ema.num_updates == 1
assert all(v.dtype == torch.float32 for v in ema.shadow.values() if v.is_floating_point())
assert all(not v.requires_grad for v in ema.shadow.values())
before = {k: v.clone() for k, v in m.state_dict().items()}
class Good:
    def evaluate(self, model, device): return 1.0, 2.0
assert ema.evaluate(m, Good(), dev) == (1.0, 2.0)
assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
assert all(ids[k] == id(v) for k, v in m.state_dict(keep_vars=True).items())
bad_key = keys[1]
saved = ema.shadow[bad_key]
ema.shadow[bad_key] = torch.zeros(2, 3, device=dev)
try:
    ema.evaluate(m, Good(), dev)
except RuntimeError:
    pass
else:
    raise AssertionError("partial swap failure not injected")
ema.shadow[bad_key] = saved
assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())

torch.manual_seed(2); accepted = t.WideResNet(2, 2, 10).to(dev)
torch.manual_seed(2); candidate = t.WideResNet(2, 2, 10).to(dev)
models = {"accepted": accepted, "candidate": candidate}
def opt(model):
    d=[p for p in model.parameters() if p.ndim>=2]; n=[p for p in model.parameters() if p.ndim<2]
    return torch.optim.SGD([{"params":d,"weight_decay":t.WEIGHT_DECAY},{"params":n,"weight_decay":0.}],lr=t.MIN_LR,momentum=t.MOMENTUM,nesterov=True)
opts={k:opt(v) for k,v in models.items()}
dists={k:torch.distributions.Beta(torch.tensor(.2,device=dev),torch.tensor(.2,device=dev)) for k in models}
g=torch.Generator().manual_seed(3)
hx=torch.randn(256,3,32,32,generator=g,pin_memory=True); hy=torch.randint(0,10,(256,),generator=g,pin_memory=True)
emas={"accepted":None,"candidate":None}
states={}
for k in models:
    torch.manual_seed(10); torch.cuda.manual_seed(20)
    states[k]=(torch.get_rng_state(),torch.cuda.get_rng_state())
def one(k,p):
    st=time.perf_counter(); x=hx.to(dev,non_blocking=True); y=hy.to(dev,non_blocking=True); o=opts[k]
    for q in o.param_groups:q["lr"]=t.learning_rate(p*300)
    o.zero_grad(set_to_none=True)
    if p<.65:
        z,a,b,u=t.mixup_batch(x,y,dists[k]); out=models[k](z); loss=u*F.cross_entropy(out,a)+(1-u)*F.cross_entropy(out,b)
    else: out=models[k](x); loss=F.cross_entropy(out,y)
    loss.backward();o.step()
    if k=="candidate" and p>=.65:
        if emas[k] is None: emas[k]=t.ModelEMA(models[k],.999)
        else: emas[k].update(models[k])
    torch.cuda.synchronize(); return 1000*(time.perf_counter()-st)
def win(k,p,n):
    torch.set_rng_state(states[k][0]);torch.cuda.set_rng_state(states[k][1]); a=[one(k,p) for _ in range(n)];states[k]=(torch.get_rng_state(),torch.cuda.get_rng_state());return statistics.mean(a)
for k in models:win(k,.5,25)
order=(("accepted","A"),("candidate","A"),("candidate","B"),("accepted","B"),("accepted","C"),("candidate","C"))
r={z:{k:[] for k in models} for z in ("mixup","hard")}
for z,p in (("mixup",.5),("hard",.8)):
    for k,l in order:
        v=win(k,p,50);r[z][k].append(v);print(z,k+l,f"{v:.6f}")
med={z:{k:statistics.median(v) for k,v in q.items()} for z,q in r.items()}
cv={z:{k:statistics.pstdev(v)/statistics.mean(v) for k,v in q.items()} for z,q in r.items()}
agg={k:.65*med["mixup"][k]+.35*med["hard"][k] for k in models}
ret=agg["accepted"]/agg["candidate"];proj=141.9*ret
print("cvs",cv);print(f"aggregates {agg} retention={ret:.6f} projection={proj:.6f}")
assert all(x<=.05 for q in cv.values() for x in q.values()) and ret>=.95 and proj>=134.8
assert sum(p.numel() for p in candidate.parameters())==691674
assert all(torch.equal(a,b) for a,b in zip(accepted.state_dict().values(),candidate.state_dict().values()))
for pa,pc in zip(accepted.parameters(),candidate.parameters()):
    sa,sc=opts["accepted"].state[pa],opts["candidate"].state[pc]
    assert sa.keys()==sc.keys()
    assert all(torch.equal(sa[k],sc[k]) if torch.is_tensor(sa[k]) else sa[k]==sc[k] for k in sa)
print("PREFLIGHT PASS")
