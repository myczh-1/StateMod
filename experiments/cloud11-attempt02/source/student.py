"""Fixed CPU student; training functions accept training examples only."""
import math
import numpy as np

def rms(x):return x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-6)

def teacher_target(natural,candidates):
    f=float(candidates['frozenset']['type_eos_logp']);s=float(candidates['set']['type_eos_logp'])
    p=float(np.exp(f-np.logaddexp(f,s)))
    answer=natural['text'].strip()
    valid=(natural['stop']=='EOS' and answer in ('frozenset','set') and np.isfinite([f,s,p]).all()
           and (p>.5)==(answer=='frozenset'))
    return dict(valid=bool(valid),p=p,action=answer if answer in ('frozenset','set') else 'INVALID',
                fro_logp=f,set_logp=s)

def validate_teachers(rows):
    return len(rows)==8 and all(r['valid'] for r in rows) and min(sum(r['action']==v for r in rows) for v in ('frozenset','set'))>=2

def fit(raw,target,cfg):
    import torch
    torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    assert raw.shape==(8,5120) and target.shape==(8,)
    assert np.isfinite(raw).all() and np.isfinite(target).all() and ((target>=0)&(target<=1)).all()
    x=rms(raw.astype(np.float64));center=x.mean(0);scale=max(float(np.sqrt(np.mean((x-center)**2))),1e-6);x=(x-center)/scale
    model=torch.nn.Sequential(torch.nn.Linear(5120,8),torch.nn.GELU(),torch.nn.Linear(8,1)).to(device='cpu',dtype=torch.float64)
    xx=torch.tensor(x,dtype=torch.float64,device='cpu');yy=torch.tensor(target,dtype=torch.float64,device='cpu')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);losses=[]
    entropy=-(torch.special.xlogy(yy,yy)+torch.special.xlogy(1-yy,1-yy)).mean()
    for step in range(cfg['steps']):
        opt.zero_grad(set_to_none=True);z=model(xx).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yy)-entropy
        assert torch.isfinite(loss),'STOP_NUMERICAL loss';loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters()),'STOP_NUMERICAL gradient'
        opt.step();losses.append(float(loss.detach()))
    weights={k:v.detach().numpy().copy() for k,v in model.state_dict().items()};weights.update(center=center,scale=np.array(scale))
    assert all(np.isfinite(v).all() for v in weights.values()),'STOP_NUMERICAL weights'
    return weights,losses

def infer(raw,w):
    x=(rms(raw.astype(np.float64))-w['center'])/float(w['scale'])
    z=x@w['0.weight'].T+w['0.bias'];g=.5*z*(1+np.array([[math.erf(float(v)/math.sqrt(2)) for v in row] for row in z]))
    logits=(g@w['2.weight'].T+w['2.bias']).ravel()
    probability=np.exp(-np.logaddexp(0,-logits))
    return logits,probability

def reconstruction_error(raw,w):
    import torch
    model=torch.nn.Sequential(torch.nn.Linear(5120,8),torch.nn.GELU(),torch.nn.Linear(8,1)).double()
    model.load_state_dict({k:torch.tensor(w[k]) for k in model.state_dict()})
    x=(rms(raw.astype(np.float64))-w['center'])/float(w['scale'])
    with torch.no_grad():z=model(torch.tensor(x,dtype=torch.float64)).flatten().numpy()
    return float(np.max(np.abs(z-infer(raw,w)[0])))

def fidelity(prob,teacher):
    p=np.array(prob);y=np.array([r['action']=='frozenset' for r in teacher]);decision=p>.5
    return dict(total=len(p),agreement=int(np.sum(decision==y)),teacher_enable=int(y.sum()),teacher_reject=int((~y).sum()),
                enables_reproduced=int(np.sum(decision&y)),rejects_reproduced=int(np.sum(~decision&~y)),
                MAE=float(np.mean(np.abs(p-np.array([r['p'] for r in teacher])))),
                always_on_agreement=int(y.sum()),always_off_agreement=int((~y).sum()))

def passes(m):
    return m['total']==8 and m['agreement']>=7 and m['teacher_reject']==4 and m['rejects_reproduced']==4 and m['enables_reproduced']>=3
