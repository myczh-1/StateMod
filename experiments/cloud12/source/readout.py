"""Frozen module arithmetic only. No training function or optimizer."""
import math
import numpy as np

def chain(raw,w):
    raw=np.asarray(raw,dtype=np.float64)
    assert raw.ndim==2 and raw.shape[1]==w['center'].size and np.isfinite(raw).all()
    rms=raw/np.sqrt(np.mean(raw*raw,axis=1,keepdims=True)+1e-6)
    x=(rms-w['center'])/float(w['scale'])
    z=x@w['0.weight'].T+w['0.bias']
    erf=np.array([[math.erf(float(v)/math.sqrt(2)) for v in row] for row in z])
    g=.5*z*(1+erf);contrib=g*w['2.weight']
    logits=contrib.sum(1)+w['2.bias'][0];p=np.exp(-np.logaddexp(0,-logits))
    return dict(raw=raw,rms=rms,preprocessed=x,preactivation=z,gelu=g,contribution=contrib,logits=logits,probability=p)

def factorial(rows):
    result=[]
    for family in ['boolean_complement','lower']:
        cell={(q['wording'],q['format']):q for q in rows if q['family']==family}
        tb=cell['train','bare']['logit'];tf=cell['train','fenced']['logit']
        hb=cell['heldout','bare']['logit'];hf=cell['heldout','fenced']['logit']
        result.append(dict(family=family,wording_bare=hb-tb,wording_fenced=hf-tf,
            format_train=tf-tb,format_heldout=hf-hb,interaction=hf-hb-tf+tb,
            decisions={word+'__'+fmt:cell[word,fmt]['enable'] for word,fmt in cell}))
    return result
