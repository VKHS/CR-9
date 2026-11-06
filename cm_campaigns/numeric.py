"""Independent probability/label scoring and explicitly paired inference.

No torch, pickle, training, manuscript constants or seed×unit IID substitution.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .common import require, EvidenceError, number, integer


def probability_scores(path: Path, metric: str, *, graph_mean_loss: bool = False) -> dict:
    with np.load(path,allow_pickle=False) as data:
        require({'y_true','y_pred','observation_id','unit_id'}<=set(data.files), f'{path.name}: prediction NPZ missing labels/IDs')
        y=data['y_true']; p=data['y_pred']; ids=data['observation_id']; units=data['unit_id']
    require(y.ndim in (1,2) and p.ndim==2 and len(y)==len(p)>0, 'Invalid prediction shape')
    require(ids.ndim==1 and units.ndim==1 and len(ids)==len(p)==len(units), 'Observation/unit IDs do not match predictions')
    require(len(np.unique(ids))==len(ids), 'Repeated observation IDs')
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Predictions must be finite probabilities')
    p=p.astype(np.float64)
    if metric in ('accuracy','macro_f1'):
        y=y.reshape(-1)
        require(np.isfinite(y).all() and np.equal(y,np.floor(y)).all() and (y>=0).all() and (y<p.shape[1]).all(), 'Invalid class labels')
        require(np.allclose(p.sum(1),1,atol=1e-5,rtol=0), 'Multiclass rows do not sum to one')
        y=y.astype(np.int64); pred=p.argmax(1)
        loss=-np.log(np.clip(p[np.arange(len(p)),y],1e-15,1.0))
        if metric=='accuracy':score=float((pred==y).mean())
        else:
            f=[]
            # Official task definition uses all declared output classes.
            for k in range(p.shape[1]):
                tp=np.sum((pred==k)&(y==k));fp=np.sum((pred==k)&(y!=k));fn=np.sum((pred!=k)&(y==k))
                f.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
            score=float(np.mean(f))
    else:
        require(metric in ('rocauc','ap') and y.shape==p.shape, 'Invalid multilabel prediction specification')
        mask=np.isfinite(y);require(np.isin(y[mask],[0,1]).all(), 'Invalid binary labels')
        require((mask.sum(1)>0).all(),'An observation has no valid tasks')
        logp=np.clip(p,1e-15,1-1e-15)
        term=np.where(mask,-np.nan_to_num(y)*np.log(logp)-(1-np.nan_to_num(y))*np.log1p(-logp),0.)
        loss=term.sum(1)/mask.sum(1)
        scores=[]
        for c in range(p.shape[1]):
            yc=y[mask[:,c],c];pc=p[mask[:,c],c];npos=int(np.sum(yc==1));nneg=int(np.sum(yc==0))
            if not npos or not nneg:continue  # official OGB-style valid tasks
            if metric=='rocauc':
                ranks=rankdata(pc,method='average');scores.append(float((ranks[yc==1].sum()-npos*(npos+1)/2)/(npos*nneg)))
            else:
                order=np.argsort(-pc,kind='stable');yy=yc[order];ss=pc[order]
                ends=np.r_[np.flatnonzero(np.diff(ss))+1,len(ss)]
                cum=np.cumsum(yy)[ends-1];increment=np.diff(np.r_[0,cum]);scores.append(float(np.sum(increment*(cum/ends))/npos))
        require(scores, 'No task with both positive and negative observations')
        score=float(np.mean(scores))
    per_unit={str(u):float(np.mean(loss[units==u])) for u in np.unique(units)}
    risk=float(np.mean(list(per_unit.values()))) if graph_mean_loss else float(loss.mean())
    return {'score':score,'loss':risk,'per_unit_loss':per_unit,'n_observations':len(p),
            'n_units':len(per_unit),'observation_id':ids,'unit_id':units,'y_true':y,'y_pred':p}


def paired_effect(matrix: np.ndarray, *, seed: int, resamples: int, alpha: float, exact_limit: int = 16) -> dict:
    """Crossed seed×unit bootstrap CI and two-sided paired unit sign-flip test.

Seeds and units are resampled as two axes; each index draw is shared by both
methods. Sign flips act on seed-averaged *units*, not on their Cartesian product.
Coverage/exchangeability are assumptions, never guaranteed by this function.
"""
    x=np.asarray(matrix,dtype=np.float64)
    require(x.ndim==2 and min(x.shape)>1 and np.isfinite(x).all(), 'Expected finite seed×unit matrix with >=2 of each')
    require(resamples>=999 and 0<alpha<1,'Insufficient resampling or invalid alpha')
    rng=np.random.default_rng(seed); s,u=x.shape;boot=np.empty(resamples)
    for b in range(resamples):
        boot[b]=x[np.ix_(rng.integers(s,size=s),rng.integers(u,size=u))].mean()
    avg=x.mean(0);observed=abs(avg.mean());tol=1e-14
    if u<=exact_limit:
        total=1<<u;count=0
        for b in range(0,total,1024):
            bits=np.arange(b,min(b+1024,total),dtype=np.uint64)[:,None]
            signs=2*((bits>>np.arange(u,dtype=np.uint64))&1).astype(float)-1
            count+=int(np.sum(np.abs((signs*avg).mean(1))>=observed-tol))
        p=count/total;method='exact paired unit sign-flip, two-sided'
    else:
        count=0
        for b in range(resamples):
            count+=abs(np.mean(avg*rng.choice([-1,1],size=u)))>=observed-tol
        p=(count+1)/(resamples+1);method='Monte Carlo paired unit sign-flip with +1 correction, two-sided'
    lo,hi=np.quantile(boot,[alpha/2,1-alpha/2])
    return {'effect':float(x.mean()),'ci_low':float(lo),'ci_high':float(hi),'p_raw':float(p),
            'seeds':s,'units':u,'resamples':resamples,'resampling_seed':seed,
            'ci_method':'crossed seed-by-unit percentile bootstrap','test':method,
            'scope':'empirical inference; depends on the declared unit/exchangeability assumptions'}


def holm(p_values: list[float]) -> list[float]:
    require(p_values and all(np.isfinite(v) and 0<=v<=1 for v in p_values),'Invalid Holm family')
    order=np.argsort(p_values,kind='stable');out=np.zeros(len(p_values));running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(p_values)-rank)*p_values[i]);out[i]=min(1.,running)
    return out.tolist()


def matrix_from_rows(records: list[dict], value, *, seed_key='seed', unit_key='unit_id') -> tuple[np.ndarray,list,list]:
    seeds=sorted({str(r[seed_key]) for r in records});units=sorted({str(r[unit_key]) for r in records})
    slots={}
    for r in records:
        key=(str(r[seed_key]),str(r[unit_key]));require(key not in slots,'Duplicate seed/unit observation')
        slots[key]=float(value(r))
    require(len(slots)==len(seeds)*len(units),'Unpaired/missing seed×unit observations; no silent imputation')
    return np.array([[slots[s,u] for u in units] for s in seeds]),seeds,units
