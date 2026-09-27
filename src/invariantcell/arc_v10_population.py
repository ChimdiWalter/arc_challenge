"""Fixed count decoders and complete six-metric summaries; no data readers."""
import math
import numpy as np
from .arc_v10 import counts

METRICS=('pds_cosine','expr_mse_unbiased_capped_norm','de_wilcoxon_lfc_nmae',
         'de_wilcoxon_direction_fidelity_yield_raw','de_wilcoxon_direction_reach_raw','de_wilcoxon_sig_jaccard')


def dispersion_fit(training):
    if not training or set(training)-{'K562','RPE1','HepG2'} or len(training)>2:
        raise ValueError('One or two declared development contexts required')
    excess=[];squares=[];depths=[]
    for context in sorted(training):
        groups=training[context]
        for label,x in groups.items():
            x=counts(x)
            if len(x)<2:raise ValueError('Variance requires at least two training cells')
            d=x.sum(1)
            if (d<=0).any():raise ValueError('Invalid training cell depth')
            factor=d.mean()/d
            normalized=x*factor[:,None];mean=normalized.mean(0)
            poisson_variance=(x*factor[:,None]**2).mean(0)
            excess.append(np.maximum(normalized.var(0,ddof=1)-poisson_variance,0.));squares.append(mean**2)
            if label=='control':
                depths.extend(d/d.mean())
    # Method of moments within each group avoids interpreting between-target shift as noise.
    alpha=np.clip(np.sum(excess,axis=0)/np.maximum(np.sum(squares,axis=0),1e-8),1e-4,10.)
    return {'alpha':alpha,'library_multipliers':np.asarray(depths),
            'training_contexts':sorted(training),'method':'pooled_within_group_depth_adjusted_moments_gamma_Poisson; alpha clipped1e-4..10'}


def decode(control,means,n,seed,mode,dispersion=None):
    control=counts(control);means=np.asarray(means,dtype=float)
    if means.ndim!=2 or means.shape[1]!=control.shape[1] or not np.isfinite(means).all() or (means<0).any():
        raise ValueError('Finite aligned nonnegative target means required')
    if type(n)is not int or n<2:raise ValueError('At least two prespecified predicted cells required')
    if mode not in ('empirical','negative_binomial'):raise ValueError('Unknown fixed count decoder')
    rng=np.random.default_rng(seed);outputs=[];base=control.mean(0)
    for target in means:
        if mode=='empirical':
            # With replacement: iid empirical draws, matching evaluator mean-variance correction.
            cells=control[rng.integers(0,len(control),n)].copy()
            measured=base>0
            cells[:,measured]*=target[measured]/base[measured]
            cells[:,~measured]=rng.poisson(target[~measured],size=(n,int((~measured).sum())))
            low=np.floor(cells);cells=low+(rng.random(cells.shape)<cells-low)
        else:
            if dispersion is None:raise ValueError('Training-only dispersion is required')
            alpha=np.asarray(dispersion['alpha']);factors=np.asarray(dispersion['library_multipliers'])
            if alpha.shape!=target.shape or not np.isfinite(alpha).all() or (alpha<=0).any() or factors.ndim!=1 or not len(factors) or not np.isfinite(factors).all() or (factors<=0).any():raise ValueError('Invalid dispersion/depth fit')
            lib=rng.choice(factors,n,replace=True)
            mu=target[None,:]*lib[:,None]
            intensity=rng.gamma(1/alpha[None,:],mu*alpha[None,:])
            cells=rng.poisson(intensity)
        if not np.isfinite(cells).all() or (cells<0).any() or (cells>np.iinfo(np.int32).max).any():raise ValueError('Unsafe decoded counts')
        outputs.append(cells.astype(np.int32))
    return np.stack(outputs)


def six_summary(wide,cohorts):
    """Keep every scored identifier even when undefined; never average raw directions."""
    record=next((r for r in wide.to_dicts() if r.get('statistic')=='mean'),{})
    cohort={r['metric']:r for r in cohorts.to_dicts()}
    values={}
    for name in METRICS:
        value=record.get(name)
        finite=value is not None and math.isfinite(float(value))
        values[name]={'value':float(value) if finite else None,'status':'COMPLETE' if finite else 'UNDEFINED',
                     'cohort':clean_json(cohort.get(name,{}))}
    return {'all_six_raw_finite':all(r['status']=='COMPLETE' for r in values.values()),
            'metrics':values,'naive_raw_average':None}


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,(float,np.floating)):return float(value) if math.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def population_diagnostics(predicted,observed,control):
    x=np.asarray(predicted,dtype=float);y=np.asarray(observed,dtype=float);control=counts(control)
    if x.shape!=y.shape or x.ndim!=3:raise ValueError('Equal fixed target/cell/gene budgets required')
    columns=np.linspace(0,x.shape[2]-1,min(32,x.shape[2]),dtype=int)
    def corr(a):
        z=np.log1p(a[:,:,columns]).reshape(-1,len(columns));z-=z.mean(0)
        scale=np.sqrt(np.sum(z*z,axis=0));return (z.T@z)/np.maximum(scale[:,None]*scale[None,:],1e-12)
    px=x.mean(1);py=y.mean(1)
    return {'cells_per_target':x.shape[1],'targets':x.shape[0],'genes':x.shape[2],
        'historical_log_mean_mse':float(np.mean((np.log1p(px)-np.log1p(py))**2)),
        'mean_cell_variance_predicted':float(x.var(1,ddof=1).mean()),'mean_cell_variance_observed':float(y.var(1,ddof=1).mean()),
        'zero_fraction_predicted':float(np.mean(x==0)),'zero_fraction_observed':float(np.mean(y==0)),
        'mean_depth_predicted':float(x.sum(2).mean()),'mean_depth_observed':float(y.sum(2).mean()),
        'depth_cv_predicted':float(x.sum(2).std()/max(x.sum(2).mean(),1e-12)),
        'depth_cv_observed':float(y.sum(2).std()/max(y.sum(2).mean(),1e-12)),
        'correlation_rmse_fixed32columns':float(np.sqrt(np.mean((corr(x)-corr(y))**2))),
        'correlation_columns':columns.tolist(),'target_mean_variance_predicted':float(np.var(np.log1p(px),axis=0).mean()),
        'target_mean_variance_observed':float(np.var(np.log1p(py),axis=0).mean()),
        'covariance_scope':'Descriptive log-count correlation on32evenly spaced axis columns; not full covariance fidelity'}
