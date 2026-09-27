"""One mean-matched count-decoder ablation on already exposed development contexts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['WANDB_MODE']='disabled';os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import argparse,contextlib,csv,hashlib,json,resource,signal,sys,time
import numpy as np
from scipy import sparse
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'src'),str(P/'scripts'),str(P/'.deps/arc_eval')]
import arc_gwps_evaluation_v12 as prior
from invariantcell.io import file_hash,utc_now
from invariantcell.arc_v10_population import METRICS,six_summary,population_diagnostics

OUT=P/'runs/post_confirmation_v13/gwps-decoder-thin-add-v1'
PLAN=OUT.with_name(OUT.name+'-plan.json')
TASK=OUT.with_name(OUT.name+'-task.json')
CHECKS=OUT.with_name(OUT.name+'-checks.json')
NAME='arc-v13-gwps-decoder-thin-add'


def thin_add(sample,effect,rng):
    """Conditional group means match multiplicative transport before count caps.

    Down: Binomial(x, exp(effect)). Up: x + Poisson(mean(sample)*(exp(effect)-1)).
    Zero effects are exact identity. No activation if every control sample is zero.
    """
    block=sparse.csr_matrix(sample,dtype=np.float64,copy=True);block.sum_duplicates()
    effect=np.asarray(effect)
    if effect.shape!=(block.shape[1],) or not np.isfinite(effect).all():raise ValueError('Invalid effect axis/values')
    if block.shape[0]==0 or not np.isfinite(block.data).all() or (block.data<0).any() or (block.data!=np.floor(block.data)).any():raise ValueError('Invalid raw controls')
    if np.asarray(block.sum(axis=1)).max()>1_000_000:raise ValueError('Input cell count cap')
    rate=np.exp(effect.astype(np.float64))
    if not np.isfinite(rate).all():raise ValueError('Unsafe rate')
    x=block.toarray().astype(np.int64);y=x.copy();down=effect<0;up=effect>0
    y[:,down]=rng.binomial(x[:,down],rate[down])
    lam=x[:,up].mean(axis=0)*(rate[up]-1)
    if not np.isfinite(lam).all() or (lam>1_000_000).any():raise ValueError('Unsafe addition rate')
    y[:,up]+=rng.poisson(lam,size=(len(x),int(up.sum())))
    if y.sum(axis=1).max()>1_000_000:raise ValueError('Output cell count cap')
    return sparse.csr_matrix(y.astype(np.float32))


def paired(source,labels,targets,effects,modeled,context,n=32,controls=64):
    if set(labels)!=set(targets)|{'non-targeting'} or np.count_nonzero(labels=='non-targeting')!=controls:raise ValueError('Control or label budget')
    if effects.shape!=(len(targets),source.shape[1]) or np.any(effects[:,~modeled]!=0):raise ValueError('Unmodeled effects')
    result=source.copy().tolil();populations=[];added=0
    for i,target in enumerate(targets):
        positions=np.flatnonzero(labels==target)
        if len(positions)!=n:raise ValueError('Target cell budget')
        key=int.from_bytes(hashlib.sha256(f'17:{context}:{target}'.encode()).digest()[:8],'big')^0xDECAFBAD
        original=source[positions];block=thin_add(original,effects[i],np.random.default_rng(key))
        if (block[:,~modeled]!=original[:,~modeled]).nnz:raise ValueError('Unsupported genes changed')
        added+=int(np.count_nonzero((original.toarray()==0)&(block.toarray()>0)))
        result[positions,:]=block;populations.append(block.toarray())
    result=result.tocsr()
    if (result[labels=='non-targeting']!=source[labels=='non-targeting']).nnz:raise ValueError('Controls changed')
    return result,np.stack(populations),added


def prepare():
    base=prior.verify_plan();pins=dict(base['input_sha256'])
    for name in ['scripts/arc_gwps_decoder_v12.py','tests/test_arc_gwps_decoder_v12.py',str(prior.PLAN.relative_to(P)),str((prior.OUTPUT/'comparison.json').relative_to(P)),str((prior.OUTPUT/'receipt.json').relative_to(P))]:pins[name]=file_hash(P/name)
    for c in prior.CONTEXTS:
        name=str((prior.OUTPUT/c/'metrics/results.csv').relative_to(P));pins[name]=file_hash(P/name)
    plan={**base,'boundary':'post_confirmation_v13_arc_dev','created_at_utc':utc_now(),'input_sha256':pins,'output':str(OUT.relative_to(P)),
        'recipe':'Binomial thinning for negative log effects; independent Poisson count addition with sample-control mean times exp(effect)-1 for positive effects. Fixed source effect shrinkage0.25; zero effects exact identity.',
        'hypothesis':'Allowing positive genes to appear in previously zero cells may improve six-metric development performance with the same conditional population means.',
        'comparison':'Paired original GWPS decoder with identical source checkpoint/control cell populations; historical B0/B1 retained as descriptive references. Same seed, one draw; no seed selection or automatic promotion.',
        'expected_mean':'Per target/gene, E[group mean | sampled controls]=sample_control_mean*exp(effect), same as original multiplicative stochastic rounding.',
        'limits':['Independent gene additions do not learn covariance or activation of genes absent from every sampled control.','One seed and two previously exposed contexts; no confirmatory significance claim.','FID undefined cohorts reported separately; no partial normalized average.'],
        'analysis':'All six native aggregates plus descriptive paired differences over common finite per-target entries, with full undefined counts; derived MSE remains native ratio-of-sums.',
        'resources':{'cpu_threads':1,'ram_gib':8,'wall_seconds':600,'workspace_growth_bytes':200*1024**2,'download_bytes':0,'gpu_hours':0,'paid_usd':0}}
    prior.save(PLAN,plan)
    task={'name':NAME,'boundary':'post_confirmation_v11','purpose':'arc','script':'scripts/arc_gwps_decoder_v12.py','args':['--run'],
        'input_sha256':{**pins,str(PLAN.relative_to(P)):file_hash(PLAN)},'outputs':[str(OUT.relative_to(P))],
        'completion_test':'Both contexts six raw metrics complete; paired cohort diagnostics; predictions committed before responses; no promotion, no official score, no new confirmation.',
        'resources':plan['resources'],'status':'READY'}
    from invariantcell.v11 import admit_task
    admit_task(P,task);prior.save(TASK,task)
    prior.save(CHECKS,[{'path':str((OUT/f).relative_to(P)),'equals':eq} for f,eq in [('receipt.json',{'status':'COMPLETE','new_confirmation':False,'model_promoted':False}),('comparison.json',{'status':'COMPLETE','across_context_scaled_mean':None,'model_promoted':False})]+[(f'{c}/result.json',{'status':'COMPLETE','context':c}) for c in prior.CONTEXTS]])
    print(json.dumps({'status':'FROZEN','plan_sha256':file_hash(PLAN),'resources':plan['resources']}))


def verify():
    plan=prior.read(PLAN)
    if plan['output']!=str(OUT.relative_to(P)) or plan['contexts']!=list(prior.CONTEXTS):raise ValueError('Plan differs')
    for name,h in plan['input_sha256'].items():
        if file_hash(P/name)!=h:raise ValueError('Input changed: '+name)
    prior.validated_alignment()
    return plan


def cohorts(new_path,old_path):
    def values(path):
        with path.open(newline='') as stream:
            return {(r['perturbation'],r['metric']):float(r['value']) if r['value'] else float('nan') for r in csv.DictReader(stream)}
    a=values(new_path);b=values(old_path);result={}
    for metric in METRICS:
        ka={k for k,v in a.items() if k[1]==metric and np.isfinite(v)};kb={k for k,v in b.items() if k[1]==metric and np.isfinite(v)};common=sorted(ka&kb)
        result[metric]={'new_finite':len(ka),'old_finite':len(kb),'common_finite':len(common),'new_only':[k[0] for k in sorted(ka-kb)],'old_only':[k[0] for k in sorted(kb-ka)],
            'mean_new_minus_old_on_common':float(np.mean([a[k]-b[k] for k in common])) if common else None,'scope':'Descriptive same-target intersection; no inferential claim. Derived aggregate metrics can have no per-target entries.'}
    return result


def run(plan):
    import anndata as ad
    import polars as pl
    from cell_eval2.cli import main as evaluator_main
    from cell_eval2.catalog import CATALOG
    if {k for k,v in CATALOG.items() if 'vcc2026' in v.profiles and v.scoring.scored}!=set(METRICS):raise ValueError('Metric catalog differs')
    OUT.mkdir(exist_ok=False);os.environ['NUMBA_CACHE_DIR']=str(OUT/'jit_cache')
    import numba
    numba.config.CACHE_DIR=str(OUT/'jit_cache');os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));start=time.monotonic()
    def timeout(*_):raise TimeoutError('570 second decoder experiment cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(570)
    try:
        genes,targets,effects,modeled=prior.validated_alignment();prior.save(OUT/'frozen_plan.json',plan)
        commitments={};populations={}
        for c in prior.CONTEXTS:
            directory=OUT/c;directory.mkdir();a=ad.read_h5ad(prior.old(c)/'B0_empirical.h5ad')
            if a.var_names.tolist()!=genes:raise ValueError('Gene axis differs')
            labels=a.obs['target'].astype(str).to_numpy()
            a.X,populations[c],added=paired(a.X.tocsr(),labels,targets,effects,modeled,c)
            prediction=directory/'GWPS_thin_add.h5ad';a.write_h5ad(prediction,compression='gzip')
            commitments[c]={'prediction_sha256':file_hash(prediction),'paired_B0_sha256':file_hash(prior.old(c)/'B0_empirical.h5ad'),'new_nonzero_entries':added,'controls_changed':False,'unsupported_genes_changed':False,'held_responses_used':False}
        prior.save(OUT/'prediction_commitment.json',{'at':utc_now(),'contexts':commitments,'plan_sha256':file_hash(PLAN)})
        previous=prior.read(prior.OUTPUT/'comparison.json');results={}
        for c in prior.CONTEXTS:
            directory=OUT/c;begun=time.monotonic()
            args=['run','--adata-pred',str(directory/'GWPS_thin_add.h5ad'),'--adata-real',str(prior.old(c)/'real_development.h5ad'),'--outdir',str(directory/'metrics'),'--config',str(prior.old(c)/'evaluator.yaml'),'--cache-real',str(directory/'real_cache')]
            with (directory/'evaluator.stdout.log').open('x') as stdout,(directory/'evaluator.stderr.log').open('x') as stderr:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):evaluator_main(args)
            summary=six_summary(pl.read_csv(directory/'metrics/agg_results.csv'),pl.read_csv(directory/'metrics/metric_aggregation.csv'))
            if not summary['all_six_raw_finite']:raise ValueError('Incomplete six metric vector')
            real=ad.read_h5ad(prior.old(c)/'real_development.h5ad');labels=real.obs['target'].astype(str).to_numpy()
            truth=np.stack([real.X[labels==t].toarray() for t in targets]);control=real.X[labels=='non-targeting'].toarray()
            summary['population_diagnostics']=population_diagnostics(populations[c],truth,control)
            results[c]={'status':'COMPLETE','context':c,'candidate':summary,'original_gwps':previous['contexts'][c]['candidate'],'frozen_comparators':previous['contexts'][c]['frozen_comparators'],
                'matched_target_cohorts':cohorts(directory/'metrics/results.csv',prior.OUTPUT/c/'metrics/results.csv'),
                'normalization':{'status':'NOT_COMPUTED','reason':'Paired raw-metric ablation; RPE1 reference scale remains invalid; no cross-context scaled score.'},'evaluator_wall_seconds':time.monotonic()-begun,'new_confirmation':False,'official_score':False}
            prior.save(directory/'result.json',results[c])
        prior.save(OUT/'comparison.json',{'status':'COMPLETE','contexts':results,'across_context_scaled_mean':None,'model_promoted':False,'official_score':False,'new_confirmation':False})
        prior.save(OUT/'receipt.json',{'status':'COMPLETE','at':utc_now(),'wall_seconds':time.monotonic()-start,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'plan_sha256':file_hash(PLAN),'new_confirmation':False,'model_promoted':False,'download_bytes':0,'gpu_hours':0,'paid_usd':0,
            'artifact_sha256':{str(f.relative_to(OUT)):file_hash(f) for f in OUT.rglob('*') if f.is_file() and 'jit_cache' not in f.parts}})
        print(json.dumps({'status':'COMPLETE','output':str(OUT.relative_to(P))}))
    except BaseException as e:
        prior.save(OUT/'failure.json',{'status':'FAILED','error':type(e).__name__+': '+str(e),'wall_seconds':time.monotonic()-start,'automatic_retry':False});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    else:
        plan=verify()
        if a.check:print(json.dumps({'status':'INPUT_PINS_VERIFIED','plan_sha256':file_hash(PLAN),'output_exists':OUT.exists(),'resources':plan['resources']}))
        else:run(plan)
