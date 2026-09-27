"""Fixed GWPS transport on paired archived B0 cells; exposed development only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['WANDB_MODE']='disabled';os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import argparse,contextlib,hashlib,json,resource,signal,sys,time
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'src'),str(P/'.deps/arc_eval')]
from invariantcell.io import file_hash,utc_now
from invariantcell.arc_predict import transport
from invariantcell.arc_v10_population import METRICS,six_summary,clean_json,population_diagnostics

PLAN=P/'runs/post_confirmation_v12/gwps-public-evaluation-v1-plan.json'
OUTPUT=P/'runs/post_confirmation_v12/gwps-public-evaluation-v1'
MODEL='artifacts/arc/gwps_development_effects_v12.npz'
CONTEXTS=('RPE1','HepG2')


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as out:json.dump(clean_json(value),out,indent=2,allow_nan=False);out.write('\n')
def old(context):return P/f'runs/post_confirmation_v10/arc-population-{context}-20260923a'


def align_effects(genes,targets,model_genes,stable_ids,model_targets,effects,modeled):
    if list(targets)!=list(model_targets):raise ValueError('Development target order differs')
    if len(genes)!=len(set(genes)) or len(targets)!=len(set(targets)):raise ValueError('Duplicate development axis')
    if len(model_genes)!=len(set(model_genes)):raise ValueError('Duplicate model gene symbols')
    if len(stable_ids)!=len(model_genes) or effects.shape!=(len(targets),len(model_genes)) or modeled.shape!=(len(model_genes),) or modeled.dtype!=np.dtype(bool):raise ValueError('Effect axes differ')
    selected=[str(s) for s,m in zip(stable_ids,modeled) if m]
    if any(not s for s in selected) or len(selected)!=len(set(selected)):raise ValueError('Modeled stable IDs missing or ambiguous')
    if not np.isfinite(effects).all() or np.any(effects[:,~modeled]!=0):raise ValueError('Invalid/unmodeled nonzero effects')
    index={str(s):i for i,(s,m) in enumerate(zip(stable_ids,modeled)) if m}
    result=np.zeros((len(targets),len(genes)),dtype=np.float32);mask=np.zeros(len(genes),dtype=bool)
    for j,g in enumerate(genes):
        if g in index:result[:,j]=effects[:,index[g]];mask[j]=True
    return result,mask


def paired_transport(source,labels,targets,effects,modeled,context,n=32,controls=64):
    if set(labels)!=set(targets)|{'non-targeting'}:raise ValueError('B0 target labels differ')
    if np.count_nonzero(labels=='non-targeting')!=controls:raise ValueError('Wrong control budget')
    result=source.copy().tolil();cells=[]
    for i,target in enumerate(targets):
        positions=np.flatnonzero(labels==target)
        if len(positions)!=n:raise ValueError('Wrong target budget')
        key=int.from_bytes(hashlib.sha256(f'17:{context}:{target}'.encode()).digest()[:8],'big')
        block=transport(source[positions],effects[i],np.random.default_rng(key^0xDECAFBAD))
        if (block[:,~modeled]!=source[positions][:,~modeled]).nnz:raise ValueError('Unmatched gene counts changed')
        result[positions,:]=block;cells.append(block.toarray())
    result=result.tocsr()
    if (result[labels=='non-targeting']!=source[labels=='non-targeting']).nnz:raise ValueError('Controls changed')
    return result,np.stack(cells)


def make_plan():
    legacy=read(P/'configs/post_confirmation_v10_arc_population_protocol.json')
    names=['scripts/arc_gwps_evaluation_v12.py','src/invariantcell/arc_predict.py','src/invariantcell/arc_v10_population.py',
           'configs/multicontext_protocol.json','configs/arc_gwps_transfer_v12.json',MODEL,
           'artifacts/arc/gwps_transfer_v12_receipt.json','data/arc/official_validation_2026/gene_names.csv',
           'src/invariantcell/arc_v10.py','tests/test_arc_gwps_evaluation_v12.py']
    pins={name:sha for name,sha in legacy['input_sha256'].items() if name.startswith('.deps/arc_eval/')}
    for c in CONTEXTS:
        directory=old(c);receipt=read(directory/'receipt.json')
        required=['B0_empirical.h5ad','real_development.h5ad','evaluator.yaml','six_metric_population_report.json',
                  'generic_baseline/baseline_agg.csv','generic_baseline/baseline_meta.json',
                  'replicate_anchor/anchor_agg.parquet','replicate_anchor/anchor_meta.json','replicate_anchor/anchor_splits.parquet']
        for name in required:
            path=directory/name
            if file_hash(path)!=receipt['artifact_sha256'][name]:raise ValueError('Archived population input changed: '+str(path))
            names.append(str(path.relative_to(P)))
        names.append(str((directory/'receipt.json').relative_to(P)))
    for name in names:pins[name]=file_hash(P/name)
    return {'status':'FROZEN_BEFORE_NEW_PREDICTIONS','boundary':'post_confirmation_v12','created_at_utc':utc_now(),
        'contexts':list(CONTEXTS),'targets':12,'genes':6642,'cells_per_target':32,'controls':64,'seed':17,
        'model':MODEL,'input_sha256':pins,'output':str(OUTPUT.relative_to(P)),
        'recipe':'Fixed source logratio shrinkage0.25 already in model; root transport unchanged; zero unmatched effects; no depth restoration',
        'paired_controls':'Exact archived B0_empirical.h5ad target populations, unchanged control block; no fresh resampling',
        'rounding_seed':'SHA256 of17:context:target first8bytes big-endian XOR0xDECAFBAD; same root transport seed policy',
        'reference_policy':'Reuse frozen B0/B1/diffusion/NB results and baseline/anchor; no old arm evaluation; RPE1 legacy scale refusal remains decisive',
        'source_scope':'GWPS K562 day8 training; RPE1/HepG2 previously exposed development; no new confirmation',
        'model_promotion':False,'official_score':False,'Jurkat_access':False,
        'resources':{'cpu_threads':1,'ram_gib':8,'wall_seconds':600,'workspace_growth_bytes':300*1024**2,'download_bytes':0,'gpu_hours':0,'paid_usd':0}}


def verify_plan():
    plan=read(PLAN)
    if plan['contexts']!=list(CONTEXTS) or plan['status']!='FROZEN_BEFORE_NEW_PREDICTIONS':raise ValueError('Wrong frozen plan')
    for name,sha in plan['input_sha256'].items():
        if file_hash(P/name)!=sha:raise ValueError('Pinned input changed: '+name)
    from cell_eval2.catalog import CATALOG
    if {k for k,v in CATALOG.items() if 'vcc2026' in v.profiles and v.scoring.scored}!=set(METRICS):raise ValueError('Six-metric catalog differs')
    receipt=read(P/'artifacts/arc/gwps_transfer_v12_receipt.json')
    if receipt['artifacts'][MODEL]['sha256']!=file_hash(P/MODEL) or receipt['recipe']['shrinkage']!=.25:raise ValueError('GWPS training receipt differs')
    validated_alignment()
    return plan


def validated_alignment():
    protocol=read(P/'configs/multicontext_protocol.json');genes=protocol['genes'];targets=protocol['targets']
    if len(genes)!=6642 or len(targets)!=12:raise ValueError('Frozen development dimensions differ')
    with np.load(P/MODEL,allow_pickle=False)as z:
        from invariantcell.arc_predict import names
        if z['genes'].tolist()!=names(P/'data/arc/official_validation_2026/gene_names.csv','gene_name'):raise ValueError('Model official symbol axis changed')
        if not z['per_target_supported'].all():raise ValueError('Development source target unsupported')
        effects,modeled=align_effects(genes,targets,z['genes'].tolist(),z['stable_ids'].tolist(),z['targets'].tolist(),z['log_fold_change'],z['modeled_gene'])
    return genes,targets,effects,modeled


def execute(plan):
    import anndata as ad
    import polars as pl
    from cell_eval2.cli import main as evaluator_main
    genes,targets,effects,modeled=validated_alignment()
    save(OUTPUT/'frozen_plan.json',plan)
    save(OUTPUT/'alignment.json',{'development_genes':len(genes),'mapped_genes':int(modeled.sum()),'unmatched_genes':int((~modeled).sum()),
        'effect_sha256':hashlib.sha256(effects.tobytes()).hexdigest(),'unmatched_effect_max_abs':float(np.abs(effects[:,~modeled]).max(initial=0))})
    populations={}; commitments={}
    # Generate and commit both contexts before opening either real-response H5AD.
    for context in CONTEXTS:
        directory=OUTPUT/context;directory.mkdir()
        b0=ad.read_h5ad(old(context)/'B0_empirical.h5ad')
        if b0.var_names.tolist()!=genes or set(b0.obs['target'].astype(str))!=set(targets)|{'non-targeting'}:raise ValueError('B0 axes/labels differ')
        labels=b0.obs['target'].astype(str).to_numpy();source=b0.X.tocsr()
        # Preserve original row order and obs IDs exactly, including evaluator control rows.
        b0.X,populations[context]=paired_transport(source,labels,targets,effects,modeled,context)
        prediction=directory/'GWPS_transport.h5ad';b0.write_h5ad(prediction,compression='gzip')
        commitments[context]={'prediction_sha256':file_hash(prediction),'paired_B0_sha256':file_hash(old(context)/'B0_empirical.h5ad'),
            'unmatched_count_entries_changed':0,'controls_changed':False,'held_responses_used':False}
    save(OUTPUT/'prediction_commitment.json',{'at':utc_now(),'contexts':commitments,'plan_sha256':file_hash(PLAN)})
    results={}
    for context in CONTEXTS:
        directory=OUTPUT/context;historical=read(old(context)/'six_metric_population_report.json')
        def call(name,args):
            begun=time.monotonic()
            with (directory/(name+'.stdout.log')).open('x')as out,(directory/(name+'.stderr.log')).open('x')as err:
                with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):evaluator_main(args)
            save(directory/(name+'_execution.json'),{'status':'COMPLETE','argv':args,'wall_seconds':time.monotonic()-begun})
        call('candidate',['run','--adata-pred',str(directory/'GWPS_transport.h5ad'),'--adata-real',str(old(context)/'real_development.h5ad'),
            '--outdir',str(directory/'metrics'),'--config',str(old(context)/'evaluator.yaml'),'--cache-real',str(directory/'real_cache')])
        summary=six_summary(pl.read_csv(directory/'metrics/agg_results.csv'),pl.read_csv(directory/'metrics/metric_aggregation.csv'))
        real=ad.read_h5ad(old(context)/'real_development.h5ad');labels=real.obs['target'].astype(str).to_numpy()
        truth=np.stack([real.X[labels==t].toarray()for t in targets]);controls=real.X[labels=='non-targeting'].toarray()
        summary['population_diagnostics']=population_diagnostics(populations[context],truth,controls)
        normalization={'status':historical['normalization']['status'],'official':False,'scores':[]}
        if context=='RPE1':
            normalization.update(status='UNDEFINED_OR_INCOMPATIBLE_NATIVE_NORMALIZATION',reason=historical['normalization']['reason'],reference_recomputed=False)
        else:
            try:
                call('normalized',['score','--user-agg',str(directory/'metrics/agg_results.csv'),'--baseline-agg',str(old(context)/'generic_baseline/baseline_agg.csv'),
                    '--anchor',str(old(context)/'replicate_anchor'),'--output',str(directory/'normalized.csv')])
                normalization['scores']=clean_json(pl.read_csv(directory/'normalized.csv').to_dicts())
            except (ValueError,SystemExit)as error:
                normalization.update(status='UNDEFINED_OR_INCOMPATIBLE_NATIVE_NORMALIZATION',reason=type(error).__name__+': '+str(error))
        results[context]={'status':'COMPLETE','context':context,'candidate':summary,'normalization':normalization,'frozen_comparators':historical['arms'],'new_confirmation':False,'official_score':False}
        save(directory/'result.json',results[context])
    save(OUTPUT/'comparison.json',{'status':'COMPLETE','contexts':results,'across_context_scaled_mean':None,
        'reason_no_mean':'RPE1 frozen native scale is undefined; do not average a partial panel','model_promoted':False,'official_score':False,'new_confirmation':False})


def main():
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-plan',action='store_true');group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare_plan:save(PLAN,make_plan());print(str(PLAN));return
    plan=verify_plan()
    if a.check:print(json.dumps({'status':'READY_NOT_EVALUATED','plan_sha256':file_hash(PLAN),'pins':len(plan['input_sha256']),'resources':plan['resources']}));return
    OUTPUT.mkdir(parents=True,exist_ok=False);os.environ['NUMBA_CACHE_DIR']=str(OUTPUT/'jit_cache')
    import numba
    numba.config.CACHE_DIR=str(OUTPUT/'jit_cache')
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));start=time.monotonic()
    def expired(*_):raise TimeoutError('GWPS evaluator570second internal cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(570)
    try:
        execute(plan)
        save(OUTPUT/'receipt.json',{'status':'COMPLETE','wall_seconds':time.monotonic()-start,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'plan_sha256':file_hash(PLAN),'new_confirmation':False,'model_promoted':False,'download_bytes':0,'gpu_hours':0,'paid_usd':0,
            'artifact_sha256':{str(x.relative_to(OUTPUT)):file_hash(x)for x in OUTPUT.rglob('*')if x.is_file()and'jit_cache'not in x.parts}})
        print(json.dumps({'status':'COMPLETE','output':str(OUTPUT.relative_to(P))}))
    except BaseException as error:
        save(OUTPUT/'failure.json',{'status':'FAILED','error':type(error).__name__+': '+str(error),'wall_seconds':time.monotonic()-start,'automatic_retry':False});raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
