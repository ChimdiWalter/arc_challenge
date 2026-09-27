"""Finite registered official-panel prediction jobs; external submission is absent."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,json,signal,sys,time
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from invariantcell.io import load,write_json,file_hash,utc_now
from invariantcell.v11 import admit_task
from invariantcell.arc_predict import predict
B='data/arc/official_validation_2026'

def prepare(variant):
    if variant not in ('V0','V1'):raise ValueError('Unknown variant')
    paths=[B+'/'+x for x in ('manifest.json','gene_names.csv','pert_counts.csv','context_A.h5ad','context_B.h5ad','context_C.h5ad')]
    paths+=['scripts/arc_official_predict_v12.py','src/invariantcell/arc_predict.py']
    if variant=='V1':paths+=['artifacts/arc/official_transfer_v12.npz','configs/arc_official_transfer_v12.json']
    output='runs/post_confirmation_v12/official-'+variant
    task={'name':'arc-v12-official-'+variant,'boundary':'post_confirmation_v11','purpose':'arc','script':'scripts/arc_official_predict_v12.py','args':['--run',variant],
        'input_sha256':{x:file_hash(P/x) for x in paths},'outputs':[output],'completion_test':'360000 cells on supplied18533gene axis; complete rawcounts prediction; no officialresponse access',
        'resources':{'cpu_threads':1,'ram_gib':8,'wall_seconds':1800,'download_bytes':0,'workspace_growth_bytes':3221225472,'gpu_hours':0,'paid_usd':0},'status':'READY'}
    admitted=admit_task(P,task)
    write_json(P,'configs/arc_official_predict_v12_'+variant+'.json',task,overwrite=False)
    write_json(P,'configs/arc_official_predict_v12_'+variant+'_checks.json',[
        {'path':output+'/receipt.json','equals':{'status':'COMPLETE','submitted':False}},
        {'path':output+'/prediction.h5ad.receipt.json','equals':{'shape':[360000,18533],'official_response_access':False}}],overwrite=False)
    print(json.dumps(admitted))

def run(variant):
    task=load(P/('configs/arc_official_predict_v12_'+variant+'.json'));admit_task(P,task)
    out=P/task['outputs'][0];out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def timeout(*_):raise TimeoutError('Finite official prediction cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1740)
    try:
        r=predict(controls=P/B,manifest=P/B/'manifest.json',genes=P/B/'gene_names.csv',perts=P/B/'pert_counts.csv',
            output=out/'prediction.h5ad',checkpoint=P/'artifacts/arc/official_transfer_v12.npz' if variant=='V1' else None)
        receipt={'status':'COMPLETE','submitted':False,'at':utc_now(),'variant':variant,'wall_seconds':time.monotonic()-start,
            'shape':r['shape'],'nnz':r['nnz'],'artifact_sha256':{x.name:file_hash(x) for x in out.iterdir() if x.is_file()},
            'vcc_prep_status':'PENDING_FINITE_NATIVE_PACKAGING_RESOURCE_PLAN','scientific_status':'UNSCORED_OFFICIAL_CONTEXT_PREDICTION'}
        write_json(P,str(out.relative_to(P)/'receipt.json'),receipt,overwrite=False);print(json.dumps(receipt))
    except BaseException as e:
        write_json(P,str(out.relative_to(P)/'failure.json'),{'error':repr(e),'wall_seconds':time.monotonic()-start},overwrite=False);raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=['V0','V1']);g.add_argument('--run',choices=['V0','V1']);a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    else:run(a.run)
