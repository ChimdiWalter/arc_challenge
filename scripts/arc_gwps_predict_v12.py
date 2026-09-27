"""Generate the new genome-wide-source candidate; preserve the zero-overlap attempt."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','POLARS_MAX_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,json,signal,sys,time
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from invariantcell.io import load,write_json,file_hash,utc_now
from invariantcell.v11 import admit_task
from invariantcell.arc_predict import predict
B='data/arc/official_validation_2026';OUT='runs/post_confirmation_v12/official-V1';TASK='configs/arc_gwps_predict_v12_task.json'
MODEL='artifacts/arc/gwps_transfer_v12.npz'

def prepare():
    paths=[B+'/'+x for x in ('manifest.json','gene_names.csv','pert_counts.csv','context_A.h5ad','context_B.h5ad','context_C.h5ad')]
    paths+=['scripts/arc_gwps_predict_v12.py','src/invariantcell/arc_predict.py',MODEL,'configs/arc_gwps_transfer_v12.json']
    task={'name':'arc-v12-official-V1-gwps','boundary':'post_confirmation_v11','purpose':'arc','script':'scripts/arc_gwps_predict_v12.py','args':['--run'],
        'input_sha256':{x:file_hash(P/x) for x in paths},'outputs':[OUT],'completion_test':'Full official manifest prediction with independently pinned GWPS effects; unsupportedgenes/targets retain controls',
        'resources':{'cpu_threads':1,'ram_gib':8,'wall_seconds':1800,'download_bytes':0,'workspace_growth_bytes':3221225472,'gpu_hours':0,'paid_usd':0},'status':'READY'}
    print(json.dumps(admit_task(P,task)));write_json(P,TASK,task,overwrite=False)
    write_json(P,'configs/arc_gwps_predict_v12_checks.json',[{'path':OUT+'/receipt.json','equals':{'status':'COMPLETE','submitted':False}},
        {'path':OUT+'/prediction.h5ad.receipt.json','equals':{'shape':[360000,18533],'official_response_access':False}}],overwrite=False)

def run():
    task=load(P/TASK);admit_task(P,task);out=P/OUT;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def timeout(*_):raise TimeoutError('Finite official prediction cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1740)
    try:
        r=predict(controls=P/B,manifest=P/B/'manifest.json',genes=P/B/'gene_names.csv',perts=P/B/'pert_counts.csv',output=out/'prediction.h5ad',checkpoint=P/MODEL)
        result={'status':'COMPLETE','submitted':False,'at':utc_now(),'wall_seconds':time.monotonic()-start,'shape':r['shape'],'nnz':r['nnz'],
            'model_sha256':file_hash(P/MODEL),'artifact_sha256':{x.name:file_hash(x) for x in out.iterdir() if x.is_file()},
            'native_prep':'PENDING_RESOURCE_CAP_APPROVAL','scientific_status':'NEW_PSEUDOBULK_TRANSFER_CANDIDATE_NOT_PROMOTED_OR_OFFICIALLY_SCORED'}
        write_json(P,OUT+'/receipt.json',result,overwrite=False);print(json.dumps(result))
    except BaseException as error:
        write_json(P,OUT+'/failure.json',{'error':repr(error),'wall_seconds':time.monotonic()-start},overwrite=False);raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    else:run()
