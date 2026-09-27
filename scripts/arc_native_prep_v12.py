"""Finite native VCC validation/packaging, admitted only under approved caps.

This runs the installed unmodified vcc prep locally; no authentication or submission.
"""
from pathlib import Path
import argparse,json,os,resource,signal,subprocess,sys,time
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from invariantcell.io import load,write_json,file_hash,utc_now,configuration
from invariantcell.post_confirmation import validate

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['V0','V1'],required=True);p.add_argument('--operation',choices=['dry-run','package'],required=True)
    p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    source=P/f'runs/post_confirmation_v12/official-{a.variant}/prediction.h5ad'
    output=P/f'runs/post_confirmation_v12/prep-{a.variant}-{a.operation}'
    b=P/'data/arc/official_validation_2026';m=load(b/'manifest.json')
    cmd=[str(P/'.venvs/vcc/bin/vcc'),'prep',str(source),'-g',str(b/'gene_names.csv'),'--perts',str(b/'pert_counts.csv'),
        '--contexts',','.join(m['contexts']),'--expected-gene-dim',str(m['n_genes']),'--cells-per-pert',str(m['cells_per_pert']),
        '--require-counts','--reject-controls','--json','-o',str(output/'prediction.vcc')]
    if a.operation=='dry-run':cmd+=['--dry-run']
    plan={'argv':cmd,'ram_gib':32,'workspace_peak_limit_gib':64,'cpu_threads':1,'wall_seconds':1800,'paid_usd':0,'submitted':False}
    if a.plan_only:print(json.dumps(plan,indent=2));return
    validate(P);cfg=configuration(P)
    if cfg['compute']['max_ram_gib']<32 or cfg['compute']['max_workspace_growth_gib']<64:
        raise PermissionError('Native full-panel prep requires the authorized32GiB RAM/64GiB storage caps')
    approval=load(P/'state/local_packaging_resource_approval_v12.json')
    if approval.get('user_approved') is not True:raise PermissionError('Explicit cap approval record missing')
    mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    available=int(mem['MemAvailable'].split()[0])*1024
    if available<40*1024**3:raise RuntimeError('Native prep requires32GiB plus8GiB currently available host headroom; retry only after resource state changes')
    if output.exists():raise FileExistsError('Preserve previous attempt: '+str(output))
    receipt=load(source.with_name('prediction.h5ad.receipt.json'))
    if file_hash(source)!=receipt['prediction_sha256']:raise ValueError('Prediction changed before packaging')
    if a.operation=='package':
        dry=load(P/f'runs/post_confirmation_v12/prep-{a.variant}-dry-run/receipt.json')
        if dry['status']!='COMPLETE' or dry['input_prediction_sha256']!=receipt['prediction_sha256']:raise ValueError('Matching successful native dry-run required before packaging')
    output.mkdir(parents=True);temp=output/'tmp';temp.mkdir();start=time.monotonic()
    def limits():
        os.nice(10);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(32*1024**3,32*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1800,1800))
    env={**os.environ,'TMPDIR':str(temp),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    with (output/'stdout.json').open('x') as stdout,(output/'stderr.log').open('x') as stderr:
        try:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,preexec_fn=limits,start_new_session=True)
            try:code=proc.wait(timeout=1770)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
            status='COMPLETE' if code==0 else 'FAILED'
        except BaseException as error:
            write_json(P,str(output.relative_to(P)/'failure.json'),{'error':repr(error),'wall_seconds':time.monotonic()-start},overwrite=False);raise
    result={**plan,'status':status,'exit_code':code,'at':utc_now(),'wall_seconds':time.monotonic()-start,
        'input_prediction_sha256':receipt['prediction_sha256'],'artifact_sha256':{x.name:file_hash(x) for x in output.iterdir() if x.is_file()},
        'native_cli_version':'0.2.1','official_score':None,'submitted':False,
        'host_mem_available_bytes_before':available,'peak_child_rss_KiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    write_json(P,str(output.relative_to(P)/'receipt.json'),result,overwrite=False);print(json.dumps(result,indent=2))
    if code:raise SystemExit(code)

if __name__=='__main__':main()
