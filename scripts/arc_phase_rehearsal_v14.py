"""Small synthetic D/E/F CLI rehearsal; never reads final or validation responses."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,subprocess,sys,time,resource
import numpy as np
from scipy import sparse
import anndata as ad
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from invariantcell.io import file_hash,utc_now
D=P/'runs/engineering_v14/synthetic_DEF_rehearsal'

def main():
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));start=time.monotonic();D.mkdir(parents=True,exist_ok=False)
    genes=['SYN_G3','SYN_G1','SYN_G2'];targets=['SYN_NEW_TARGET_9','SYN_NEW_TARGET_2'];contexts=['D','E','F']
    manifest={'synthetic':True,'label':'SYNTHETIC_ENGINEERING_NOT_FINAL_DATA','contexts':contexts,'n_genes':3,'n_constructs':2,'cells_per_pert':4,'control_label':'non-targeting','context_col':'context','pert_col':'target_gene'}
    (D/'synthetic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(D/'synthetic_genes.csv').write_text('gene_name\n'+'\n'.join(genes)+'\n');(D/'synthetic_perts.csv').write_text('target_gene\n'+'\n'.join(targets)+'\n')
    for i,c in enumerate(contexts):
        a=ad.AnnData(sparse.csr_matrix([[0,2+i,0],[4,0,1+i],[3,1,0]],dtype=np.float32),obs=pd.DataFrame({'context':[c]*3,'target_gene':['non-targeting']*3},index=[f'SYN_{c}_{j}'for j in range(3)]),var=pd.DataFrame(index=genes));a.uns['fixture_scope']='SYNTHETIC_ENGINEERING';a.write_h5ad(D/f'context_{c}.h5ad')
    np.savez_compressed(D/'synthetic_checkpoint.npz',genes=genes,targets=targets,log_fold_change=np.array([[.1,-.2,0],[0,0,0]],dtype=np.float32))
    config={'seed':17,'max_nnz':4750000000,'checkpoint':'synthetic_checkpoint.npz','code_sha256':file_hash(P/'src/invariantcell/arc_predict.py'),'scope':'Synthetic phase plumbing, not response prediction evidence'};(D/'synthetic_config.json').write_text(json.dumps(config,indent=2)+'\n')
    receipts=[]
    for repetition in (1,2):
        output=D/f'synthetic_prediction_{repetition}.h5ad';args=[sys.executable,'-B','-m','invariantcell.arc_predict','--controls',str(D),'--manifest',str(D/'synthetic_manifest.json'),'--genes',str(D/'synthetic_genes.csv'),'--perts',str(D/'synthetic_perts.csv'),'--checkpoint',str(D/'synthetic_checkpoint.npz'),'--output',str(output),'--seed','17']
        with (D/f'cli_{repetition}.log').open('x')as log:subprocess.run(args,cwd=P,env={**os.environ,'PYTHONPATH':str(P/'src')},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        a=ad.read_h5ad(output);r=json.loads(output.with_suffix('.h5ad.receipt.json').read_text());receipts.append(r)
        assert a.shape==(24,3) and a.var_names.tolist()==genes
        assert set(a.obs.context)==set(contexts) and set(a.obs.target_gene)==set(targets)
        assert set(a.obs.groupby(['context','target_gene'],observed=True).size())=={4}
        assert sparse.isspmatrix_csr(a.X) and np.isfinite(a.X.data).all() and (a.X.data>=0).all() and (a.X.data==np.floor(a.X.data)).all()
        if repetition==1:first=a
        else:assert (first.X!=a.X).nnz==0 and first.obs.equals(a.obs) and first.var.equals(a.var)
    deterministic=['model_sha256','control_sha256','manifest_sha256','genes_sha256','perts_sha256','code_sha256','seed','shape','nnz','context_nnz']
    assert all(receipts[0][k]==receipts[1][k]for k in deterministic)
    result={'status':'PASS_SYNTHETIC_PHASE_AGNOSTIC_CLI','at':utc_now(),'scope':'Tiny synthetic D/E/F fixture; not final challenge data or a full-size submission. No hidden responses accessed.','dynamic_contexts':contexts,'dynamic_targets':targets,'exact_gene_order':genes,'shape':[24,3],'deterministic_matrix_obs_var':True,'deterministic_receipt_fields':deterministic,'config_sha256':file_hash(D/'synthetic_config.json'),'code_sha256':config['code_sha256'],'wall_seconds':time.monotonic()-start,'maxrss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'artifact_sha256':{str(f.relative_to(D)):file_hash(f)for f in D.iterdir()if f.is_file()}}
    (D/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');report=P/'reports/arc/phase_independent_runner_v14.json';report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['status','shape','deterministic_matrix_obs_var','wall_seconds']}))

if __name__=='__main__':main()
