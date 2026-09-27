"""Phase-independent, bounded-memory official-control count transport.

No response reader. The checkpoint is a pinned, separately trained array artifact.
Zero effects preserve control entries exactly; all generated artifacts are predictions.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,time
import numpy as np
from .io import file_hash,write_json,utc_now

def names(path,header):
    with Path(path).open(newline='') as f:rows=list(csv.reader(f))
    if rows and rows[0]==[header]:rows=rows[1:]
    if not rows or any(len(r)!=1 or not r[0] for r in rows):raise ValueError('Expected one named column')
    values=[r[0] for r in rows]
    if len(values)!=len(set(values)):raise ValueError('Duplicate identifier')
    return values

def contract(manifest,genes,perts):
    m=json.loads(Path(manifest).read_text());g=names(genes,'gene_name');t=names(perts,'target_gene')
    contexts=m['contexts']
    if not contexts or len(set(contexts))!=len(contexts) or any(not isinstance(c,str) or not c or '/' in c or '\\' in c or c in ('.','..') for c in contexts):raise ValueError('Invalid context identifiers')
    if len(g)!=m['n_genes'] or len(t)!=m['n_constructs']:raise ValueError('Manifest/axis cardinality mismatch')
    n=m['cells_per_pert']
    if type(n)is not int or n<=0 or m['control_label'] in t:raise ValueError('Invalid prediction panel')
    return m,g,t

def transport(sample,effect,rng):
    from scipy import sparse
    block=sparse.csr_matrix(sample,dtype=np.float64,copy=True);block.sum_duplicates();block.sort_indices()
    if not np.isfinite(effect).all() or len(effect)!=block.shape[1]:raise ValueError('Invalid frozen effect')
    if not np.isfinite(block.data).all() or (block.data<0).any() or (block.data!=np.floor(block.data)).any():raise ValueError('Invalid raw controls')
    adjusted=block.data*np.exp(effect[block.indices])
    if not np.isfinite(adjusted).all() or (adjusted>1_000_000).any():raise ValueError('Unsafe transported count')
    floor=np.floor(adjusted);block.data=(floor+(rng.random(len(floor))<adjusted-floor)).astype('float32')
    block.eliminate_zeros()
    if (np.asarray(block.sum(axis=1,dtype=np.float64)).ravel()>1_000_000).any():raise ValueError('Cell count cap exceeded')
    return block

def predict(*,controls,manifest,genes,perts,output,checkpoint=None,seed=17,max_nnz=4750000000):
    import h5py,anndata as ad
    start=time.monotonic();out=Path(output).resolve();root=Path(controls).resolve()
    m,g,t=contract(manifest,genes,perts);contexts=m['contexts'];n=m['cells_per_pert'];expected=len(contexts)*len(t)*n
    if expected>400000:raise ValueError('Official maximum cells exceeded')
    if out.exists() or out.with_suffix(out.suffix+'.partial').exists():raise FileExistsError(out)
    effects=np.zeros((len(t),len(g)),dtype='float32');model_hash=None
    if checkpoint:
        cp=Path(checkpoint);model_hash=file_hash(cp)
        with np.load(cp,allow_pickle=False) as z:
            if z['genes'].tolist()!=g or z['targets'].tolist()!=t:raise ValueError('Checkpoint axis mismatch')
            effects=z['log_fold_change']
        if effects.shape!=(len(t),len(g)) or not np.isfinite(effects).all():raise ValueError('Invalid checkpoint')
    out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix(out.suffix+'.partial')
    string=h5py.string_dtype('utf-8');row=0;nnz=0;control_pins={};context_counts={}
    label='V1_NORMALIZED_LOG_RATIO_CONTROL_TRANSPORT_UNVALIDATED' if checkpoint else 'V0_REAL_OFFICIAL_CONTROL_RESAMPLING_NO_EFFECT'
    with h5py.File(temp,'x') as f:
        f.attrs.update({'encoding-type':'anndata','encoding-version':'0.1.0'})
        x=f.create_group('X');x.attrs.update({'encoding-type':'csr_matrix','encoding-version':'0.1.0','shape':(expected,len(g))})
        kw=dict(maxshape=(None,),chunks=(262144,),compression='gzip',compression_opts=1,shuffle=True)
        data=x.create_dataset('data',shape=(0,),dtype='float32',**kw);idx=x.create_dataset('indices',shape=(0,),dtype='int32',**kw)
        ptr=x.create_dataset('indptr',shape=(expected+1,),dtype='int64');ptr[0]=0
        obs=f.create_group('obs');obs.attrs.update({'encoding-type':'dataframe','encoding-version':'0.2.0','_index':'_index','column-order':np.array(['target_gene','context'],dtype=object)})
        columns={}
        for key in ('_index','target_gene','context'):
            columns[key]=obs.create_dataset(key,shape=(expected,),dtype=string)
            columns[key].attrs.update({'encoding-type':'string-array','encoding-version':'0.2.0'})
        var=f.create_group('var');var.attrs.update({'encoding-type':'dataframe','encoding-version':'0.2.0','_index':'_index','column-order':np.array([],dtype=string)})
        axis=var.create_dataset('_index',data=g,dtype=string);axis.attrs.update({'encoding-type':'string-array','encoding-version':'0.2.0'})
        for key in ('obsm','varm','obsp','varp','layers','uns'):
            group=f.create_group(key);group.attrs.update({'encoding-type':'dict','encoding-version':'0.1.0'})
        desc=f['uns'].create_dataset('artifact_label',data=label,dtype=string);desc.attrs.update({'encoding-type':'string','encoding-version':'0.2.0'})
        for c in contexts:
            source=root/('context_'+c+'.h5ad');control_pins[str(source)]=file_hash(source)
            a=ad.read_h5ad(source)
            if a.var_names.tolist()!=g or set(a.obs[m['context_col']].astype(str))!={c} or set(a.obs[m['pert_col']].astype(str))!={m['control_label']}:raise ValueError('Control axis/labels differ')
            base=a.X.tocsr();context_nnz=0
            for j,target in enumerate(t):
                key=int.from_bytes(hashlib.sha256(f'{seed}:{c}:{target}'.encode()).digest()[:8],'big')
                sample=base[np.random.default_rng(key).integers(len(a),size=n)]
                block=transport(sample,effects[j],np.random.default_rng(key^0xDECAFBAD))
                end=nnz+block.nnz
                if end>max_nnz:raise ValueError('Stored-entry cap exceeded')
                data.resize((end,));idx.resize((end,));data[nnz:end]=block.data;idx[nnz:end]=block.indices
                ptr[row+1:row+n+1]=block.indptr[1:]+nnz
                columns['_index'][row:row+n]=[str(i) for i in range(row,row+n)]
                columns['target_gene'][row:row+n]=[target]*n;columns['context'][row:row+n]=[c]*n
                row+=n;nnz=end;context_nnz+=block.nnz
            context_counts[c]=context_nnz;f.flush();print(json.dumps({'context_complete':c,'rows':row,'nnz':nnz}),flush=True)
            del a,base
    os.replace(temp,out)
    receipt={'status':'PREDICTION_GENERATED_NOT_SCORED','at':utc_now(),'label':label,'shape':[row,len(g)],'nnz':nnz,'context_nnz':context_counts,
        'model_sha256':model_hash,'control_sha256':control_pins,'manifest_sha256':file_hash(manifest),'genes_sha256':file_hash(genes),'perts_sha256':file_hash(perts),
        'prediction_sha256':file_hash(out),'code_sha256':file_hash(__file__),'seed':seed,'wall_seconds':time.monotonic()-start,
        'submitted':False,'official_response_access':False,'inference':'Multiplicative source-normalized mean effect on resampled raw official controls, stochastic rounding; no total-depth restoration.',
        'limitations':['New transport variant; no official score or demonstrated benefit.','Zero control entries stay zero; rare activation and covariance changes are not modeled.',
            'Unsupported target/gene effects are exactly zero; no extra sparsification.']}
    write_json(out.parent,out.name+'.receipt.json',receipt,overwrite=False)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--controls',required=True);p.add_argument('--manifest',required=True);p.add_argument('--genes',required=True)
    p.add_argument('--perts',required=True);p.add_argument('--checkpoint');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=17)
    print(json.dumps(predict(**vars(p.parse_args())),indent=2))

if __name__=='__main__':main()
