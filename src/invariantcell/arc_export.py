"""Stream raw-count predictions into one ordered, sparse H5AD.

Phase labels, targets, genes and cell counts are supplied, never inferred from
anonymous challenge labels. Default callback is the explicit B0 fallback.
"""
from pathlib import Path
import os


def export_counts(path, *, genes, panel, controls, control_genes, seed=17,
                  batch_size=64, predictor=None, label, max_nnz=2_000_000_000):
    import h5py
    import numpy as np
    from scipy import sparse
    genes=tuple(genes)
    if not genes or len(set(genes))!=len(genes) or any(not isinstance(g,str) or not g for g in genes):
        raise ValueError('unique ordered gene identifiers required')
    if batch_size<1 or not label:raise ValueError('positive batch size and explicit artifact label required')
    if not panel or len({(p['context'],p['target']) for p in panel})!=len(panel):
        raise ValueError('nonempty unique context-target panel required')
    for group in panel:
        if not group['context'] or not group['target'] or group['target'] in {'control','non-targeting'}:
            raise ValueError('invalid context/target or submitted control group')
        if type(group['cells']) is not int or group['cells']<=0:raise ValueError('positive cell count required')
        context=group['context']
        if tuple(control_genes[context])!=genes:raise ValueError('control gene order differs from supplied reference')
        if controls[context].shape[1]!=len(genes) or controls[context].shape[0]<1:raise ValueError('control shape mismatch')
    target=Path(path)
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.partial')
    if temporary.exists():raise FileExistsError(temporary)
    rng=np.random.default_rng(seed);rows=0;nnz=0;fallbacks=[]
    expected=sum(g['cells'] for g in panel)
    string=h5py.string_dtype('utf-8')
    def array(group,name,data):
        d=group.create_dataset(name,data=data,dtype=string)
        d.attrs.update({'encoding-type':'string-array','encoding-version':'0.2.0'})
        return d
    with h5py.File(temporary,'x') as f:
        f.attrs.update({'encoding-type':'anndata','encoding-version':'0.1.0'})
        x=f.create_group('X');x.attrs.update({'encoding-type':'csr_matrix','encoding-version':'0.1.0','shape':(expected,len(genes))})
        data=x.create_dataset('data',shape=(0,),maxshape=(None,),dtype='uint32',chunks=True)
        indices=x.create_dataset('indices',shape=(0,),maxshape=(None,),dtype='int32',chunks=True)
        indptr=x.create_dataset('indptr',shape=(1,),maxshape=(None,),dtype='int64',chunks=True);indptr[0]=0
        obs=f.create_group('obs');obs.attrs.update({'encoding-type':'dataframe','encoding-version':'0.2.0','_index':'_index','column-order':np.array(['target_gene','context'],dtype=object)})
        columns={}
        for name in ('_index','target_gene','context'):
            d=obs.create_dataset(name,shape=(0,),maxshape=(None,),dtype=string,chunks=True)
            d.attrs.update({'encoding-type':'string-array','encoding-version':'0.2.0'});columns[name]=d
        var=f.create_group('var');var.attrs.update({'encoding-type':'dataframe','encoding-version':'0.2.0','_index':'_index','column-order':np.array([],dtype=string)})
        array(var,'_index',list(genes))
        for name in ('obsm','varm','obsp','varp','layers','uns'):
            g=f.create_group(name);g.attrs.update({'encoding-type':'dict','encoding-version':'0.1.0'})
        d=f['uns'].create_dataset('artifact_label',data=label,dtype=string);d.attrs.update({'encoding-type':'string','encoding-version':'0.2.0'})
        for group in panel:
            context,target_gene=group['context'],group['target']
            base=sparse.csr_matrix(controls[context])
            for start in range(0,group['cells'],batch_size):
                n=min(batch_size,group['cells']-start)
                sample=base[rng.integers(base.shape[0],size=n)]
                predicted=predictor(context,target_gene,sample) if predictor else None
                if predicted is None:
                    predicted=sample
                    if (context,target_gene) not in fallbacks:fallbacks.append((context,target_gene))
                block=sparse.csr_matrix(predicted);block.sum_duplicates();block.eliminate_zeros();block.sort_indices()
                v=block.data
                if block.shape!=(n,len(genes)) or not np.all(np.isfinite(v)) or np.any(v<0) or np.any(v!=np.floor(v)) or np.any(v>np.iinfo('uint32').max):
                    raise ValueError('prediction must be finite nonnegative integral raw counts on exact gene axis')
                if nnz+block.nnz>max_nnz:raise ValueError('stored-entry budget exceeded')
                if np.any(np.asarray(block.astype('uint64').sum(axis=1)).ravel()>1_000_000):raise ValueError('per-cell count total exceeds format cap')
                end=nnz+block.nnz
                data.resize((end,));indices.resize((end,));data[nnz:end]=v;indices[nnz:end]=block.indices
                indptr.resize((rows+n+1,));indptr[rows+1:rows+n+1]=block.indptr[1:]+nnz
                for d in columns.values():d.resize((rows+n,))
                columns['_index'][rows:rows+n]=[str(i) for i in range(rows,rows+n)]
                columns['context'][rows:rows+n]=[context]*n;columns['target_gene'][rows:rows+n]=[target_gene]*n
                rows+=n;nnz=end
        if rows!=expected:raise ValueError('cell total mismatch')
    os.replace(temporary,target)
    return {'path':str(target),'shape':[rows,len(genes)],'nnz':nnz,'seed':seed,'batch_size':batch_size,
            'label':label,'fallback_groups':fallbacks,'submission_approval':False}
