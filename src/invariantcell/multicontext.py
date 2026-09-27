"""New disjoint-target development CV; Jurkat confirmation stays closed."""
from pathlib import Path
from collections import Counter
import hashlib
import time
import uuid

from .io import load,digest,file_hash,write_json,utc_now


def freeze(project):
    import h5py
    from .real_data import column
    project=Path(project);output=project/'configs/multicontext_protocol.json'
    if output.exists():
        receipt=load(project/'state/multicontext_freeze.json')
        if file_hash(output)!=receipt['protocol_sha256']:raise ValueError('frozen protocol changed')
        return load(output)
    old=load(project/'configs/data_manifest_real.yaml');oldsplit=load(project/'configs/split_manifest_real.yaml')
    sources={c:dict(old['sources'][c]) for c in ('K562','RPE1')}
    for c in ('HepG2','Jurkat'):
        source=load(project/('state/public_download_'+c+'.json'))
        if source['status']!='VERIFIED':raise ValueError('download is not verified')
        sources[c]=source
    metadata={}
    for c,asset in sources.items():
        with h5py.File(asset['path'],'r') as f:
            gene_field='ensembl_id' if c in ('K562','RPE1') else 'gene_id'
            target_field='perturbation' if c in ('K562','RPE1') else 'gene'
            genes=column(f['var'],gene_field);symbols=column(f['var'],'gene_name')
            labels=column(f['obs'],target_field);barcodes=column(f['obs'],str(f['obs'].attrs['_index']))
            metadata[c]={'genes':genes,'symbols':symbols,'labels':labels,'barcodes':barcodes,
                'counts':Counter(labels),'control':'control' if c in ('K562','RPE1') else 'non-targeting'}
            if len(set(genes))!=len(genes) or len(set(barcodes))!=len(barcodes):raise ValueError('ambiguous identifiers')
    common=set.intersection(*(set(m['genes']) for m in metadata.values()))
    genes=sorted(common)
    symbols={}
    for m in metadata.values():
        for gene,symbol in zip(m['genes'],m['symbols']):
            if gene in common:symbols.setdefault(gene,set()).add(symbol)
    agreed={next(iter(v)) for v in symbols.values() if len(v)==1}
    eligible=set.intersection(*(set(t for t,n in m['counts'].items() if n>=32) for m in metadata.values()))
    eligible &= agreed
    # All old response targets are excluded in every context, not merely old cells.
    eligible -= set(oldsplit['targets'])
    rank=lambda value:hashlib.sha256(('InvariantCell-v8-seed17:'+str(value)).encode()).hexdigest()
    targets=sorted(eligible,key=rank)[:12]
    if len(targets)<12:raise ValueError('insufficient shared disjoint target panel')
    rows={};columns={}
    for c,m in metadata.items():
        cols={g:i for i,g in enumerate(m['genes'])};columns[c]=[cols[g] for g in genes]
        groups={}
        for target,n in [(m['control'],64)]+[(t,32) for t in targets]:
            choices=[i for i,v in enumerate(m['labels']) if v==target]
            if len(choices)<n:raise ValueError('insufficient group cells')
            groups['control' if target==m['control'] else target]=sorted(sorted(choices,key=lambda i:rank(c+':'+m['barcodes'][i]))[:n])
        rows[c]=groups
    result={'schema_version':1,'status':'FROZEN_METADATA_ONLY','created_at':utc_now(),'seed':17,
        'modality':'CRISPRi','sources':sources,'genes':genes,'gene_columns':columns,'targets':targets,'rows':rows,
        'development_contexts':['K562','RPE1','HepG2'],'confirmation_contexts':['Jurkat'],
        'development_protocol':'Rotate whole-context holdout; two training contexts per fold. Training-only context PCA.',
        'excluded_old_response_targets':oldsplit['targets'],'confirmation_outcomes_opened':False,
        'models':{'B0':{},'B1':{'shrinkage':[0.25,0.5,1.0]},'B2':{'alpha':[0.1,1.0,10.0]}},
        'primary_metric':'mean squared difference of log1p group mean raw counts, all aligned genes',
        'gene_axis_basis':'All mutually assayed Ensembl IDs, exact ordered intersection; not official 18,533-axis data.',
        'selection_basis':'Seeded metadata-only target/barcode sampling; no expression-driven gene/target selection.',
        'information_boundary':['Jurkat expression forbidden to development loader.',
            'RPE1 is previously exposed at context level; new response targets are disjoint from every old response target.',
            'Old RPE1 panel remains retired and is not used for selection, calibration or fitting here.',
            'Published source-level/literature aggregate findings were read; no claim of pristine upstream evidence.',
            'Unknown pretraining overlap prevents declaring pretrained-model confirmation clean.'],
        'future_confirmation_gate':'Freeze winning fit/config, calibration, candidate predictions and all policy selections before evaluator opens Jurkat responses.',
        'nih_prize_eligibility':'UNRESOLVED; this is Arc/shared local development'}
    write_json(project,'configs/multicontext_protocol.json',result,overwrite=False)
    write_json(project,'state/multicontext_freeze.json',{'created_at':utc_now(),'protocol_sha256':file_hash(output),
        'expression_values_read':False,'old_targets_disjoint':True,'gene_count':len(genes),'target_count':len(targets)},overwrite=False)
    return result


def development_counts(protocol,context):
    import h5py
    import numpy as np
    if context not in protocol['development_contexts'] or context in protocol['confirmation_contexts']:
        raise PermissionError('confirmation expression is inaccessible to development')
    groups=protocol['rows'][context];selected=sorted(set(i for rows in groups.values() for i in rows))
    position={r:i for i,r in enumerate(selected)}
    with h5py.File(protocol['sources'][context]['path'],'r') as f:
        values=np.asarray(f['X'][selected,:],dtype=np.float64)[:,protocol['gene_columns'][context]]
    if np.any(~np.isfinite(values)) or np.any(values<0) or np.any(values!=np.floor(values)):
        raise ValueError('accessed source values are not raw counts')
    return {target:values[[position[r] for r in rows]] for target,rows in groups.items()}


def development_run(project):
    import numpy as np
    project=Path(project);p=freeze(project);start=time.monotonic()
    run=project/'runs'/('multicontext-development-'+uuid.uuid4().hex[:10]);run.mkdir(parents=True)
    data={}
    for c in p['development_contexts']:
        data[c]=development_counts(p,c)
        write_json(project,str(run.relative_to(project)/'progress.json'),{'status':'LOADING_DEVELOPMENT','contexts_loaded':list(data),'confirmation_opened':False})
    control={c:d['control'].mean(axis=0) for c,d in data.items()}
    observed={c:{t:d[t].mean(axis=0) for t in p['targets']} for c,d in data.items()}
    targets=p['targets'];tcount=len(targets);scores=[];arrays={}
    for held in p['development_contexts']:
        training=[c for c in p['development_contexts'] if c!=held]
        profile=np.stack([np.log1p(control[c]) for c in training]);center=profile.mean(axis=0)
        _,singular,vt=np.linalg.svd(profile-center,full_matrices=False)
        rank=int(np.sum(singular>1e-10));basis=vt[:rank].T
        scales=np.maximum(singular[:rank],1e-8)
        def feature(context,index):
            z=((np.log1p(control[context])-center)@basis)/scales
            one=np.eye(tcount)[index]
            return np.concatenate([one,(one[:,None]*z).ravel()])
        x=np.stack([feature(c,i) for c in training for i in range(tcount)])
        y=np.stack([observed[c][t]-control[c] for c in training for t in targets])
        prediction={'B0':np.stack([control[held]]*tcount)}
        pooled=np.stack([np.mean([observed[c][t]-control[c] for c in training],axis=0) for t in targets])
        for strength in p['models']['B1']['shrinkage']:
            prediction['B1:shrink='+str(strength)]=np.maximum(0,control[held]+strength*pooled)
        test_x=np.stack([feature(held,i) for i in range(tcount)])
        for alpha in p['models']['B2']['alpha']:
            coef=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@y)
            prediction['B2:alpha='+str(alpha)]=np.maximum(0,control[held]+test_x@coef)
        truth=np.stack([observed[held][t] for t in targets])
        for model,pred in prediction.items():
            error=np.mean((np.log1p(pred)-np.log1p(truth))**2,axis=1)
            for i,t in enumerate(targets):scores.append({'heldout_development_context':held,'target':t,'model':model,'error':float(error[i]),'training_context_count':len(training),'context_rank':rank})
            arrays[held+'__'+model]=pred.astype('float32')
        write_json(project,str(run.relative_to(project)/'progress.json'),{'status':'DEVELOPMENT_FOLD_COMPLETE','context':held,'score_rows':len(scores),'confirmation_opened':False})
    models=sorted({r['model'] for r in scores})
    aggregate={m:float(np.mean([r['error'] for r in scores if r['model']==m])) for m in models}
    winner=min(models,key=lambda m:(aggregate[m],m))
    write_json(project,str(run.relative_to(project)/'scores.json'),{'rows':scores,'mean_development_error':aggregate,'development_winner':winner})
    np.savez_compressed(run/'development_prediction_means.npz',**arrays)
    receipt={'status':'MULTICONTEXT_DEVELOPMENT_BASELINES_WORKING','created_at':utc_now(),
        'protocol_sha256':file_hash(project/'configs/multicontext_protocol.json'),'run_path':str(run),
        'development_contexts':p['development_contexts'],'confirmation_contexts':p['confirmation_contexts'],
        'confirmation_outcomes_opened':False,'genes':len(p['genes']),'targets':len(targets),
        'mean_development_error':aggregate,'development_winner':winner,'wall_seconds':time.monotonic()-start,
        'gpu_hours':0,'paid_usd':0,'scope':'Development CV only; no fresh confirmation or official Arc score.',
        'artifact_sha256':{f.name:file_hash(f) for f in run.iterdir() if f.is_file()}}
    write_json(project,str(run.relative_to(project)/'receipt.json'),receipt)
    write_json(project,'reports/arc/multicontext_development.json',receipt)
    return receipt
