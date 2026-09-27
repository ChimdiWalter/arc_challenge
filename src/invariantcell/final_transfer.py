"""Portable original diffusion/B1 context transfer; candidate preparation only.

Query accepts control counts and fitted state, never held perturbation responses.
No file readers for raw data, final-selection decision, or confirmation gate live here.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .development_challengers import (ALLOWED_CONTEXTS, SHRINKAGE, counts, fit_transform,
    transform, pairwise_squared, interpolation_weights, fingerprint)
from .io import digest, file_hash

RECIPES=('diffusion_manifold_proxy','B1:shrink=0.25')
ARRAY_NAMES={'columns','center','scale','training_points','phi','eigenvalues','bandwidth',
    'context_features','context_offsets','target_deltas','training_control_means','target_gene_columns'}


def _axis(values,name):
    if not isinstance(values,(list,tuple)) or not values or any(not isinstance(x,str) or not x for x in values) or len(set(values))!=len(values):
        raise ValueError('Unique explicit '+name+' required')


def fit_transfer(training_groups,targets,genes,target_gene_columns,recipe='diffusion_manifold_proxy'):
    if recipe not in RECIPES: raise ValueError('Unknown transfer recipe')
    contexts=sorted(training_groups)
    if len(contexts) not in (2,3) or any(c not in ALLOWED_CONTEXTS for c in contexts):
        raise ValueError('Only two or three canonical development contexts may fit')
    _axis(targets,'targets');_axis(genes,'genes')
    indices=np.asarray(target_gene_columns)
    if indices.shape!=(len(targets),) or indices.dtype.kind not in 'iu' or (indices<0).any() or (indices>=len(genes)).any() or len(set(indices.tolist()))!=len(indices):
        raise ValueError('Unique exact target gene columns required')
    controls=[];deltas=[];cell_counts={}
    for context in contexts:
        groups=training_groups[context]
        if set(groups)!={'control',*targets}:raise ValueError('Training groups differ from frozen targets')
        x=counts(groups['control'])
        if x.shape[1]!=len(genes):raise ValueError('Control gene width mismatch')
        controls.append(x);response=[];cell_counts[context]={'control':len(x)}
        for target in targets:
            y=counts(groups[target])
            if y.shape[1]!=len(genes):raise ValueError('Response gene width mismatch')
            response.append(y.mean(axis=0)-x.mean(axis=0));cell_counts[context][target]=len(y)
        deltas.append(np.stack(response))
    fitted=fit_transform(controls)
    points=[transform(x,fitted) for x in controls];x=np.vstack(points)
    offsets=np.cumsum([0]+[len(v) for v in points])
    phi=np.empty((len(x),0));lambdas=np.empty(0);bandwidth=1.;features=np.zeros((len(contexts),1))
    if recipe=='diffusion_manifold_proxy':
        # Same operations/order as development_challengers.diffusion_features.
        d2=pairwise_squared(x);positive=d2[d2>1e-12]
        bandwidth=float(np.median(positive)) if len(positive) else 1.
        kernel=np.exp(-d2/(2*bandwidth));degree=kernel.sum(axis=1)
        symmetric=kernel/np.sqrt(degree[:,None]*degree[None,:])
        values,vectors=np.linalg.eigh(symmetric);order=np.argsort(values)[::-1]
        order=order[1:1+min(8,len(order)-1)];order=order[values[order]>1e-8]
        phi=vectors[:,order]/np.sqrt(degree[:,None]);lambdas=values[order]
        if len(order):features=np.stack([phi[a:b].mean(axis=0) for a,b in zip(offsets[:-1],offsets[1:])])
    arrays={'columns':fitted['columns'],'center':fitted['center'],'scale':fitted['scale'],
        'training_points':x,'phi':phi,'eigenvalues':lambdas,'bandwidth':np.array(bandwidth),
        'context_features':features,'context_offsets':offsets,'target_deltas':np.stack(deltas),
        'training_control_means':np.stack([v.mean(axis=0) for v in controls]),'target_gene_columns':indices.astype(np.int64)}
    metadata={'schema_version':1,'status':'CANDIDATE_FITTED_NOT_PROMOTED','recipe':recipe,'numpy_version':np.__version__,
        'contexts':contexts,'targets':list(targets),'gene_ids':list(genes),'genes_sha256':digest(list(genes)),
        'targets_sha256':digest(list(targets)),'shrinkage':SHRINKAGE,'cell_counts':cell_counts,
        'training_control_sha256':fitted['training_control_sha256'],'training_points_sha256':fingerprint(x),
        'fit_scope':'Only canonical development groups; transforms/eigensystem fit on training controls only',
        'query_scope':'Held controls only; no held-response argument or raw-file reader',
        'promoted':False,'frozen_for_confirmation':False,
        'native_uncertainty':'Gene mean of source-weighted variance of log1p(clipped held-control plus each shrunken source delta); descriptive, not calibrated coverage',
        'method_scope':'Original Gaussian diffusion/Nystrom transfer or uniform B1; no ToPoMAN reproduction or causal claim'}
    validate_fitted(metadata,arrays)
    return metadata,arrays


def validate_fitted(metadata,arrays):
    if metadata.get('schema_version')!=1 or metadata.get('recipe') not in RECIPES or metadata.get('shrinkage')!=.25:
        raise ValueError('Unregistered transfer state')
    genes=metadata['gene_ids'];targets=metadata['targets'];contexts=metadata['contexts']
    _axis(genes,'genes');_axis(targets,'targets');_axis(contexts,'contexts')
    if contexts!=sorted(contexts) or len(contexts) not in (2,3) or any(c not in ALLOWED_CONTEXTS for c in contexts):raise ValueError('Invalid fitted contexts')
    if digest(genes)!=metadata.get('genes_sha256') or digest(targets)!=metadata.get('targets_sha256'):raise ValueError('Axis metadata changed')
    if set(arrays)!=ARRAY_NAMES:raise ValueError('Incomplete fitted arrays')
    for name,array in arrays.items():
        if not isinstance(array,np.ndarray) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():raise ValueError('Nonfinite/unsafe fitted array '+name)
    c,t,g=len(contexts),len(targets),len(genes);cols=arrays['columns'];p=len(cols)
    if cols.ndim!=1 or cols.dtype.kind not in 'iu' or not 1<=p<=min(256,g) or len(set(cols.tolist()))!=p or (cols<0).any() or (cols>=g).any():raise ValueError('Invalid fitted gene columns')
    x=arrays['training_points'];n=x.shape[0];rank=len(arrays['eigenvalues'])
    shapes={'center':(p,),'scale':(p,),'training_points':(n,p),'phi':(n,rank),
        'eigenvalues':(rank,),'bandwidth':(),'context_features':(c,max(1,rank)),
        'context_offsets':(c+1,),'target_deltas':(c,t,g),'training_control_means':(c,g),'target_gene_columns':(t,)}
    if any(arrays[k].shape!=shape for k,shape in shapes.items()):raise ValueError('Fitted array dimensions disagree')
    if (arrays['scale']<=0).any() or float(arrays['bandwidth'])<=0 or (arrays['eigenvalues']<=1e-8).any() or rank>min(8,n-1):raise ValueError('Invalid diffusion scaling/rank')
    offsets=arrays['context_offsets'];idx=arrays['target_gene_columns']
    if offsets.dtype.kind not in 'iu' or offsets[0]!=0 or offsets[-1]!=n or (np.diff(offsets)<=0).any():raise ValueError('Invalid context point offsets')
    if idx.dtype.kind not in 'iu' or (idx<0).any() or (idx>=g).any() or len(set(idx.tolist()))!=t:raise ValueError('Invalid target gene mapping')
    if (arrays['training_control_means']<0).any():raise ValueError('Invalid training control means')
    if fingerprint(x)!=metadata.get('training_points_sha256'):raise ValueError('Training point identity mismatch')
    if metadata['recipe']=='B1:shrink=0.25' and rank:raise ValueError('B1 cannot contain an active diffusion eigensystem')
    return True


def query_transfer(held_controls,metadata,arrays):
    """Predict a full target/gene panel using supplied held controls only."""
    validate_fitted(metadata,arrays)
    held=counts(held_controls)
    if held.shape[1]!=len(metadata['gene_ids']):raise ValueError('Held-control gene width mismatch')
    if metadata['recipe']=='B1:shrink=0.25':
        weights=np.full(len(metadata['contexts']),1/len(metadata['contexts']));fallback=False;query_feature=np.zeros(1)
    else:
        if len(arrays['eigenvalues']):
            points=transform(held,arrays)
            log_affinity=-pairwise_squared(points,arrays['training_points'])/(2*float(arrays['bandwidth']))
            affinity=np.exp(log_affinity-log_affinity.max(axis=1,keepdims=True))
            extension=(affinity/affinity.sum(axis=1,keepdims=True))@arrays['phi']/arrays['eigenvalues']
            query_feature=extension.mean(axis=0)
        else:query_feature=np.zeros(1)
        weights,fallback=interpolation_weights(list(arrays['context_features']),query_feature)
    base=held.mean(axis=0);deltas=arrays['target_deltas']
    if metadata['recipe']=='B1:shrink=0.25':prediction=np.maximum(0,base+.25*deltas.mean(axis=0))
    else:prediction=np.maximum(0,base+.25*np.tensordot(weights,deltas,axes=(0,0)))
    alternatives=np.log1p(np.maximum(0,base[None,None,:]+.25*deltas))
    average=np.tensordot(weights,alternatives,axes=(0,0))
    variance=np.tensordot(weights,(alternatives-average[None,:,:])**2,axes=(0,0)).mean(axis=1)
    effect=np.mean((np.log1p(prediction)-np.log1p(base))**2,axis=1)
    distance=float(np.mean((np.log1p(base)-np.log1p(arrays['training_control_means']).mean(axis=0))**2))
    idx=arrays['target_gene_columns'];detection=(held[:,idx]>0).mean(axis=0);abundance=np.log1p(base[idx])
    features=[{'native_uncertainty':float(variance[i]),'control_distance':distance,'effect_size':float(effect[i]),
        'target_detection_fraction':float(detection[i]),'target_control_log_mean':float(abundance[i])} for i in range(len(metadata['targets']))]
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
    return prediction,features,{'source_weights':weights.tolist(),'uniform_weight_fallback':fallback,
        'held_control_feature':query_feature.tolist(),'held_response_access':False,'held_controls_fit_eigensystem':False,
        'recipe':metadata['recipe'],'training_contexts':metadata['contexts'],'shrinkage':.25}


def save_fitted(directory,metadata,arrays):
    validate_fitted(metadata,arrays);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    with (directory/'fitted.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    body={**metadata,'arrays_file':'fitted.npz','arrays_sha256':file_hash(directory/'fitted.npz'),
        'array_shapes':{k:list(v.shape) for k,v in arrays.items()}}
    result={**body,'artifact_hash':digest(body)}
    with (directory/'fitted.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return result


def load_fitted(directory):
    directory=Path(directory);metadata=json.loads((directory/'fitted.json').read_text())
    body={k:v for k,v in metadata.items() if k!='artifact_hash'}
    if digest(body)!=metadata.get('artifact_hash') or metadata.get('arrays_file')!='fitted.npz' or file_hash(directory/'fitted.npz')!=metadata.get('arrays_sha256'):raise ValueError('Fitted artifact changed')
    with np.load(directory/'fitted.npz',allow_pickle=False) as bundle:arrays={k:bundle[k] for k in bundle.files}
    validate_fitted(metadata,arrays)
    if metadata['array_shapes']!={k:list(v.shape) for k,v in arrays.items()}:raise ValueError('Array shape manifest differs')
    return metadata,arrays
