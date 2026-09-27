"""Original development-only context transfer features; no causal guarantees.

Every challenger changes only the two-context interpolation of target deltas.
This is not a reproduction of GOPA attention, ToPoMAN, or CC-SPR's solver.
"""
from __future__ import annotations
import hashlib
import itertools
import numpy as np

ALLOWED_CONTEXTS = frozenset({'K562', 'RPE1', 'HepG2'})
COMPONENTS = ('geometry', 'diffusion_manifold_proxy', 'topology_h0',
              'ccspr_f2_cycle_proxy', 'gopa_pathway_proxy')
SHRINKAGE = 0.25


def counts(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('nonempty finite nonnegative count matrix required')
    if np.any(x != np.floor(x)):
        raise ValueError('raw integer counts required')
    return x


def fingerprint(array):
    x = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(str(x.shape).encode() + x.tobytes()).hexdigest()


def fit_transform(training_controls, gene_cap=256):
    """Gene selection/centering/scaling see training controls only."""
    x = np.log1p(np.vstack([counts(v) for v in training_controls]))
    variance = x.var(axis=0)
    columns = np.argsort(-variance, kind='stable')[:min(gene_cap, x.shape[1])]
    center = x[:, columns].mean(axis=0)
    scale = np.maximum(x[:, columns].std(axis=0), 1e-6)
    return {'columns': columns, 'center': center, 'scale': scale,
            'training_control_sha256': fingerprint(x)}


def transform(values, fitted):
    return (np.log1p(counts(values))[:, fitted['columns']] - fitted['center']) / fitted['scale']


def pairwise_squared(x, y=None):
    y = x if y is None else y
    return np.maximum(0.0, np.sum(x*x,axis=1)[:,None] + np.sum(y*y,axis=1)[None,:] - 2*x@y.T)


def h0_signature(points, quantiles=16):
    """Finite H0 Rips deaths via an exact dense Prim MST, retaining zero edges."""
    d = np.sqrt(pairwise_squared(points));n=len(d)
    if n < 2: raise ValueError('H0 signature needs at least two cells')
    used=np.zeros(n,dtype=bool);best=np.full(n,np.inf);best[0]=0.;deaths=[]
    for step in range(n):
        j=int(np.argmin(np.where(used,np.inf,best)))
        if step:deaths.append(float(best[j]))
        used[j]=True;best=np.minimum(best,d[j])
    return np.quantile(deaths,np.linspace(0,1,quantiles))


def cycle_signature(points, landmark_cap=16):
    """Exact finite F2 H1 reduction + greedy homologous cycle shortening.

    This CC-SPR-derived proxy uses a fixed bounded Rips complex and symmetric
    difference by earlier triangle boundaries. It is not the original real-field
    convex solver, a global minimum, or a biological invariant.
    """
    points=points[np.linspace(0,len(points)-1,min(landmark_cap,len(points)),dtype=int)]
    d=np.sqrt(pairwise_squared(points));n=len(points)
    simplices=[(0.,(i,)) for i in range(n)]
    simplices += [(float(d[i,j]),(i,j)) for i,j in itertools.combinations(range(n),2)]
    simplices += [(max(float(d[i,j]),float(d[i,k]),float(d[j,k])),(i,j,k))
                  for i,j,k in itertools.combinations(range(n),3)]
    simplices.sort(key=lambda s:(s[0],len(s[1]),s[1]))
    index={s:i for i,(_,s) in enumerate(simplices)}
    reduced={};pairs=[]
    for death,(value,simplex) in enumerate(simplices):
        column=set() if len(simplex)==1 else {index[simplex[:i]+simplex[i+1:]] for i in range(len(simplex))}
        while column and max(column) in reduced:
            column ^= reduced[max(column)]
        if column:
            birth=max(column);reduced[birth]=column.copy()
            if len(simplex)==3 and len(simplices[birth][1])==2 and value>simplices[birth][0]+1e-12:
                pairs.append((value-simplices[birth][0],simplices[birth][0],value,column.copy()))
    pairs.sort(key=lambda row:(-row[0],row[1],row[2]))
    features=np.zeros(9);details=[]
    for k,(life,birth,death,cycle) in enumerate(pairs[:3]):
        # Reduced death boundary is an actual F2 cycle homologous to the birth.
        cutoff=(birth+death)/2
        triangles=[]
        for value,s in simplices:
            if len(s)==3 and value < cutoff:
                triangles.append({index[s[:i]+s[i+1:]] for i in range(3)})
        initial=len(cycle)
        for _ in range(100):
            proposals=[cycle ^ boundary for boundary in triangles]
            better=[c for c in proposals if len(c)<len(cycle)]
            if not better:break
            cycle=min(better,key=lambda c:(len(c),tuple(sorted(c))))
        vertex_boundary=set()
        for edge in cycle:
            for vertex in simplices[edge][1]:
                vertex_boundary.symmetric_difference_update({vertex})
        if vertex_boundary or not cycle:raise ValueError('cycle invariant failed')
        features[3*k:3*k+3]=[life,len(cycle)/max(1,n),len(cycle)/max(1,initial)]
        details.append({'birth':birth,'death':death,'initial_edges':initial,'sparse_edges':len(cycle)})
    return features,{'finite_h1_count':len(pairs),'landmarks':n,'representatives':details,
                     'coefficient_field':'F2','optimization':'greedy triangle-boundary flips; local only'}


def diffusion_features(training_points, held_points, dimensions=8):
    """Original Gaussian diffusion coordinates with train-only Nyström extension."""
    x=np.vstack(training_points);d2=pairwise_squared(x)
    positive=d2[d2>1e-12];bandwidth=float(np.median(positive)) if len(positive) else 1.
    kernel=np.exp(-d2/(2*bandwidth));degree=kernel.sum(axis=1)
    symmetric=kernel/np.sqrt(degree[:,None]*degree[None,:])
    values,vectors=np.linalg.eigh(symmetric);order=np.argsort(values)[::-1]
    order=order[1:1+min(dimensions,len(order)-1)];order=order[values[order]>1e-8]
    phi=vectors[:,order]/np.sqrt(degree[:,None]);lambdas=values[order]
    if not len(order):return [np.zeros(1) for _ in training_points],np.zeros(1),{'rank':0,'bandwidth':bandwidth}
    log_affinity=-pairwise_squared(held_points,x)/(2*bandwidth)
    affinity=np.exp(log_affinity-log_affinity.max(axis=1,keepdims=True))
    # Nyström eigenfunction extension; no refitting with held-context cells.
    extension=(affinity/affinity.sum(axis=1,keepdims=True))@phi/lambdas
    offsets=np.cumsum([0]+[len(v) for v in training_points])
    features=[phi[a:b].mean(axis=0) for a,b in zip(offsets[:-1],offsets[1:])]
    return features,extension.mean(axis=0),{'rank':len(order),'bandwidth':bandwidth,
        'training_landmark_sha256':fingerprint(x),'held_context_used_in_eigensolve':False}


def pathway_matrix(manifest, genes):
    """Use only explicit positive, nonobsolete, unambiguous human memberships."""
    if manifest.get('status')!='CLEARED_POSITIVE_HUMAN_MEMBERSHIPS':
        raise ValueError('cleared positive human annotation manifest required')
    if manifest.get('genes_sha256')!=hashlib.sha256('\n'.join(genes).encode()).hexdigest():
        raise ValueError('pathway mapping differs from frozen Ensembl order')
    columns={g:i for i,g in enumerate(genes)};terms=[];vectors=[]
    for row in manifest['terms']:
        members=row['ensembl_genes']
        if row.get('taxon')!=9606 or row.get('negated') is not False or row.get('obsolete') is not False:
            raise ValueError('invalid pathway annotation semantics')
        if len(set(members))!=len(members) or not 5 <= len(members) <= 500:
            raise ValueError('pathway size or member identity invalid')
        vector=np.zeros(len(genes))
        for gene in members:
            if gene not in columns:raise ValueError('unknown pathway member')
            vector[columns[gene]]=1/len(members)
        terms.append(row['term_id']);vectors.append(vector)
    if not vectors or len(set(terms))!=len(terms):raise ValueError('nonempty unique pathway terms required')
    return np.stack(vectors,axis=1),terms


def interpolation_weights(source_features, held_feature):
    x=np.stack(source_features);q=np.asarray(held_feature)
    distances=np.sum((x-q)**2,axis=1)
    differences=pairwise_squared(x);positive=differences[differences>1e-12]
    if not len(positive):return np.full(len(x),1/len(x)),True
    scale=float(np.median(positive))
    logits=-distances/(2*scale);weights=np.exp(logits-logits.max());weights/=weights.sum()
    return weights,False


def predict_fold(component, training_groups, held_control, targets, genes, *, pathway_manifest=None):
    """No held response argument exists. Unknown targets are not silently added."""
    if component not in COMPONENTS:raise ValueError('unknown challenger')
    contexts=tuple(sorted(training_groups))
    if len(contexts) not in (2,3) or any(c not in ALLOWED_CONTEXTS for c in contexts):
        raise ValueError('two or three permitted development training contexts required')
    if not targets or len(set(targets))!=len(targets):raise ValueError('unique targets required')
    for c in contexts:
        for t in targets:
            if counts(training_groups[c][t]).shape[1]!=len(genes):raise ValueError('response gene shape mismatch')
    controls=[counts(training_groups[c]['control']) for c in contexts]
    held_control=counts(held_control)
    if held_control.shape[1]!=len(genes) or any(x.shape[1]!=len(genes) for x in controls):
        raise ValueError('gene shape mismatch')
    fitted=fit_transform(controls)
    train_points=[transform(x,fitted) for x in controls];held_points=transform(held_control,fitted)
    method_details={}
    if component=='geometry':
        features=[x.mean(axis=0) for x in train_points];held_feature=held_points.mean(axis=0)
    elif component=='diffusion_manifold_proxy':
        features,held_feature,method_details=diffusion_features(train_points,held_points)
    elif component=='topology_h0':
        features=[h0_signature(x) for x in train_points];held_feature=h0_signature(held_points)
    elif component=='ccspr_f2_cycle_proxy':
        results=[cycle_signature(x) for x in [*train_points,held_points]]
        features=[r[0] for r in results[:-1]];held_feature=results[-1][0]
        method_details={'cycle_qc':[r[1] for r in results]}
    else:
        matrix,terms=pathway_matrix(pathway_manifest,genes)
        # Membership weighting is fixed; expression centering uses training only.
        features=[np.log1p(x).mean(axis=0)@matrix for x in controls]
        held_feature=np.log1p(held_control).mean(axis=0)@matrix
        method_details={'term_ids':terms,'annotation_sha256':pathway_manifest.get('source_sha256')}
    weights,fallback=interpolation_weights(features,held_feature)
    deltas=np.stack([np.stack([counts(training_groups[c][t]).mean(axis=0)-controls[i].mean(axis=0)
                              for t in targets]) for i,c in enumerate(contexts)])
    base=held_control.mean(axis=0)
    incumbent=np.maximum(0,base+SHRINKAGE*deltas.mean(axis=0))
    candidate=np.maximum(0,base+SHRINKAGE*np.tensordot(weights,deltas,axes=(0,0)))
    if not np.isfinite(candidate).all():raise ValueError('nonfinite candidate')
    detail={'component':component,'training_contexts':contexts,'source_weights':weights.tolist(),
        'uniform_weight_fallback':fallback,'shrinkage':SHRINKAGE,
        'train_only_transform':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fitted.items()},
        'training_context_features':[x.tolist() for x in features],'held_control_feature':held_feature.tolist(),
        'method_details':method_details,'held_response_access':False}
    return candidate,incumbent,detail
