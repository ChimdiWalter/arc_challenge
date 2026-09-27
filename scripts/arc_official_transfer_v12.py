"""Freeze source rows and train a compact official-axis normalized B1-like artifact.

Only K562/RPE1/HepG2 expression is admitted. Jurkat is metadata-only and excluded
from every fit. This is new development, not fresh confirmation or official scoring.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'src'))
from invariantcell.real_data import column

DEV=('K562','RPE1','HepG2')
CONFIG='configs/arc_official_transfer_v12.json'
ARTIFACT='artifacts/arc/official_transfer_v12.npz'
RECEIPT='artifacts/arc/official_transfer_v12_receipt.json'
MAX_GROWTH=150*1024**2


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def csv_values(path,keys):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        matched=[key for key in keys if key in (reader.fieldnames or [])]
        if len(matched)!=1:raise ValueError(f'Expected one named column from {keys}, got {reader.fieldnames}')
        rows=list(reader)
    values=[str(row[matched[0]]) for row in rows]
    if not values or any(not v or v!=v.strip() for v in values) or len(values)!=len(set(values)):
        raise ValueError('Official identifiers must be nonempty, unique and unmodified')
    return values,rows


def source_metadata(context,asset):
    import h5py
    if context not in (*DEV,'Jurkat'):raise ValueError('Unknown source context')
    path=Path(asset['path'])
    if path.stat().st_size!=asset['bytes']:raise ValueError('Source size differs from manifest')
    with h5py.File(path,'r') as h:
        genes=column(h['var'],'ensembl_id' if context in ('K562','RPE1') else 'gene_id')
        symbols=column(h['var'],'gene_name')
        labels=column(h['obs'],'perturbation' if context in ('K562','RPE1') else 'gene')
        barcodes=column(h['obs'],str(h['obs'].attrs['_index'])) if context in DEV else []
        # X values are never touched in this metadata function.
    if len(genes)!=len(symbols) or len(genes)!=len(set(genes)) or any(not g.startswith('ENSG') for g in genes):
        raise ValueError('Source stable gene axis is ambiguous')
    if context in DEV and (len(barcodes)!=len(labels) or len(barcodes)!=len(set(barcodes))):
        raise ValueError('Source cell identity is ambiguous')
    groups=defaultdict(list)
    for i,label in enumerate(labels):groups[label].append(i)
    return {'genes':genes,'symbols':symbols,'labels':labels,'barcodes':barcodes,'groups':dict(groups),
            'control':'control' if context in ('K562','RPE1') else 'non-targeting'}


def gene_mapping(official,metadata):
    """Exclude non-bijective IDs/symbols globally; never merge aliases or columns."""
    symbol_ids=defaultdict(set);id_symbols=defaultdict(set)
    for c in DEV:
        for gene,symbol in zip(metadata[c]['genes'],metadata[c]['symbols']):
            if symbol:symbol_ids[symbol].add(gene);id_symbols[gene].add(symbol)
    stable=[];reasons=[];columns={c:[] for c in DEV}
    indexes={c:{g:i for i,g in enumerate(metadata[c]['genes'])} for c in DEV}
    frequencies={c:Counter(metadata[c]['symbols']) for c in DEV}
    for symbol in official:
        candidates=symbol_ids.get(symbol,set())
        gene=next(iter(candidates)) if len(candidates)==1 else None
        ambiguous=gene is not None and (len(id_symbols[gene])!=1 or any(frequencies[c][symbol]>1 for c in DEV))
        if gene is None or ambiguous:
            stable.append('');reasons.append('ambiguous_mapping' if candidates else 'not_measured_in_training_sources')
            for c in DEV:columns[c].append(-1)
        else:
            stable.append(gene);reasons.append('unique_stable_id_symbol_mapping')
            for c in DEV:columns[c].append(indexes[c].get(gene,-1))
    return stable,columns,reasons


def sample_rows(meta,context,target,n,excluded=()):
    blocked=set(excluded)
    available=meta['groups'].get(target,[]) if 'groups' in meta else [i for i,label in enumerate(meta['labels']) if label==target]
    choices=[i for i in available if i not in blocked]
    rank=lambda i:hashlib.sha256(f"17:{context}:{target}:{meta['barcodes'][i]}".encode()).hexdigest()
    return sorted(sorted(choices,key=rank)[:n])


def freeze(project,official_dir):
    project=Path(project);destination=project/CONFIG
    if destination.exists():raise FileExistsError('Official transfer protocol already frozen')
    base_path=project/'configs/multicontext_protocol.json';base=json.loads(base_path.read_text())
    genes,gene_rows=csv_values(Path(official_dir)/'gene_names.csv',('gene_name','gene','gene_names','genes','feature','index'))
    targets,target_rows=csv_values(Path(official_dir)/'pert_counts.csv',('target_gene','perturbation','gene','gene_name','target','pert'))
    if len(targets)*len(genes)*5>140*1024**2:raise ValueError('Artifact axes exceed preregistered growth estimate')
    metadata={c:source_metadata(c,base['sources'][c]) for c in DEV}
    # Jurkat labels/var only. Its entire response-bearing container is NOT rehashed.
    jurkat=source_metadata('Jurkat',base['sources']['Jurkat']) if 'Jurkat' in base['sources'] else None
    stable,columns,reasons=gene_mapping(genes,metadata)
    old_path=project/'configs/split_manifest_real.yaml'
    old=json.loads(old_path.read_text()) if old_path.exists() else {'rows':{}}
    excluded={c:set() for c in DEV}
    for c in DEV:
        for rows in old.get('rows',{}).get(c,{}).get('evaluation',{}).values():excluded[c].update(rows)
    rows={};sources={}
    for c in DEV:
        source=dict(base['sources'][c]);path=Path(source['path'])
        if 'jurkat' in path.name.lower():raise PermissionError('Jurkat expression source is forbidden')
        actual=file_hash(path)
        if actual!=source['sha256']:raise ValueError('Source SHA256 differs from manifest: '+c)
        sources[c]={'path':str(path.resolve()),'sha256':actual,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        m=metadata[c];selected={'control':sample_rows(m,c,m['control'],64,excluded[c])}
        if len(selected['control'])!=64:raise ValueError('Need 64 permitted source controls: '+c)
        for target in targets:
            chosen=sample_rows(m,c,target,32,excluded[c])
            if chosen:selected[target]=chosen
        rows[c]=selected
    cfg={'schema_version':1,'status':'FROZEN_BEFORE_RESPONSE_VALUE_ACCESS','boundary':'official_arc_v12_new_development',
         'training_contexts':list(DEV),'Jurkat_training':False,'official_response_access':False,
         'seed':17,'controls_per_context':64,'max_response_cells_per_target_context':32,
         'transform':{'target_sum':10000.0,'normalization_axis':'all measured source genes per cell',
                      'effect':'log((mean(normalized response)+1)/(mean(normalized controls)+1))',
                      'pseudocount':1.0,'shrinkage':0.25,'pooling':'equal available context/gene weights','clip':None,
                      'unknown_gene_effect':0.0,'unsupported_target_effect':0.0,'global_depth_restoration':False},
         'genes':genes,'targets':targets,'stable_ids':stable,'gene_columns':columns,'mapping_reason':reasons,
         'rows':rows,'sources':sources,'source_gene_axis_sha256':{c:digest(metadata[c]['genes']) for c in DEV},
         'row_label_sha256':{c:digest(metadata[c]['labels']) for c in DEV},
         'input_sha256':{str(base_path.relative_to(project)):file_hash(base_path),
                         str((Path(official_dir)/'gene_names.csv').resolve().relative_to(project.resolve())):file_hash(Path(official_dir)/'gene_names.csv'),
                         str((Path(official_dir)/'pert_counts.csv').resolve().relative_to(project.resolve())):file_hash(Path(official_dir)/'pert_counts.csv')},
         'retired_evaluation_rows_excluded':{c:sorted(excluded[c]) for c in DEV},
         'limits':{'cpu_threads':1,'ram_gib':8,'wall_seconds':1200,'growth_bytes':MAX_GROWTH,'download_bytes':0,'gpu_hours':0},
         'limitations':['New normalized-log-ratio variant with fixed archived B1 shrink0.25, not numerically identical to raw-delta B1.',
                        'Source measured axes differ; within-source normalization does not prove complete transcriptome calibration.',
                        'Sources and public targets previously exposed; no fresh confirmation, novel-target generalization or official score.',
                        'Old archived evaluation rows excluded; Jurkat excluded from all fitting.']}
    if old_path.exists():cfg['input_sha256'][str(old_path.relative_to(project))]=file_hash(old_path)
    code=project/'scripts/arc_official_transfer_v12.py'
    if code.exists():cfg['input_sha256']['scripts/arc_official_transfer_v12.py']=file_hash(code)
    save(destination,cfg)
    out=project/'reports/arc';out.mkdir(parents=True,exist_ok=True)
    counts={c:Counter(metadata[c]['labels']) for c in DEV};jurkat_counts=Counter(jurkat['labels']) if jurkat else {}
    with (out/'target_coverage_audit.csv').open('x',newline='') as f:
        fields=['target','requested_cells',*[c+'_available_cells' for c in DEV],*[c+'_selected_cells' for c in DEV],
                'Jurkat_available_metadata_only','Jurkat_used_for_training','support_context_count','fallback']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for target,entry in zip(targets,target_rows):
            support=sum(target in rows[c] for c in DEV)
            item={'target':target,'requested_cells':next((entry[k] for k in ('count','n_cells','cell_count','num_cells') if k in entry),'UNSPECIFIED'),
                  'Jurkat_available_metadata_only':jurkat_counts.get(target,0) if jurkat else 'NOT_READ',
                  'Jurkat_used_for_training':False,'support_context_count':support,'fallback':'control' if not support else 'modeled_genes_only'}
            item.update({c+'_available_cells':counts[c][target] for c in DEV})
            item.update({c+'_selected_cells':len(rows[c].get(target,[])) for c in DEV});writer.writerow(item)
    with (out/'full_gene_biological_coverage.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['official_index','gene_symbol','stable_id',*[c+'_column' for c in DEV],'training_context_count','mapping_status','unmodeled_behavior'])
        for i,gene in enumerate(genes):writer.writerow([i,gene,stable[i],*[columns[c][i] for c in DEV],sum(columns[c][i]>=0 for c in DEV),reasons[i],'exact_control_if_no_effect'])
    return cfg


def normalize_counts(block):
    import numpy as np
    x=np.asarray(block,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any():
        raise ValueError('Source block must be finite nonnegative integral raw counts')
    totals=x.sum(axis=1)
    if (totals<=0).any():raise ValueError('Source cell library totals must be positive')
    return x*(10000.0/totals[:,None])


def validate_config(project,cfg):
    if cfg.get('training_contexts')!=list(DEV) or set(cfg['sources'])!=set(DEV) or cfg.get('Jurkat_training') is not False:
        raise PermissionError('Only canonical three development sources may train')
    if cfg['transform']!={'target_sum':10000.0,'normalization_axis':'all measured source genes per cell',
        'effect':'log((mean(normalized response)+1)/(mean(normalized controls)+1))','pseudocount':1.0,
        'shrinkage':0.25,'pooling':'equal available context/gene weights','clip':None,
        'unknown_gene_effect':0.0,'unsupported_target_effect':0.0,'global_depth_restoration':False}:raise ValueError('Frozen recipe changed')
    for name,expected in cfg['input_sha256'].items():
        if file_hash(Path(project)/name)!=expected:raise ValueError('Frozen input hash mismatch: '+name)
    for c in DEV:
        if 'jurkat' in Path(cfg['sources'][c]['path']).name.lower():raise PermissionError('Forbidden Jurkat source')
        groups=cfg['rows'][c];allrows=[r for rs in groups.values() for r in rs]
        if len(set(allrows))!=len(allrows) or set(allrows)&set(cfg['retired_evaluation_rows_excluded'][c]):raise ValueError('Overlapping/excluded source rows')
        if len(groups['control'])!=64 or any(not 1<=len(rs)<=32 for t,rs in groups.items() if t!='control'):raise ValueError('Cell cap invalid')
        if any(t not in cfg['targets'] for t in groups if t!='control'):raise ValueError('Unregistered target')
    return cfg


def source_means(cfg,context):
    import h5py
    import numpy as np
    if context not in DEV:raise PermissionError('Jurkat responses may never be read')
    source=cfg['sources'][context];path=Path(source['path'])
    if file_hash(path)!=source['sha256']:raise ValueError('Source SHA256 mismatch before response access')
    meta=source_metadata(context,source)
    if digest(meta['genes'])!=cfg['source_gene_axis_sha256'][context] or digest(meta['labels'])!=cfg['row_label_sha256'][context]:
        raise ValueError('Frozen metadata differs')
    for i,col in enumerate(cfg['gene_columns'][context]):
        if col>=0 and (col>=len(meta['genes']) or meta['genes'][col]!=cfg['stable_ids'][i] or meta['symbols'][col]!=cfg['genes'][i]):
            raise ValueError('Frozen official/stable-ID source mapping differs')
    names=['control',*cfg['targets']];group_index={t:i for i,t in enumerate(names)}
    assignments={r:group_index[t] for t,rs in cfg['rows'][context].items() for r in rs}
    for t,rs in cfg['rows'][context].items():
        label=meta['control'] if t=='control' else t
        if any(meta['labels'][r]!=label for r in rs):raise ValueError('Frozen response membership differs')
    sums=np.zeros((len(names),len(meta['genes'])),dtype=np.float64);n=np.zeros(len(names),dtype=np.int64)
    with h5py.File(path,'r') as h:
        x=h['X']
        if not isinstance(x,h5py.Dataset):raise ValueError('Only verified dense source X is supported')
        width=x.chunks[0] if x.chunks else 256
        buckets=defaultdict(list)
        for row in sorted(assignments):buckets[row//width].append(row)
        for indices in buckets.values():
            values=normalize_counts(x[indices,:]);labels=np.array([assignments[r] for r in indices])
            np.add.at(sums,labels,values);np.add.at(n,labels,1)
    sums/=np.maximum(n,1)[:,None]
    return sums,n


def train(project,expected_protocol_sha256):
    import numpy as np
    project=Path(project)
    if file_hash(project/CONFIG)!=expected_protocol_sha256:raise ValueError('Frozen protocol SHA256 does not match explicit training commitment')
    cfg=validate_config(project,json.loads((project/CONFIG).read_text()))
    output=project/ARTIFACT;receipt_path=project/RECEIPT
    if output.exists() or receipt_path.exists():raise FileExistsError('Training output/attempt already exists; no overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();receipt={'status':'STARTED','protocol_sha256':file_hash(project/CONFIG),'training_contexts':list(DEV),'Jurkat_X_opened':False,'official_response_access':False}
    save(receipt_path,receipt)
    try:
        t,g=len(cfg['targets']),len(cfg['genes']);total=np.zeros((t,g),dtype=np.float64);support=np.zeros((t,g),dtype=np.uint8)
        context_support=np.zeros(t,dtype=np.uint8);selected={}
        for c in DEV:
            means,n=source_means(cfg,c);columns=np.asarray(cfg['gene_columns'][c]);good=np.flatnonzero(columns>=0)
            selected[c]={'controls':int(n[0]),'responses':int(n[1:].sum()),'targets':int((n[1:]>0).sum())}
            for i in np.flatnonzero(n[1:]>0):
                effect=np.log((means[i+1,columns[good]]+1.0)/(means[0,columns[good]]+1.0))
                total[i,good]+=effect;support[i,good]+=1;context_support[i]+=1
        logfc=(0.25*total/np.maximum(support,1)).astype(np.float32)
        if not np.isfinite(logfc).all() or (logfc[support==0]!=0).any():raise ValueError('Invalid or unsupported nonzero effect')
        per_gene=np.array([sum(cfg['gene_columns'][c][i]>=0 for c in DEV) for i in range(g)],dtype=np.uint8)
        arrays={'targets':np.asarray(cfg['targets'],dtype=str),'genes':np.asarray(cfg['genes'],dtype=str),
                'log_fold_change':logfc,'support_context_count':context_support,'modeled_gene':per_gene>0,
                'per_target_supported':context_support>0,'per_gene_context_count':per_gene,
                'per_target_gene_context_count':support}
        with output.open('xb') as f:np.savez_compressed(f,**arrays)
        if output.stat().st_size>MAX_GROWTH:raise ValueError('Artifact exceeds growth cap')
        receipt.update(status='TRAINED_NEW_DEVELOPMENT_NOT_VALIDATED',artifact=str(output.relative_to(project)),artifact_sha256=file_hash(output),
                       artifact_bytes=output.stat().st_size,genes=g,targets=t,modeled_genes=int((per_gene>0).sum()),
                       supported_targets=int((context_support>0).sum()),unsupported_targets=[cfg['targets'][i] for i in np.flatnonzero(context_support==0)],
                       selected=selected,arrays={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
                       log_fold_change_min=float(logfc.min()),log_fold_change_max=float(logfc.max()),
                       source_sha256={c:cfg['sources'][c]['sha256'] for c in DEV},limits=cfg['limits'],
                       fallback='Exact zero log-fold-change at unsupported target/gene entries; root decoder preserves corresponding control counts.')
    except BaseException as error:
        receipt.update(status='FAILED_ATTEMPT_CONSUMED',error=type(error).__name__+': '+str(error));raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze',action='store_true');action.add_argument('--train',action='store_true');action.add_argument('--check',action='store_true')
    parser.add_argument('--protocol-sha256',help='Required for training; binds the prior metadata freeze')
    parser.add_argument('--official-dir',type=Path,default=P/'data/arc/official_validation_2026')
    args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    def expired(*_):raise TimeoutError('Official transfer 1200-second cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1200)
    if args.freeze:
        cfg=freeze(P,args.official_dir)
        print(json.dumps({'status':cfg['status'],'genes':len(cfg['genes']),'targets':len(cfg['targets']),'protocol_sha256':file_hash(P/CONFIG)}))
    elif args.train:
        if not args.protocol_sha256:parser.error('--train requires --protocol-sha256 from prior --freeze')
        print(json.dumps(train(P,args.protocol_sha256),indent=2))
    else:
        cfg=validate_config(P,json.loads((P/CONFIG).read_text()))
        print(json.dumps({'status':'FROZEN_METADATA_CHECKED','genes':len(cfg['genes']),'targets':len(cfg['targets']),'response_values_read':False,'protocol_sha256':file_hash(P/CONFIG)}))

if __name__=='__main__':main()
