"""Fixed, source-only GWPS pseudo-bulk transport. No Jurkat or official outcomes.

Rows are guide/population aggregates, not cells. Per-row normalization is
invariant to positive sum/mean scaling. No outcome-derived metadata selects rows.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,re,resource,signal,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import h5py
P=Path(__file__).resolve().parents[1]
CONFIG='configs/arc_gwps_transfer_v12.json'
RECEIPT='artifacts/arc/gwps_transfer_v12_receipt.json'
SOURCE='data/public/gwps_bulk_v12/raw.h5ad'
EXPECTED_SOURCE='7cec96b3b76169abbf6b6ab9d10bf00d71d942d89e63292351f745e130b154db'
RECIPE={'target_sum':10000.,'pseudocount':1.,'shrinkage':.25,'ntc_selection':'all non-targeting rows',
        'ntc_pool':'equal normalized-row arithmetic mean','target_pool':'equal per-guide log-ratio arithmetic mean',
        'clip':None,'cell_count_weighting':False,'global_depth_restoration':False}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def strings(values):return [x.decode() if isinstance(x,bytes) else str(x) for x in values]


def identity_column(group,name):
    node=group[name]
    if isinstance(node,h5py.Group):
        categories=strings(node['categories'][()]);codes=node['codes'][()]
    elif '__categories' in group and name in group['__categories']:
        categories=strings(group['__categories'][name][()]);codes=node[()]
    else:return strings(node[()])
    if np.asarray(codes).dtype.kind not in 'iu' or (codes<0).any() or (codes>=len(categories)).any():raise ValueError('Invalid categorical identity')
    return [categories[int(i)] for i in codes]


def parse_label(identity):
    if identity.endswith('_non-targeting_non-targeting_non-targeting'):return 'non-targeting'
    match=re.fullmatch(r'\d+_([^_]+)_.+_(ENSG\d+(?:\.\d+)?)',identity)
    return match.group(1) if match else None


def metadata(path):
    with h5py.File(path,'r') as h:
        rows=identity_column(h['obs'],str(h['obs'].attrs['_index']))
        ids=identity_column(h['var'],str(h['var'].attrs['_index']))
        symbols=identity_column(h['var'],'gene_name')
        shape=list(h['X'].shape) # Storage metadata only; X values unopened.
        cell_counts=np.asarray(h['obs']['num_cells_filtered'][()],dtype=float) if 'num_cells_filtered' in h['obs'] else None
    if len(rows)!=len(set(rows)) or len(ids)!=len(set(ids)) or shape!=[len(rows),len(ids)]:raise ValueError('Nonunique or inconsistent source identity axis')
    return {'rows':rows,'ids':ids,'symbols':symbols,'labels':[parse_label(r) for r in rows],
            'cell_counts':cell_counts,'shape':shape}


def read_names(path,header):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=[header]:raise ValueError('Official single-column axis header differs')
        values=[row[header] for row in reader]
    if not values or len(values)!=len(set(values)) or any(not x for x in values):raise ValueError('Invalid official axis')
    return values


def freeze(project,source_sha=EXPECTED_SOURCE):
    project=Path(project);path=project/SOURCE
    if (project/CONFIG).exists():raise FileExistsError('Separate GWPS protocol already frozen')
    if sha(path)!=source_sha:raise ValueError('Verified GWPS source hash changed')
    official=project/'data/arc/official_validation_2026'
    genes=read_names(official/'gene_names.csv','gene_name');targets=read_names(official/'pert_counts.csv','target_gene')
    development=json.loads((project/'configs/multicontext_protocol.json').read_text())['targets']
    meta=metadata(path);counts=Counter(meta['symbols']);index={s:i for i,s in enumerate(meta['symbols']) if counts[s]==1}
    columns=[index.get(g,-1) for g in genes];stable=[meta['ids'][i] if i>=0 else '' for i in columns]
    group_rows=defaultdict(list)
    for i,label in enumerate(meta['labels']):
        if label is not None:group_rows[label].append(i)
    ntc=group_rows['non-targeting']
    if not ntc:raise ValueError('No non-targeting source rows')
    selected={t:group_rows.get(t,[]) for t in sorted(set(targets+development))}
    inputs=['data/arc/official_validation_2026/gene_names.csv','data/arc/official_validation_2026/pert_counts.csv','configs/multicontext_protocol.json']
    script=project/'scripts/arc_gwps_transfer_v12.py'
    if script.exists():inputs.append('scripts/arc_gwps_transfer_v12.py')
    cfg={'schema_version':1,'status':'FROZEN_BEFORE_GWPS_X_ACCESS','source':SOURCE,'source_sha256':source_sha,
         'source_bytes':path.stat().st_size,'training_context':'K562_GWPS_pseudobulk','genes':genes,'targets':targets,
         'development_targets':development,'source_columns':columns,'stable_ids':stable,'ntc_rows':ntc,'target_rows':selected,
         'selected_row_identity':{str(i):meta['rows'][i] for i in sorted(set(ntc+[i for rs in selected.values() for i in rs]))},
         'source_row_axis_sha256':digest(meta['rows']),'source_gene_axis_sha256':digest(meta['ids']),
         'recipe':RECIPE,'input_sha256':{n:sha(project/n) for n in inputs},
         'limits':{'cpu_threads':1,'ram_gib':8,'wall_seconds':600,'growth_bytes':100*1024**2,'downloads':0},
         'provenance_limits':['Source raw pseudo-bulk sums versus means unresolved; normalization removes positive row scale.',
                              'Rows are guide/population aggregates, not independent cells; no cell-count division or weighting.',
                              'Earlier metadata scanner exposed fold_expr/p-value previews for first8rows; this is training-source development, never fresh confirmation.',
                              'Outcome-derived metadata including core_control,fold_expr,pvalues,UMI and quality statistics excluded from row selection/weights.',
                              'All585non-targeting rows used, not selected514core controls.','No Jurkat or official response values.']}
    save(project/CONFIG,cfg)
    report=project/'reports/arc/coverage_v12_gwps.csv';report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['panel','target','pseudobulk_rows','declared_filtered_cells_coverage_only','fallback'])
        for panel,names in [('official',targets),('development',development)]:
            for target in names:
                rows=selected[target];cells=meta['cell_counts'][rows] if meta['cell_counts'] is not None else np.array([])
                total=float(cells.sum()) if len(cells) and np.isfinite(cells).all() else ''
                writer.writerow([panel,target,len(rows),total,'no_effect' if not rows else 'mapped_genes_only'])
    return cfg


def normalized_rows(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all() or (x<0).any():raise ValueError('Pseudo-bulk values must be finite and nonnegative; fractional values permitted')
    totals=x.sum(axis=1)
    if (totals<=0).any():raise ValueError('Zero pseudo-bulk row total')
    return x*(10000./totals[:,None])


def effect_rows(normalized,control_mean):return np.log((normalized+1.)/(control_mean+1.))


def check(project,cfg):
    if cfg.get('source')!=SOURCE or cfg.get('training_context')!='K562_GWPS_pseudobulk' or cfg.get('recipe')!=RECIPE:raise PermissionError('Only frozen GWPS recipe/source may fit')
    for name,expected in cfg['input_sha256'].items():
        if sha(Path(project)/name)!=expected:raise ValueError('Frozen input changed: '+name)
    targets=set(cfg['targets']+cfg['development_targets'])
    if set(cfg['target_rows'])!=targets or not cfg['ntc_rows']:raise ValueError('Source target/control panel changed')
    flat=cfg['ntc_rows']+[i for rows in cfg['target_rows'].values() for i in rows]
    if len(flat)!=len(set(flat)):raise ValueError('Overlapping selected source rows')
    return cfg


def train(project,expected_protocol_sha):
    project=Path(project);start=time.monotonic();protocol=project/CONFIG
    if sha(protocol)!=expected_protocol_sha:raise ValueError('Explicit protocol hash mismatch before source access')
    cfg=check(project,json.loads(protocol.read_text()));receipt_path=project/RECEIPT
    destinations=[project/'artifacts/arc/gwps_transfer_v12.npz',project/'artifacts/arc/gwps_development_effects_v12.npz']
    if receipt_path.exists() or any(p.exists() for p in destinations):raise FileExistsError('GWPS attempt/output already exists')
    receipt={'status':'STARTED','protocol_sha256':expected_protocol_sha,'source_sha256':cfg['source_sha256'],
             'official_response_access':False,'Jurkat_access':False,'fresh_confirmation':False}
    save(receipt_path,receipt)
    try:
        path=project/cfg['source']
        if sha(path)!=cfg['source_sha256']:raise ValueError('GWPS raw bytes changed')
        meta=metadata(path)
        if digest(meta['rows'])!=cfg['source_row_axis_sha256'] or digest(meta['ids'])!=cfg['source_gene_axis_sha256']:raise ValueError('GWPS axes changed')
        for i,name in cfg['selected_row_identity'].items():
            if meta['rows'][int(i)]!=name:raise ValueError('Frozen row identity changed')
        for t,rows in cfg['target_rows'].items():
            if any(meta['labels'][i]!=t for i in rows):raise ValueError('Wrong target membership')
        if any(meta['labels'][i]!='non-targeting' for i in cfg['ntc_rows']):raise ValueError('Wrong control membership')
        columns=np.asarray(cfg['source_columns']);good=np.flatnonzero(columns>=0)
        for i in good:
            if meta['ids'][columns[i]]!=cfg['stable_ids'][i] or meta['symbols'][columns[i]]!=cfg['genes'][i]:raise ValueError('Gene mapping changed')
        response={};fractional_entries=0;values_read=0
        with h5py.File(path,'r') as h:
            control_sum=np.zeros(len(meta['ids']),dtype=np.float64)
            for offset in range(0,len(cfg['ntc_rows']),64):
                rows=cfg['ntc_rows'][offset:offset+64];raw=np.asarray(h['X'][rows,:],dtype=np.float64)
                fractional_entries+=int(np.count_nonzero(raw!=np.floor(raw)));values_read+=raw.size
                control_sum+=normalized_rows(raw).sum(axis=0)
            control_mean=control_sum/len(cfg['ntc_rows'])
            for target,rows in cfg['target_rows'].items():
                effect=np.zeros(len(cfg['genes']),dtype=np.float32)
                if rows:
                    raw=np.asarray(h['X'][rows,:],dtype=np.float64);fractional_entries+=int(np.count_nonzero(raw!=np.floor(raw)));values_read+=raw.size
                    guide_effect=effect_rows(normalized_rows(raw),control_mean)
                    effect[good]=(.25*guide_effect[:,columns[good]].mean(axis=0)).astype(np.float32)
                if not np.isfinite(effect).all():raise ValueError('Nonfinite fitted effect')
                response[target]=effect
        artifacts={}
        for names,destination in zip((cfg['targets'],cfg['development_targets']),destinations):
            effects=np.stack([response[t] for t in names]);supported=np.array([bool(cfg['target_rows'][t]) for t in names])
            arrays={'genes':np.asarray(cfg['genes'],dtype=str),'stable_ids':np.asarray(cfg['stable_ids'],dtype=str),
                    'targets':np.asarray(names,dtype=str),'log_fold_change':effects,'modeled_gene':columns>=0,
                    'per_target_supported':supported,'support_context_count':supported.astype(np.uint8),
                    'source_row_count':np.array([len(cfg['target_rows'][t]) for t in names],dtype=np.int32)}
            with destination.open('xb') as f:np.savez_compressed(f,**arrays)
            artifacts[str(destination.relative_to(project))]={'sha256':sha(destination),'bytes':destination.stat().st_size,
                'shape':list(effects.shape),'supported_targets':int(supported.sum()),'unmatched_gene_max_abs':float(np.max(np.abs(effects[:,columns<0]))) if (columns<0).any() else 0.,
                'unsupported_target_max_abs':float(np.max(np.abs(effects[~supported]))) if (~supported).any() else 0.,
                'effect_min':float(effects.min()),'effect_max':float(effects.max())}
        if sum(a['bytes'] for a in artifacts.values())>cfg['limits']['growth_bytes']:raise ValueError('Artifact growth cap exceeded')
        receipt.update(status='TRAINED_DEVELOPMENT_SOURCE_NOT_VALIDATED',artifacts=artifacts,modeled_genes=len(good),
                       ntc_pseudobulk_rows=len(cfg['ntc_rows']),target_pseudobulk_rows=sum(map(len,cfg['target_rows'].values())),
                       accessed_matrix_values=values_read,fractional_accessed_entries=fractional_entries,
                       source_sums_vs_means='UNRESOLVED; no need to infer because per-row normalization is scale invariant',
                       recipe=cfg['recipe'],no_effect_targets=[t for t in cfg['targets'] if not cfg['target_rows'][t]],
                       limits=cfg['limits'])
    except BaseException as error:receipt.update(status='FAILED_ATTEMPT_CONSUMED',error=type(error).__name__+': '+str(error));raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('freeze','check','train'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--protocol-sha256');args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    def expired(*_):raise TimeoutError('GWPS 600-second cap reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    if args.freeze:
        cfg=freeze(P);print(json.dumps({'protocol_sha256':sha(P/CONFIG),'official_covered':sum(bool(cfg['target_rows'][t]) for t in cfg['targets']),
                                      'development_covered':sum(bool(cfg['target_rows'][t]) for t in cfg['development_targets']),
                                      'genes_mapped':sum(i>=0 for i in cfg['source_columns']),'ntc_rows':len(cfg['ntc_rows'])}))
    elif args.train:
        if not args.protocol_sha256:parser.error('--train requires the prior --freeze protocol hash')
        print(json.dumps(train(P,args.protocol_sha256),indent=2))
    else:
        cfg=check(P,json.loads((P/CONFIG).read_text()));print(json.dumps({'status':'FROZEN_GWPS_METADATA_CHECKED','protocol_sha256':sha(P/CONFIG),'X_values_read':False}))

if __name__=='__main__':main()
