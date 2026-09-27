"""Small original image-only baseline and independent NuCLS measurement scorer.

No learning, parameter search, clinical endpoint or P09 genomic pairing. Source
rectangles validate object localization, not pixel boundaries or nuclear area.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

RECIPE = {'version':'hematoxylin_otsu_components_v1','gaussian_sigma_pixels':0.8,
          'histogram_bins':256,'minimum_component_pixels':20,'maximum_component_pixels':5000,
          'connectivity':8,'box_iou_threshold':0.5}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_package(package):
    package=Path(package).resolve();p=json.loads(package.read_text());root=package.parent
    if p['recipe']!=RECIPE or p['outputs_examined_before_freeze'] is not False:
        raise ValueError('recipe changed or package was not frozen before outcomes')
    if p['rights']['data_license']!='CC0-1.0':raise ValueError('unsupported rights declaration')
    paths={}
    for role,record in p['assets'].items():
        candidate=root/record['path']
        if candidate.is_symlink():raise ValueError('asset symlink')
        path=candidate.resolve()
        if not path.is_relative_to(root):raise ValueError('asset escapes package')
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('asset changed: '+role)
        paths[role]=path
    for name,expected in p['code_sha256'].items():
        code=Path(__file__).resolve().parents[2]/name
        if sha(code)!=expected:raise ValueError('frozen measurement code changed')
    scale=p['pixel_size_um']
    if scale is not None and (len(scale)!=2 or not all(np.isfinite(v) and v>0 for v in scale)):
        raise ValueError('invalid physical scale')
    return p,paths


def inspect_package(package):
    """Hash, dimensional, coordinate and semantic checks; no predictions/scores."""
    p,paths=load_package(package)
    with Image.open(paths['image']) as im:
        if im.mode!='RGB':raise ValueError('source image must be RGB')
        width,height=im.size
    with Image.open(paths['mask']) as im:
        if im.size!=(width,height) or im.mode!='RGB':raise ValueError('source mask/image mismatch')
    rows=read_annotations(paths['annotations'],(height,width))
    return {'status':'READY_FOR_ROOT_QUEUED_MEASUREMENT','package_sha256':sha(package),
            'shape':[height,width,3],'reference_objects':len(rows),
            'annotation_types':{t:sum(r['type']==t for r in rows) for t in ('rectangle','polyline')},
            'pixel_size_um':p['pixel_size_um'],'prediction_computed':False,'accuracy_computed':False}


def read_annotations(path,shape):
    rows=[];height,width=shape
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            if row['type'] not in ('rectangle','polyline'):raise ValueError('unsupported reference geometry')
            box=tuple(int(row[k]) for k in ('xmin','ymin','xmax','ymax'))
            if not (0<=box[0]<width and box[0]<=box[2]<=width and 0<=box[1]<height and box[1]<=box[3]<=height):raise ValueError('reference outside image')
            xs=[int(v) for v in row['coords_x'].split(',')];ys=[int(v) for v in row['coords_y'].split(',')]
            if len(xs)!=len(ys) or len(xs)<3:raise ValueError('invalid geometry')
            if any(not(0<=x<=width and 0<=y<=height) for x,y in zip(xs,ys)):raise ValueError('coordinate outside image')
            rows.append({'type':row['type'],'box':(box[0],box[1],min(box[2],width-1),min(box[3],height-1)),
                         'source_box':box,'image_edge_clipped':box[2]==width or box[3]==height,'vertices':list(zip(xs,ys)),
                         'classification':row['raw_classification'],'source_row':len(rows)})
    if not rows:raise ValueError('empty independent reference')
    return rows


def otsu(values,bins=256):
    lo,hi=float(values.min()),float(values.max())
    if hi==lo:return hi
    histogram,edges=np.histogram(values,bins=bins,range=(lo,hi));centers=(edges[:-1]+edges[1:])/2
    weights=np.cumsum(histogram);sums=np.cumsum(histogram*centers);total=weights[-1]
    denom=weights*(total-weights)
    between=np.divide((sums[-1]*weights-sums*total)**2,denom,out=np.zeros_like(sums,dtype=float),where=denom>0)
    return float(centers[int(np.argmax(between))])


def predict_instances(rgb):
    """Receives RGB only: annotation/FOV/scale values cannot tune predictions."""
    a=np.asarray(rgb)
    if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8:raise ValueError('expected uint8 RGB image')
    optical=-np.log((a.astype(float)+1)/256)
    score=ndimage.gaussian_filter(optical[:,:,0]-0.5*optical[:,:,2],RECIPE['gaussian_sigma_pixels'])
    binary=score>otsu(score,RECIPE['histogram_bins'])
    labels,_=ndimage.label(binary,structure=np.ones((3,3),dtype=int))
    sizes=np.bincount(labels.ravel());keep=(sizes>=RECIPE['minimum_component_pixels'])&(sizes<=RECIPE['maximum_component_pixels']);keep[0]=False
    labels,_=ndimage.label(keep[labels],structure=np.ones((3,3),dtype=int))
    return labels.astype(np.int32)


def references(rows,encoded_mask):
    a=np.asarray(encoded_mask)
    if a.ndim!=3 or a.shape[2]!=3:raise ValueError('invalid NuCLS encoding')
    classes=a[:,:,0];known={0,253,99,*range(1,13)}
    if not set(np.unique(classes)).issubset(known):raise ValueError('unknown NuCLS mask class')
    roi=classes>0
    if not roi.any():raise ValueError('empty intended annotation region')
    shape=classes.shape;foreground=Image.new('1',(shape[1],shape[0]));ignored=Image.new('1',(shape[1],shape[0]))
    fd,idraw=ImageDraw.Draw(foreground),ImageDraw.Draw(ignored)
    objects=[]
    for r in rows:
        if r['classification']=='unlabeled':idraw.rectangle(r['box'],fill=1);continue
        objects.append(r)
        if r['type']=='polyline':fd.polygon(r['vertices'],fill=1)
        else:idraw.rectangle(r['box'],fill=1)
    valid=roi & ~np.asarray(ignored,dtype=bool) & (classes!=99)
    return np.asarray(foreground,dtype=bool)&valid,valid,roi,objects


def instance_boxes(labels,roi):
    boxes=[];ids=[]
    for ident,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None:continue
        yy,xx=np.where(labels[sl]==ident);yy=yy+sl[0].start;xx=xx+sl[1].start
        if not roi[int(np.rint(yy.mean())),int(np.rint(xx.mean()))]:continue
        boxes.append((int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())));ids.append(ident)
    return boxes,ids


def box_iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])+1);y=max(0,min(a[3],b[3])-max(a[1],b[1])+1)
    intersection=x*y;union=(a[2]-a[0]+1)*(a[3]-a[1]+1)+(b[2]-b[0]+1)*(b[3]-b[1]+1)-intersection
    return intersection/union if union else 0.0


def match_objects(predicted,reference,threshold=0.5):
    scores=np.array([[box_iou(a,b) for b in reference] for a in predicted],dtype=float).reshape(len(predicted),len(reference))
    matches=[]
    if scores.size:
        # Cardinality first, IoU second. Never let a subthreshold match consume a valid match.
        reward=(scores>=threshold)*(min(scores.shape)+1)+scores*(scores>=threshold)
        ii,jj=linear_sum_assignment(-reward)
        matches=[{'prediction_index':int(i),'reference_index':int(j),'box_iou':float(scores[i,j])} for i,j in zip(ii,jj) if scores[i,j]>=threshold]
    tp=len(matches);fp=len(predicted)-tp;fn=len(reference)-tp
    return {'true_positive':tp,'false_positive':fp,'false_negative':fn,'matches':matches,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def score_measurements(labels,rows,encoded_mask,pixel_size_um=None):
    if labels.ndim!=2 or labels.shape!=encoded_mask.shape[:2] or not np.issubdtype(labels.dtype,np.integer) or (labels<0).any():
        raise ValueError('invalid inferred instance labels')
    truth,valid,roi,objects=references(rows,encoded_mask)
    inferred=(labels>0)&valid;tp=int((inferred&truth).sum());fp=int((inferred&~truth&valid).sum());fn=int((~inferred&truth).sum())
    pboxes,ids=instance_boxes(labels,roi)
    ref=[r for r in objects if roi[(r['box'][1]+r['box'][3])//2,(r['box'][0]+r['box'][2])//2]]
    count=len(pboxes);expected=len(ref);refarea=int(truth.sum());predarea=int(inferred.sum());unitarea=None if pixel_size_um is None else float(np.prod(pixel_size_um))
    return {'pixel':{'evaluated_pixels':int(valid.sum()),'excluded_rectangle_or_unknown_pixels':int((roi&~valid).sum()),
                     'true_positive':tp,'false_positive':fp,'false_negative':fn,'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
                     'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None},
            'objects':{'reference_count':expected,'predicted_count':count,'signed_count_error':count-expected,
                       'absolute_count_error':abs(count-expected),'matching':match_objects(pboxes,[r['box'] for r in ref],RECIPE['box_iou_threshold']),
                       'prediction_ids':ids,'reference_polylines':sum(r['type']=='polyline' for r in ref),
                       'reference_rectangles':sum(r['type']=='rectangle' for r in ref)},
            'area':{'scope':'annotated FOV excluding rectangle-only/unknown reference regions',
                    'reference_pixels':refarea,'predicted_pixels':predarea,'absolute_error_pixels':abs(predarea-refarea),
                    'reference_um2':None if unitarea is None else refarea*unitarea,
                    'predicted_um2':None if unitarea is None else predarea*unitarea,
                    'absolute_error_um2':None if unitarea is None else abs(predarea-refarea)*unitarea},
            'scope':'One preselected human breast-cancer FOV; no genomic pairing or clinical validation.'}
