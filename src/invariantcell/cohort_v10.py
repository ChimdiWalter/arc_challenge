"""Donor-unit validation around the audited, MIT-licensed source bootstrap."""
import math
from ._sb_donor_bootstrap import donor_bootstrap_ci

def donor_difference(records,metric,control='Control',case='COVID-19',n_boot=1000,seed=17):
    import pandas as pd
    if not isinstance(records,list) or not records:raise ValueError('Donor records required')
    if control==case or type(n_boot)is not int or not 100<=n_boot<=10000:raise ValueError('Invalid groups or bounded bootstrap budget')
    ids=set();groups={control:0,case:0}
    for r in records:
        donor=r.get('donor_id');condition=r.get('condition');value=r.get(metric)
        if not isinstance(donor,str) or not donor or donor in ids:raise ValueError('One aggregate row per independently identified donor required')
        if condition not in groups or type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite donor metric and known group required')
        ids.add(donor);groups[condition]+=1
    if min(groups.values())<2:raise ValueError('At least two donors per group; this does not establish adequate power')
    result=donor_bootstrap_ci(pd.DataFrame(records),metric,control,case,n_boot=n_boot,seed=seed)
    return {**result,'independent_unit':'donor','donors_per_group':groups,'source_method':'sb_class_project:donor_bootstrap_ci',
        'limitations':['Percentile bootstrap conditional on exchangeable independent donors; not a proven domain-generalization guarantee.',
         'No cell-level pseudoreplication; this function does not establish causality or observed temporal progression.']}
