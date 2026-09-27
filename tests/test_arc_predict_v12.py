import json,tempfile,unittest
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from invariantcell.arc_predict import predict,transport

class OfficialTransportTests(unittest.TestCase):
    def test_unsupported_genes_exactly_preserved_and_counts_integral(self):
        x=sparse.csr_matrix([[0,3,2],[4,0,1]])
        y=transport(x,np.array([0.,np.log(2),0.]),np.random.default_rng(17))
        np.testing.assert_array_equal(y.toarray(),[[0,6,2],[4,0,1]])
        with self.assertRaises(ValueError):transport(x,np.array([0.,float('nan'),0.]),np.random.default_rng(17))

    def test_dynamic_opaque_phase_full_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'genes.csv').write_text('gene_name\nz\na\n');(p/'perts.csv').write_text('target_gene\nnew_target\n')
            m={'contexts':['future_context'],'n_genes':2,'n_constructs':1,'cells_per_pert':4,'control_label':'non-targeting','context_col':'context','pert_col':'target_gene'}
            (p/'manifest.json').write_text(json.dumps(m))
            a=ad.AnnData(sparse.csr_matrix([[0,2],[3,0]]),obs=pd.DataFrame({'context':['future_context']*2,'target_gene':['non-targeting']*2},index=['c1','c2']),var=pd.DataFrame(index=['z','a']))
            a.write_h5ad(p/'context_future_context.h5ad')
            r=predict(controls=p,manifest=p/'manifest.json',genes=p/'genes.csv',perts=p/'perts.csv',output=p/'out.h5ad')
            x=ad.read_h5ad(p/'out.h5ad');self.assertEqual(x.shape,(4,2));self.assertEqual(x.var_names.tolist(),['z','a'])
            self.assertEqual(set(x.obs.context),{'future_context'});self.assertEqual(set(x.obs.target_gene),{'new_target'})
            self.assertFalse(r['official_response_access']);self.assertEqual(x.X.data.dtype,np.dtype('float32'))
            np.savez(p/'bad.npz',genes=['a','z'],targets=['new_target'],log_fold_change=np.zeros((1,2)))
            with self.assertRaisesRegex(ValueError,'axis mismatch'):predict(controls=p,manifest=p/'manifest.json',genes=p/'genes.csv',perts=p/'perts.csv',checkpoint=p/'bad.npz',output=p/'bad.h5ad')
