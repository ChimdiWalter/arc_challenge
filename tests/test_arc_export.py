import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
import anndata
from invariantcell.arc_export import export_counts


class ArcExportTests(unittest.TestCase):
    def run_export(self,root,**overrides):
        kw=dict(genes=['g2','g1','g3'],panel=[dict(context='opaque',target='unknown_target',cells=5)],
            controls={'opaque':sparse.csr_matrix([[0,2,0],[1,0,3]])},control_genes={'opaque':['g2','g1','g3']},label='SYNTHETIC_TEST_ONLY')
        kw.update(overrides);return export_counts(Path(root)/'fixture.h5ad',**kw)

    def test_roundtrip_gene_order_count_panel_and_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.run_export(d,batch_size=2);a=anndata.read_h5ad(r['path'])
            self.assertEqual(list(a.var_names),['g2','g1','g3']);self.assertEqual(a.shape,(5,3))
            self.assertEqual(set(a.obs.context),{'opaque'});self.assertEqual(set(a.obs.target_gene),{'unknown_target'})
            self.assertEqual(r['fallback_groups'],[('opaque','unknown_target')]);self.assertTrue(np.issubdtype(a.X.dtype,np.integer))

    def test_stream_batch_size_preserves_samples(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            x=anndata.read_h5ad(self.run_export(a,batch_size=1)['path']).X
            y=anndata.read_h5ad(self.run_export(b,batch_size=4)['path']).X
            self.assertEqual((x!=y).nnz,0)

    def test_fractional_or_negative_counts_rejected(self):
        for value in (-1,.5,float('nan')):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):self.run_export(d,predictor=lambda c,t,x:np.full(x.shape,value))

    def test_gene_order_and_control_submission_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):self.run_export(d,control_genes={'opaque':['g1','g2','g3']})
            with self.assertRaises(ValueError):self.run_export(d,panel=[dict(context='opaque',target='control',cells=5)])
