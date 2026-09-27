"""Synthetic-only residual math, fallback and outer-group isolation tests."""
import unittest
import numpy as np
from invariantcell.arc_v10 import fit,predict,profile,mean_error,calibrate,calibrated_error,METHODS

OPTIONS={'target_sum':10000.,'shrinkage':.25,'rank':2,'ridge_alpha':10.,'residual_strength':.1,'context_clip':3.,'seed':17,'neural_hidden':4,'neural_steps':20,'neural_learning_rate':.005,'neural_weight_decay':.01,'fit_neural':True}

class ArcV10Tests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(91)
        targets=['a','b','c']
        groups={c:{t:rng.poisson(4+i,(6,8))+1 for t in ['control',*targets]} for i,c in enumerate(['K562','RPE1','HepG2'])}
        return groups,targets
    def test_normalization_uses_per_cell_depth_before_mean(self):
        x=np.array([[1,1],[8,2]])
        log,depth=profile(x,100.)
        np.testing.assert_allclose(np.expm1(log),[65,35]);self.assertEqual(depth,6.)
    def test_normalization_depth_scaling_invariant(self):
        groups,_=self.fixture();x=groups['K562']['control']
        np.testing.assert_allclose(profile(x)[0],profile(10*x)[0])
    def test_fractional_negative_nonfinite_zero_depth_refused(self):
        for x in ([[1.5,2]], [[-1,2]], [[float('nan'),2]], [[0,0]]):
            with self.subTest(x=x),self.assertRaises(ValueError):profile(x)
    def test_historical_metric_independent_simple_calculation(self):
        p=np.array([[1.,3.]]);y=np.array([[2.,4.]])
        expected=((np.log(2)-np.log(3))**2+(np.log(4)-np.log(5))**2)/2
        self.assertAlmostEqual(mean_error(p,y)[0],expected)
    def test_one_context_disables_context_residual_for_all_families(self):
        groups,targets=self.fixture();model=fit({'K562':groups['K562']},targets,OPTIONS)
        results=[predict(model,groups['RPE1']['control'],targets,m) for m in METHODS]
        self.assertTrue(model['zero_residual_fallback'])
        for r in results:np.testing.assert_array_equal(r['normalized_means'],results[0]['normalized_means'])
    def test_missing_target_and_control_use_explicit_no_effect(self):
        groups,targets=self.fixture();model=fit({'K562':groups['K562']},targets,OPTIONS)
        result=predict(model,groups['RPE1']['control'],['missing','control'],'neural')
        expected=np.expm1(profile(groups['RPE1']['control'])[0])
        np.testing.assert_allclose(result['normalized_means'],np.stack([expected,expected]))
        self.assertEqual(result['fallback_targets'],['missing','control'])
    def test_all_families_finite_with_same_depth_budget(self):
        groups,targets=self.fixture();model=fit({c:groups[c] for c in ('K562','RPE1')},targets,OPTIONS)
        for method in METHODS:
            r=predict(model,groups['HepG2']['control'],targets,method)
            self.assertTrue(np.isfinite(r['normalized_means']).all())
            np.testing.assert_allclose(r['normalized_means'].sum(1),10000.)
            np.testing.assert_allclose(r['raw_depth_matched_means'].sum(1),groups['HepG2']['control'].sum(1).mean())
    def test_outer_responses_cannot_change_fit_predictions_or_calibration(self):
        groups,targets=self.fixture();training={c:groups[c] for c in ('K562','RPE1')}
        model=fit(training,targets,OPTIONS);cal=calibrate(training,targets,OPTIONS,'neural')
        before=predict(model,groups['HepG2']['control'],targets,'neural')['normalized_means'].copy()
        for t in targets:groups['HepG2'][t][:]=10**6
        after=predict(fit(training,targets,OPTIONS),groups['HepG2']['control'],targets,'neural')['normalized_means']
        np.testing.assert_array_equal(before,after)
        self.assertEqual(cal['training_contexts'],['K562','RPE1'])
        for row in cal['lineage']:
            self.assertNotIn('HepG2',row['base_training_contexts'])
            self.assertNotIn(row['inner_held'],row['base_training_contexts'])
            self.assertTrue(row['zero_context_residual'])
    def test_undeclared_context_refused(self):
        groups,targets=self.fixture()
        with self.assertRaises(ValueError):fit({'Jurkat':groups['K562']},targets,OPTIONS)
    def test_zero_effect_rank_falls_back_without_nan_fit(self):
        groups={c:{t:np.ones((4,5)) for t in ('control','a','b')} for c in ('K562','RPE1')}
        model=fit(groups,['a','b'],OPTIONS);self.assertTrue(model['zero_residual_fallback'])
        r=predict(model,np.ones((3,5)),['a','b'],'neural')
        np.testing.assert_allclose(r['normalized_means'],2000.)
    def test_calibration_is_finite_and_three_group_fit_refused(self):
        groups,targets=self.fixture();train={c:groups[c] for c in ('K562','RPE1')}
        cal=calibrate(train,targets,OPTIONS,'low_rank');model=fit(train,targets,OPTIONS)
        features=predict(model,groups['HepG2']['control'],targets,'low_rank')['features']
        self.assertTrue(np.isfinite(calibrated_error(features,cal)).all())
        with self.assertRaises(ValueError):calibrate(groups,targets,OPTIONS,'low_rank')
        with self.assertRaises(ValueError):fit(groups,targets,OPTIONS)

if __name__=='__main__':unittest.main()
