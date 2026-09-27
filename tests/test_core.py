"""Engineering invariants on tiny invented values, not challenge experiments."""
from dataclasses import FrozenInstanceError, replace
import unittest

from invariantcell.core import (BiologicalTransition, DependencyRef, miniature_fixture,
                                no_effect, pooled_delta, validate_miniature_panel)


def group(record_id="control", context="heldout", target=None, counts=((4, 8),), **kwargs):
    return BiologicalTransition(record_id, context, target, ("A", "B"), counts,
                                synthetic=True, **kwargs)


class SharedCoreTests(unittest.TestCase):
    def test_raw_counts_reject_float_bool_negative_and_wrong_shape(self):
        for counts in (((1.0, 2),), ((True, 2),), ((-1, 2),), ((1,),), ()):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                group(counts=counts)
        with self.assertRaises(ValueError):
            BiologicalTransition("x", "c", None, ("A", "A"), ((1, 2),))

    def test_freezes_nested_inputs_and_roundtrips(self):
        counts = [[1, 2]]
        record = group(counts=counts)
        counts[0][0] = 900
        self.assertEqual(record.counts, ((1, 2),))
        with self.assertRaises(FrozenInstanceError):
            record.context = "changed"
        self.assertEqual(record, BiologicalTransition.from_dict(record.to_dict()))

    def test_no_effect_does_not_invent_observed_event_time(self):
        control = group(event_time="2020-01-01T00:00:00Z")
        prediction = no_effect(control, "target")
        self.assertEqual(prediction.counts, control.counts)
        self.assertEqual(prediction.state_kind, "predicted")
        self.assertIsNone(prediction.event_time)
        self.assertTrue(prediction.synthetic)
        self.assertEqual(prediction.dependencies[0].dependency_id, control.record_id)

    def test_context_transfer_delta_and_shrinkage_known_arithmetic(self):
        training = (group("train_ctrl", "train", counts=((2, 4),)),
                    group("train_target", "train", "target", ((6, 0),)))
        prediction = pooled_delta(training, group(), "target", shrinkage=1)
        self.assertEqual(prediction.counts, ((6, 6),))  # +4,-4 shrunk by 1/2
        self.assertEqual(len(prediction.dependencies), 3)
        self.assertEqual(pooled_delta(training, group(counts=((0, 0),)), "target", 0).counts, ((4, 0),))

    def test_heldout_context_cannot_leak_into_training(self):
        with self.assertRaisesRegex(ValueError, "held-out"):
            pooled_delta((group("leak", target="target"),), group(), "target")

    def test_training_requires_observed_matching_controls_gene_order_and_domain(self):
        cases = [
            (group("target", "train", "target"),),
            (replace(group("target", "train", "target"), state_kind="predicted"),),
            (replace(group("ctrl", "train"), genes=("B", "A")),),
            (replace(group("ctrl", "train"), synthetic=False),),
        ]
        for training in cases:
            with self.subTest(training=training), self.assertRaises(ValueError):
                pooled_delta(training, group(), "target")

    def test_unknown_target_falls_back_and_invalid_shrinkage_rejected(self):
        prediction = pooled_delta((group("train_ctrl", "train"),), group(), "unknown")
        self.assertEqual(prediction.counts, group().counts)
        self.assertIn("unknown_target", prediction.method)
        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                pooled_delta((), group(), "target", value)

    def test_miniature_validation_is_explicitly_not_official(self):
        prediction = no_effect(group(), "target")
        result = validate_miniature_panel((prediction,), ("A", "B"), (("heldout", "target"),), 1)
        self.assertEqual(result["scope"], "INTERNAL_MINIATURE_ONLY")
        self.assertFalse(result["official_arc_validation"])
        for panel in ((group(),), (prediction, prediction), ()):
            with self.assertRaises(ValueError):
                validate_miniature_panel(panel, ("A", "B"), (("heldout", "target"),), 1)

    def test_fixture_is_deterministic_and_unambiguously_synthetic(self):
        first = miniature_fixture(contexts=1, perturbations=1, genes=2, cells_per_group=2)
        self.assertEqual(first, miniature_fixture(contexts=1, perturbations=1, genes=2, cells_per_group=2))
        self.assertTrue(all(record.synthetic and record.event_time is None for record in first))
        self.assertTrue(all(record.record_id.startswith("SYNTHETIC:") for record in first))

    def test_dependency_hash_and_timezone_are_checked(self):
        with self.assertRaises(ValueError):
            DependencyRef("bad", "abc", None)
        with self.assertRaises(ValueError):
            group(event_time="2020-01-01")
        shifted = group(event_time="2020-01-01T01:00:00+01:00")
        self.assertEqual(shifted.event_time, "2020-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
