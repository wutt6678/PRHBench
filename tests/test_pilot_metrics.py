"""Unit tests for analysis/pilot_metrics.py, using synthetic validation
records (no real training run needed) to check the wiring and sign
conventions of the persistence-curve math: matched deltas, P-AUC, and the
recovery half-life.

The synthetic numbers reproduce the "promising result" example from the
PRHBench pilot design (section 19): Delta_H(0)=15 shrinking to 2 by k=60,
which should half-recover (drop to <= 7.5) at k=20.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import pilot_metrics as pm  # noqa: E402

ATTACK_END_STEP = 60
KS = [0, 5, 10, 20, 40, 60]

# Delta_H^rho(k) = -[15, 13, 11, 7, 4, 2] (attacked branch worse throughout).
DELTA_H = [15, 13, 11, 7, 4, 2]
GAP_EFFECT = 5.0  # constant Delta_G^rho(k) = +5 (attacked branch's gap is bigger)


def make_records():
    records = []
    for k, dh in zip(KS, DELTA_H):
        step = ATTACK_END_STEP + k
        # Control branch (rho=0): flat H=10, flat O=10 (gap=0).
        records.append(
            {"seed": 17, "rho": 0.0, "global_step": step, "hidden_reward": 10.0, "observed_reward": 10.0}
        )
        # Treated branch (rho=1): H depressed by dh, O = H + GAP_EFFECT.
        h_treated = 10.0 - dh
        o_treated = h_treated + GAP_EFFECT
        records.append(
            {
                "seed": 17,
                "rho": 1.0,
                "global_step": step,
                "hidden_reward": h_treated,
                "observed_reward": o_treated,
            }
        )
    return pm.records_from_rows(records)


class TestWashoutCurve(unittest.TestCase):
    def test_control_curve_is_flat(self):
        records = make_records()
        curve = pm.washout_curve(records, seed=17, rho=0.0, attack_end_step=ATTACK_END_STEP, ks=KS)
        np.testing.assert_allclose(curve["hidden"], [10.0] * len(KS))
        np.testing.assert_allclose(curve["gap"], [0.0] * len(KS))

    def test_missing_checkpoint_raises(self):
        records = make_records()
        with self.assertRaises(KeyError):
            pm.washout_curve(records, seed=17, rho=0.0, attack_end_step=ATTACK_END_STEP, ks=[0, 1])

    def test_duplicate_record_raises(self):
        records = make_records() + [
            pm.ValidationRecord(seed=17, rho=0.0, global_step=ATTACK_END_STEP, hidden_reward=0.0, observed_reward=0.0)
        ]
        with self.assertRaises(ValueError):
            pm.washout_curve(records, seed=17, rho=0.0, attack_end_step=ATTACK_END_STEP, ks=KS)


class TestMatchedDeltas(unittest.TestCase):
    def test_delta_hidden_matches_synthetic_construction(self):
        records = make_records()
        deltas = pm.matched_deltas(records, seed=17, rho=1.0, attack_end_step=ATTACK_END_STEP, ks=KS)
        np.testing.assert_allclose(deltas["delta_hidden"], [-d for d in DELTA_H])

    def test_delta_gap_matches_synthetic_construction(self):
        records = make_records()
        deltas = pm.matched_deltas(records, seed=17, rho=1.0, attack_end_step=ATTACK_END_STEP, ks=KS)
        np.testing.assert_allclose(deltas["delta_gap"], [GAP_EFFECT] * len(KS))


class TestPAuc(unittest.TestCase):
    def test_p_auc_hidden_matches_direct_trapz(self):
        delta_hidden = np.array([-d for d in DELTA_H], dtype=float)
        ks = np.array(KS, dtype=float)
        expected = -np.trapz(delta_hidden, ks) / ks[-1]
        self.assertAlmostEqual(pm.p_auc_hidden(delta_hidden, ks), expected)
        # A persistently-worse attacked branch (delta_hidden always negative)
        # should give a *positive* P_H (net persistence of harm).
        self.assertGreater(pm.p_auc_hidden(delta_hidden, ks), 0)

    def test_p_auc_gap_matches_direct_trapz(self):
        delta_gap = np.array([GAP_EFFECT] * len(KS), dtype=float)
        ks = np.array(KS, dtype=float)
        expected = np.trapz(delta_gap, ks) / ks[-1]
        self.assertAlmostEqual(pm.p_auc_gap(delta_gap, ks), expected)
        self.assertAlmostEqual(pm.p_auc_gap(delta_gap, ks), GAP_EFFECT)  # constant -> trivially its own mean

    def test_p_auc_requires_k_start_at_zero(self):
        with self.assertRaises(ValueError):
            pm.p_auc_hidden(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


class TestRecoveryHalfLife(unittest.TestCase):
    def test_half_life_matches_synthetic_construction(self):
        # |Delta_H| = [15, 13, 11, 7, 4, 2]; half of 15 is 7.5; first k with
        # |Delta_H(k)| <= 7.5 is k=20 (|Delta_H(20)|=7).
        delta_hidden = np.array([-d for d in DELTA_H], dtype=float)
        ks = np.array(KS, dtype=float)
        self.assertEqual(pm.recovery_half_life(delta_hidden, ks), 20.0)

    def test_never_half_recovers_returns_none(self):
        delta_hidden = np.array([-15, -14, -13, -12, -11, -10], dtype=float)
        ks = np.array(KS, dtype=float)
        self.assertIsNone(pm.recovery_half_life(delta_hidden, ks))

    def test_zero_baseline_returns_none(self):
        delta_hidden = np.array([0.0, 0.0], dtype=float)
        ks = np.array([0.0, 5.0])
        self.assertIsNone(pm.recovery_half_life(delta_hidden, ks))


class TestSummarize(unittest.TestCase):
    def test_summary_matches_expected_signature(self):
        records = make_records()
        result = pm.summarize(records, seed=17, rho=1.0, attack_end_step=ATTACK_END_STEP, ks=KS)
        self.assertEqual(result["k_half"], 20.0)
        self.assertGreater(result["p_auc_hidden"], 0)
        self.assertAlmostEqual(result["p_auc_gap"], GAP_EFFECT)
        # Convincing PRH signature (delta_H<0 and delta_G>0) at every k>0.
        self.assertTrue(all(result["prh_signature_at_k"]))

    def test_unsorted_ks_rejected(self):
        records = make_records()
        with self.assertRaises(ValueError):
            pm.summarize(records, seed=17, rho=1.0, attack_end_step=ATTACK_END_STEP, ks=[0, 10, 5])

    def test_never_reaching_half_life_reports_bound_not_extrapolation(self):
        records = []
        for k, dh in zip(KS, [15, 14, 13, 12, 11, 10]):
            step = ATTACK_END_STEP + k
            records.append(
                {"seed": 1, "rho": 0.0, "global_step": step, "hidden_reward": 10.0, "observed_reward": 10.0}
            )
            records.append(
                {
                    "seed": 1,
                    "rho": 1.0,
                    "global_step": step,
                    "hidden_reward": 10.0 - dh,
                    "observed_reward": 10.0 - dh,
                }
            )
        recs = pm.records_from_rows(records)
        result = pm.summarize(recs, seed=1, rho=1.0, attack_end_step=ATTACK_END_STEP, ks=KS)
        self.assertEqual(result["k_half"], "> 60")


class TestGate3LearnableCheck(unittest.TestCase):
    def test_improved_case(self):
        records = pm.records_from_rows(
            [
                {"seed": 17, "rho": 0.0, "global_step": 0, "hidden_reward": 5.0, "observed_reward": 5.0},
                {"seed": 17, "rho": 0.0, "global_step": 40, "hidden_reward": 9.0, "observed_reward": 9.0},
            ]
        )
        result = pm.gate3_learnable_check(records, seed=17)
        self.assertTrue(result["hidden_reward_improved"])
        self.assertAlmostEqual(result["hidden_reward_delta"], 4.0)

    def test_no_records_raises(self):
        with self.assertRaises(KeyError):
            pm.gate3_learnable_check([], seed=99)


if __name__ == "__main__":
    unittest.main()
