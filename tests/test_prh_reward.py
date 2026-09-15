"""Engineering tests for the PRH reward-selection module (item 23 of the
PRHBench pilot design's first-commit checklist).

These test the pure logic in `agent_system/environments/prh.py` in
isolation from the rest of the training stack (no torch/ray/gymnasium
required), so they can run on a plain machine before any GPU launch.

Run with:
    python3 -m unittest discover -s PRHBench/tests -v
"""

import importlib.util
import os
import unittest

import numpy as np

# Load prh.py directly by file path rather than via `import agent_system...`.
# The agent_system package's __init__ chain (agent_system -> .environments)
# imports env_manager.py, which requires torch/omegaconf/ray at import time.
# prh.py itself only depends on numpy, so loading it this way lets these
# tests run on a plain machine with no ML stack installed at all — exactly
# the point of keeping this module dependency-free (see prh.py's docstring).
_PRH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "upstream", "agent_system", "environments", "prh.py"
)
_spec = importlib.util.spec_from_file_location("prh", _PRH_PATH)
prh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prh)

PRHConfig = prh.PRHConfig
PRHRewardRouter = prh.PRHRewardRouter
sample_poison_mask = prh.sample_poison_mask
select_training_reward = prh.select_training_reward


def make_infos(hidden, observed):
    """Build a list of info dicts, one per (hidden, observed) pair."""
    return [{"hidden_reward": h, "observed_reward": o} for h, o in zip(hidden, observed)]


class TestSelectTrainingReward(unittest.TestCase):
    def test_clean_uses_hidden_by_default(self):
        self.assertEqual(
            select_training_reward(hidden_reward=1.0, observed_reward=9.0, poisoned=False),
            1.0,
        )

    def test_poisoned_uses_observed_by_default(self):
        self.assertEqual(
            select_training_reward(hidden_reward=1.0, observed_reward=9.0, poisoned=True),
            9.0,
        )

    def test_missing_hidden_raises_when_strict(self):
        with self.assertRaises(RuntimeError):
            select_training_reward(
                hidden_reward=None, observed_reward=9.0, poisoned=False, strict_hidden_reward=True
            )

    def test_missing_hidden_falls_back_to_zero_when_not_strict(self):
        value = select_training_reward(
            hidden_reward=None, observed_reward=9.0, poisoned=False, strict_hidden_reward=False
        )
        self.assertEqual(value, 0.0)

    def test_missing_hidden_does_not_raise_if_poisoned_selects_observed(self):
        # Selected channel is "observed" (poisoned), so a None hidden reward
        # is irrelevant to select_training_reward's own channel lookup even
        # under strict=True (PRHRewardRouter enforces the stronger
        # "always require hidden reward" invariant separately — see below).
        value = select_training_reward(
            hidden_reward=None, observed_reward=9.0, poisoned=True, strict_hidden_reward=True
        )
        self.assertEqual(value, 9.0)


class TestSamplePoisonMask(unittest.TestCase):
    def test_rho_zero_is_never_poisoned(self):
        rng = np.random.default_rng(0)
        mask = sample_poison_mask(rng, batch_size=1000, poison_prob=0.0)
        self.assertFalse(mask.any())

    def test_rho_one_is_always_poisoned(self):
        rng = np.random.default_rng(0)
        mask = sample_poison_mask(rng, batch_size=1000, poison_prob=1.0)
        self.assertTrue(mask.all())

    def test_rho_quarter_is_approximately_right(self):
        rng = np.random.default_rng(0)
        mask = sample_poison_mask(rng, batch_size=200_000, poison_prob=0.25)
        realized = mask.mean()
        self.assertAlmostEqual(realized, 0.25, delta=0.01)

    def test_seed_changes_only_the_mask(self):
        mask_a = sample_poison_mask(np.random.default_rng(1), batch_size=64, poison_prob=0.5)
        mask_b = sample_poison_mask(np.random.default_rng(2), batch_size=64, poison_prob=0.5)
        self.assertEqual(mask_a.shape, mask_b.shape)
        self.assertFalse(np.array_equal(mask_a, mask_b))


class TestPRHRewardRouter(unittest.TestCase):
    def _router(self, poison_prob, strict=True, seed=10017):
        config = PRHConfig(
            enabled=True,
            poison_prob=poison_prob,
            poison_seed=seed,
            strict_hidden_reward=strict,
        )
        return PRHRewardRouter(config)

    def test_rho_zero_every_training_reward_equals_hidden(self):
        router = self._router(poison_prob=0.0)
        rng = np.random.default_rng(42)
        hidden = rng.normal(size=32)
        observed = rng.normal(size=32) + 100.0  # far from hidden, so any mixup is obvious
        infos = make_infos(hidden, observed)
        router.new_episode(batch_size=32)
        used = router.route(rewards=observed, infos=infos)
        np.testing.assert_allclose(used, hidden)

    def test_rho_one_every_training_reward_equals_observed(self):
        router = self._router(poison_prob=1.0)
        rng = np.random.default_rng(42)
        hidden = rng.normal(size=32)
        observed = rng.normal(size=32) + 100.0
        infos = make_infos(hidden, observed)
        router.new_episode(batch_size=32)
        used = router.route(rewards=observed, infos=infos)
        np.testing.assert_allclose(used, observed)

    def test_realized_poisoning_fraction_near_quarter(self):
        router = self._router(poison_prob=0.25)
        n = 5000
        router.new_episode(batch_size=n)
        self.assertAlmostEqual(router.realized_poison_rate, 0.25, delta=0.02)

    def test_poison_mask_constant_across_steps_within_episode(self):
        router = self._router(poison_prob=0.5, seed=7)
        router.new_episode(batch_size=16)
        mask_at_reset = router.poison_mask.copy()

        rng = np.random.default_rng(0)
        for _ in range(5):
            hidden = rng.normal(size=16)
            observed = rng.normal(size=16)
            infos = make_infos(hidden, observed)
            router.route(rewards=observed, infos=infos)
            # route() must not mutate the mask itself.
            np.testing.assert_array_equal(router.poison_mask, mask_at_reset)

    def test_changing_poison_seed_changes_only_the_mask(self):
        router_a = self._router(poison_prob=0.5, seed=1)
        router_b = self._router(poison_prob=0.5, seed=2)
        mask_a = router_a.new_episode(batch_size=64)
        mask_b = router_b.new_episode(batch_size=64)
        self.assertEqual(mask_a.shape, mask_b.shape)
        self.assertFalse(np.array_equal(mask_a, mask_b))

    def test_hidden_and_observed_reward_never_overwritten(self):
        router = self._router(poison_prob=0.5, seed=3)
        rng = np.random.default_rng(0)
        hidden = rng.normal(size=8)
        observed = rng.normal(size=8)
        infos = make_infos(hidden, observed)
        router.new_episode(batch_size=8)
        router.route(rewards=observed, infos=infos)
        for i, info in enumerate(infos):
            self.assertEqual(info["hidden_reward"], hidden[i])
            self.assertEqual(info["observed_reward"], observed[i])
            # New PRH-specific keys are additive, not replacements.
            self.assertIn("prh_clean_reward", info)
            self.assertIn("prh_proxy_reward", info)
            self.assertIn("prh_poisoned", info)
            self.assertIn("prh_training_reward", info)

    def test_missing_hidden_reward_raises_even_on_poisoned_row(self):
        # Stronger invariant than select_training_reward alone: the router
        # always requires a hidden-reward signal when strict, regardless of
        # which channel this row's mask would select, because the hidden
        # reward is also needed for evaluation (proxy_hidden_gap).
        router = self._router(poison_prob=1.0, strict=True)
        infos = [{"hidden_reward": None, "observed_reward": 5.0}]
        router.new_episode(batch_size=1)
        with self.assertRaises(RuntimeError):
            router.route(rewards=[5.0], infos=infos)

    def test_missing_hidden_reward_tolerated_when_not_strict(self):
        router = self._router(poison_prob=0.0, strict=False)
        infos = [{"hidden_reward": None, "observed_reward": 5.0}]
        router.new_episode(batch_size=1)
        used = router.route(rewards=[5.0], infos=infos)
        self.assertEqual(used[0], 0.0)

    def test_route_without_new_episode_raises(self):
        router = self._router(poison_prob=0.5)
        infos = make_infos([1.0], [2.0])
        with self.assertRaises(RuntimeError):
            router.route(rewards=[2.0], infos=infos)

    def test_missing_hidden_reward_tolerated_on_row_already_done(self):
        # Reproduces a real behavior of the underlying gridworld (see
        # tests/test_env_manager_integration.py): once a row terminates, it
        # is not auto-reset, and stepping it again yields hidden_reward=None
        # even though strict_hidden_reward=true. That degenerate row is
        # masked out downstream via active_masks, so the router must not
        # raise for it once `dones` has marked it inactive.
        router = self._router(poison_prob=0.0, strict=True)
        router.new_episode(batch_size=2)

        # Step 1: row 0 terminates, row 1 keeps going -- both still report
        # a real hidden reward this step, so nothing should raise yet.
        infos_step1 = [
            {"hidden_reward": 49.0, "observed_reward": 49.0},
            {"hidden_reward": -1.0, "observed_reward": -1.0},
        ]
        router.route(rewards=[49.0, -1.0], infos=infos_step1, dones=np.array([True, False]))
        self.assertTrue(infos_step1[0]["prh_active"])
        self.assertTrue(infos_step1[1]["prh_active"])

        # Step 2: row 0 (already done) degenerates to hidden_reward=None;
        # row 1 (still active) keeps reporting a real hidden reward. This
        # must NOT raise, and row 0 must be reported as inactive.
        infos_step2 = [
            {"hidden_reward": None, "observed_reward": 0.0},
            {"hidden_reward": -1.0, "observed_reward": -1.0},
        ]
        used = router.route(rewards=[0.0, -1.0], infos=infos_step2, dones=np.array([False, False]))
        self.assertFalse(infos_step2[0]["prh_active"])
        self.assertTrue(infos_step2[1]["prh_active"])
        self.assertEqual(used[0], 0.0)  # tolerated fallback for the inactive row
        self.assertEqual(used[1], -1.0)  # unaffected: still active, rho=0 -> hidden

    def test_still_active_row_with_missing_hidden_reward_still_raises(self):
        # The tolerance above is specifically for already-inactive rows --
        # a still-active row missing its hidden reward is exactly the
        # original failure mode strict_hidden_reward is meant to catch.
        router = self._router(poison_prob=0.0, strict=True)
        router.new_episode(batch_size=1)
        infos = [{"hidden_reward": None, "observed_reward": 1.0}]
        with self.assertRaises(RuntimeError):
            router.route(rewards=[1.0], infos=infos, dones=np.array([False]))

    def test_route_without_dones_treats_every_row_as_active(self):
        # Omitting `dones` (e.g. single-step ad hoc use) must preserve the
        # original strict behavior: every row is treated as active.
        router = self._router(poison_prob=0.0, strict=True)
        router.new_episode(batch_size=1)
        infos = [{"hidden_reward": None, "observed_reward": 1.0}]
        with self.assertRaises(RuntimeError):
            router.route(rewards=[1.0], infos=infos)


class TestPRHConfig(unittest.TestCase):
    def test_disabled_when_env_has_no_prh_node(self):
        class FakeEnvCfg:
            pass

        config = PRHConfig.from_omegaconf(FakeEnvCfg())
        self.assertFalse(config.enabled)

    def test_reads_fields_from_namespace_like_node(self):
        class FakePRHNode:
            enabled = True
            poison_prob = 0.25
            clean_reward = "hidden"
            poison_reward = "observed"
            poison_unit = "episode"
            poison_seed = 123
            strict_hidden_reward = True

        class FakeEnvCfg:
            prh = FakePRHNode()

        config = PRHConfig.from_omegaconf(FakeEnvCfg())
        self.assertTrue(config.enabled)
        self.assertEqual(config.poison_prob, 0.25)
        self.assertEqual(config.poison_seed, 123)

    def test_rejects_out_of_range_poison_prob(self):
        with self.assertRaises(ValueError):
            PRHConfig(enabled=True, poison_prob=1.5)

    def test_rejects_non_episode_poison_unit(self):
        with self.assertRaises(NotImplementedError):
            PRHConfig(enabled=True, poison_unit="step")


if __name__ == "__main__":
    unittest.main()
