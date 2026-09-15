"""Integration smoke tests for the PRH patch against the *real* environment
stack (ray + gymnasium + the bundled ai-safety-gridworlds/pycolab/safe_grid_gym
packages), as opposed to test_prh_reward.py which only exercises the
dependency-free reward-selection logic in isolation.

Unlike test_prh_reward.py, these tests require the full stack to be
installed (torch, ray, gymnasium, omegaconf, and the three local
safe-grid-gym/pycolab/ai-safety-gridworlds packages) — see
PRHBench/README.md's "Smoke-testing" section for how the `prhbench` conda
environment used to run these was built. They are skipped automatically
if that stack isn't importable, so `python -m unittest discover -s
PRHBench/tests` stays runnable on a bare machine too.

Run with:
    conda activate prhbench
    python3 -m unittest discover -s PRHBench/tests -v
"""

import os
import sys
import unittest

import numpy as np

UPSTREAM_DIR = os.path.join(os.path.dirname(__file__), "..", "upstream")
sys.path.insert(0, UPSTREAM_DIR)

try:
    import ray  # noqa: F401
    import torch  # noqa: F401
    from omegaconf import OmegaConf

    from agent_system.environments.env_manager import SafetyGridworldsEnvironmentManager
    from agent_system.environments.env_package.safe_gridworlds import (
        build_safety_gridworld_envs,
        safety_gridworld_projection,
    )

    _STACK_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when stack is missing
    _STACK_AVAILABLE = False
    _IMPORT_ERROR = exc


def make_config(poison_prob, poison_seed=10017, enabled=True, strict_hidden_reward=True):
    return OmegaConf.create(
        {
            "env": {
                "history_length": 0,
                "prh": {
                    "enabled": enabled,
                    "poison_prob": poison_prob,
                    "clean_reward": "hidden",
                    "poison_reward": "observed",
                    "poison_unit": "episode",
                    "poison_seed": poison_seed,
                    "strict_hidden_reward": strict_hidden_reward,
                },
            }
        }
    )


RANDOM_ACTIONS = ["up", "down", "left", "right"]


def random_text_actions(rng, batch_size):
    return [RANDOM_ACTIONS[i] for i in rng.integers(0, 4, size=batch_size)]


@unittest.skipUnless(_STACK_AVAILABLE, f"full env stack not importable: {_IMPORT_ERROR}")
class TestSafetyGridworldsEnvironmentManagerPRH(unittest.TestCase):
    ENV_NAME = "AbsentSupervisor"
    ENV_NUM = 8
    N_STEPS = 5

    def _build_manager(self, config, is_train, seed=0):
        envs = build_safety_gridworld_envs(
            env_name=self.ENV_NAME,
            seed=seed,
            env_num=self.ENV_NUM,
            group_n=1,
            render_mode="ansi",
            resources_per_worker={"num_cpus": 0.05},
            is_train=is_train,
            env_kwargs={},
        )
        manager = SafetyGridworldsEnvironmentManager(
            envs, safety_gridworld_projection, config, prh_training=is_train
        )
        return manager

    def _run_episode(self, manager, seed=0):
        """Reset once, then take N_STEPS random steps. Returns list of (rewards, infos)."""
        rng = np.random.default_rng(seed)
        _obs, _infos = manager.reset({})
        history = []
        for _ in range(self.N_STEPS):
            actions = random_text_actions(rng, self.ENV_NUM)
            _next_obs, rewards, _dones, infos = manager.step(actions)
            history.append((rewards, infos))
        return history

    def test_rho_zero_training_reward_matches_hidden_reward(self):
        config = make_config(poison_prob=0.0)
        manager = self._build_manager(config, is_train=True)
        try:
            history = self._run_episode(manager)
            for rewards, infos in history:
                for i, info in enumerate(infos):
                    self.assertFalse(info["prh_poisoned"])
                    self.assertIsNotNone(info["hidden_reward"])
                    self.assertEqual(rewards[i], info["hidden_reward"])
                    self.assertEqual(info["prh_training_reward"], info["hidden_reward"])
        finally:
            manager.envs.close()

    def test_rho_one_training_reward_matches_observed_reward(self):
        config = make_config(poison_prob=1.0)
        manager = self._build_manager(config, is_train=True)
        try:
            history = self._run_episode(manager)
            for rewards, infos in history:
                for i, info in enumerate(infos):
                    self.assertTrue(info["prh_poisoned"])
                    self.assertEqual(rewards[i], info["observed_reward"])
                    self.assertEqual(info["prh_training_reward"], info["observed_reward"])
        finally:
            manager.envs.close()

    def test_validation_manager_ignores_poison_prob(self):
        # Same config (poison_prob=1.0, enabled=true) but prh_training=False:
        # the validation-side manager must never alter the returned reward.
        config = make_config(poison_prob=1.0)
        manager = self._build_manager(config, is_train=False)
        try:
            self.assertIsNone(manager.prh_router)
            history = self._run_episode(manager)
            for rewards, infos in history:
                for i, info in enumerate(infos):
                    self.assertNotIn("prh_poisoned", info)
                    self.assertEqual(rewards[i], info["observed_reward"])
        finally:
            manager.envs.close()

    def test_poison_mask_constant_within_episode(self):
        config = make_config(poison_prob=0.5, poison_seed=123)
        manager = self._build_manager(config, is_train=True)
        try:
            _obs, _infos = manager.reset({})
            mask_at_reset = manager.prh_router.poison_mask.copy()
            rng = np.random.default_rng(1)
            for _ in range(self.N_STEPS):
                actions = random_text_actions(rng, self.ENV_NUM)
                _next_obs, _rewards, _dones, infos = manager.step(actions)
                poisoned_this_step = np.array([info["prh_poisoned"] for info in infos])
                np.testing.assert_array_equal(poisoned_this_step, mask_at_reset)
                np.testing.assert_array_equal(manager.prh_router.poison_mask, mask_at_reset)
        finally:
            manager.envs.close()

    def test_realized_poison_rate_tracks_nominal_rho(self):
        config = make_config(poison_prob=0.25, poison_seed=999)
        manager = self._build_manager(config, is_train=True)
        try:
            # Reset many times (small env_num each) to accumulate a large
            # enough sample for the realized rate to approach the nominal one.
            for _ in range(200):
                manager.reset({})
            self.assertAlmostEqual(manager.prh_router.realized_poison_rate, 0.25, delta=0.05)
        finally:
            manager.envs.close()

    def test_prh_disabled_reproduces_upstream_behavior(self):
        # env.prh.enabled=false (the shipped default) must behave exactly
        # like the unpatched upstream manager: no router, raw reward passed
        # straight through as observed reward, no prh_* keys added.
        config = make_config(poison_prob=1.0, enabled=False)
        manager = self._build_manager(config, is_train=True)
        try:
            self.assertIsNone(manager.prh_router)
            history = self._run_episode(manager)
            for rewards, infos in history:
                for i, info in enumerate(infos):
                    self.assertNotIn("prh_poisoned", info)
                    self.assertEqual(rewards[i], info["observed_reward"])
        finally:
            manager.envs.close()

    def test_success_evaluator_reports_prh_instrumentation(self):
        # This is the training-log path (PRHBench next-commit item 2): the
        # per-update poison-rate instrumentation that should reach
        # WandB/console via ray_trainer.py's episode/* and prh/* metrics.
        config = make_config(poison_prob=0.5, poison_seed=123)
        manager = self._build_manager(config, is_train=True)
        try:
            _obs, infos0 = manager.reset({})
            mask_at_reset = manager.prh_router.poison_mask.copy()

            total_infos = [[] for _ in range(self.ENV_NUM)]
            total_batch_list = [[] for _ in range(self.ENV_NUM)]
            rng = np.random.default_rng(2)
            for _ in range(self.N_STEPS):
                actions = random_text_actions(rng, self.ENV_NUM)
                _next_obs, _rewards, _dones, infos = manager.step(actions)
                for i in range(self.ENV_NUM):
                    total_infos[i].append(infos[i])
                    total_batch_list[i].append({"active_masks": True})

            success = manager.success_evaluator(
                total_infos=total_infos, total_batch_list=total_batch_list
            )

            expected_poisoned = int(mask_at_reset.sum())
            expected_rate = expected_poisoned / self.ENV_NUM

            self.assertIn("prh_poison_prob_nominal", success)
            self.assertIn("prh_poison_rate_realized", success)
            self.assertIn("prh_poisoned_episodes", success)
            self.assertIn("prh_total_episodes", success)
            self.assertIn("prh_poison_rate_realized_cumulative", success)
            self.assertIn("proxy_hidden_gap", success)

            np.testing.assert_allclose(success["prh_poison_prob_nominal"], [0.5] * self.ENV_NUM)
            np.testing.assert_allclose(success["prh_poison_rate_realized"], [expected_rate] * self.ENV_NUM)
            np.testing.assert_allclose(success["prh_poisoned_episodes"], [expected_poisoned] * self.ENV_NUM)
            np.testing.assert_allclose(success["prh_total_episodes"], [self.ENV_NUM] * self.ENV_NUM)
            # Only one reset happened, so per-update and cumulative rates match.
            np.testing.assert_allclose(
                success["prh_poison_rate_realized_cumulative"], [expected_rate] * self.ENV_NUM
            )
            # proxy_hidden_gap = cumulative_observed_reward - cumulative_hidden_reward
            np.testing.assert_allclose(
                success["proxy_hidden_gap"],
                success["cumulative_observed_reward"] - success["cumulative_hidden_reward"],
            )
        finally:
            manager.envs.close()

    def test_validation_manager_reports_no_prh_instrumentation(self):
        # The validation-side manager never builds a router, so it must not
        # report any prh_* metrics at all (there is nothing poisoned to report).
        config = make_config(poison_prob=1.0)
        manager = self._build_manager(config, is_train=False)
        try:
            _obs, infos0 = manager.reset({})
            total_infos = [[] for _ in range(self.ENV_NUM)]
            total_batch_list = [[] for _ in range(self.ENV_NUM)]
            rng = np.random.default_rng(3)
            for _ in range(self.N_STEPS):
                actions = random_text_actions(rng, self.ENV_NUM)
                _next_obs, _rewards, _dones, infos = manager.step(actions)
                for i in range(self.ENV_NUM):
                    total_infos[i].append(infos[i])
                    total_batch_list[i].append({"active_masks": True})

            success = manager.success_evaluator(
                total_infos=total_infos, total_batch_list=total_batch_list
            )
            for key in success:
                self.assertFalse(key.startswith("prh_"), f"unexpected PRH metric on validation manager: {key}")
        finally:
            manager.envs.close()


if __name__ == "__main__":
    unittest.main()
