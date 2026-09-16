# PRHBench handover: resuming on a new machine

Written for whoever (human or a fresh Claude Code session with no memory
of prior work on this repo) picks up PRHBench on a new machine. State as
of commit `2dc4288` (2026-09-16). Read `README.md` for the full design;
this file is just the "how to pick up where it left off" checklist.

## Why we're moving machines

The previous node is a 4-GPU cluster (RTX 6000 Ada, 49 GB each) shared
with other users' jobs. Free memory routinely sat below the 20 GiB
safety threshold for 30-45+ minute stretches, and even after a GPU
cleared that bar, another tenant's process claimed nearly all of it
(47 GiB down to 135 MiB free) within the ~1 minute it took to start
training, causing repeated CUDA OOM crashes. This is external
contention, not a bug in this project — see README "Status" for the
detailed evidence. Move to a machine with more headroom (ideally not
shared, or with more consistently free memory) and resume from there.

## What does and doesn't transfer via git

Only code, patches, tests, and docs are in this repo. **Not** transferred,
and not needed to be:
- `upstream/` — a full third-party clone, recreated by `scripts/setup_upstream.sh`.
- `checkpoints/`, `run_logs/`, `wandb/` — gitignored training artifacts.
  Gate 1's exact numeric results are already written into `README.md`'s
  Status section, so losing the old checkpoint files themselves doesn't
  lose that evidence.
- The `prhbench` / `prhbench-train` conda environments — recreated by
  the setup scripts below.

## Setup on the new machine

```bash
git clone https://github.com/wutt6678/PRHBench.git
cd PRHBench

# Lightweight env (test suites, no GPU training stack)
conda env create -f environment.yml
conda activate prhbench
pip install --index-url https://download.pytorch.org/whl/cu128 torch   # CUDA build, if on GPU
bash scripts/setup_upstream.sh
python3 -m unittest discover -s tests   # should be 56 passed, 8 skipped (the skipped ones need `prhbench-train`'s real ray/vllm stack)

# Full training env (needed for anything below)
conda env create -f environment-train.yml
conda activate prhbench-train
bash scripts/setup_training_env.sh   # vLLM first, then gridworlds, then requirements_safety.txt, then upstream itself, editable
```

If you name the training env something other than `prhbench-train`,
either rename it or pass `TRAIN_CONDA_ENV=<name>` to every script below
— `run_phase.sh` activates it by exact name (sourcing `conda.sh`
directly, not `~/.bashrc`, so it also works under `nohup`; see README
"Env name matters").

Make sure `HF_TOKEN` is set or cached at `~/.cache/huggingface/token`.

## Resuming the actual work

Nothing from Gates 2-3 or the pilot completed on the old machine — every
attempt was blocked before a single checkpoint was written. There is
nothing to resume *from*; just launch the pipeline fresh:

```bash
mkdir -p run_logs
nohup env ENV_NAME=AbsentSupervisor MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
  SEED=17 N_GPUS=1 PROJECT_NAME=prhbench_pilot MIN_DELTA=0.0 \
  bash scripts/run_full_pipeline.sh > run_logs/orchestrator_stdout.log 2>&1 &
disown
```

This one command runs, unattended, in order:
1. **Gate 2** (`run_resume_smoke.sh`): short continuous-vs-segmented
   resume diagnostic. Logged but never blocks the rest.
2. **Gate 3 phase A** (0→10 updates, `rho=0`): stops the whole pipeline
   if `scripts/check_learnable.py` finds no hidden-reward improvement.
3. **Gate 3 phase B** (10→40 updates): same learnability check, 0-vs-40.
4. **The seed-17 pilot**: for each dose in `{0, 0.25, 1.0}`, an attack
   phase (`resume_from` the step-40 checkpoint, 40→60) then a washout
   phase (60→120), with per-stage GPU auto-selection and OOM-specific
   retry throughout.

Check progress any time with `tail -f run_logs/PIPELINE_STATUS.txt` —
it's designed not to need babysitting (indefinite GPU-wait by default,
1s polling, automatic OOM retry). See README's "Running everything
unattended" section for the full list of tuning env vars
(`MIN_FREE_MIB`, `GPU_MAX_WAIT_SECONDS`, `MAX_ATTEMPTS`, etc.) if the
new machine's contention profile is different enough to want different
defaults.

If a previous attempt was killed or failed partway through Stage 2+,
check `upstream/checkpoints/prhbench_pilot/` for a manifest-only
directory (contains only `prhbench_manifest.json`, no `global_step_N/`
subdirectory) and `rm -rf` it before relaunching — that's leftover
bookkeeping from `run_phase.sh`, not real training state.

## Once the pilot finishes

Compute the persistence curves with `analysis/pilot_metrics.py`
(`gate3_learnable_check`, `matched_deltas`, `p_auc_hidden`/`p_auc_gap`,
`recovery_half_life`) against the attack+washout validation logs for
each dose, comparing `rho ∈ {0.25, 1.0}` against the `rho=0` control.

**Do not** add more doses, seeds, or models, and don't move on to
BoatRace or any other gridworld, until these seed-17 curves have been
reviewed together — that decision was made explicitly earlier in this
project and still holds.

## One caveat for later, not now

The PRH reward router's RNG restarts fresh in every `run_phase.sh`
process (it isn't checkpointed across phase boundaries). This is fine
for the current design, where each phase runs start-to-finish
uninterrupted. It would need attention before any future work that
resumes a phase mid-way with multiple seeds, since the poison-mask
sequence would restart rather than continue. Not a blocker for the
seed-17 pilot as currently scoped.
