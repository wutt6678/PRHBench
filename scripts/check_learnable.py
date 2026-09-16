#!/usr/bin/env python3
# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Automated Gate-3 learnability check, straight from a run_phase.sh log.

Scans one or more console log files (as produced by grpo_train.sh's
console logger) for lines of the form:

    step:<N> - ... - val/cumulative_hidden_reward_mean:<H> - ... \
        val/cumulative_observed_reward_mean:<O> - ...

(exactly what verl's ray_trainer.py prints per update -- see
PRHBench/README.md's "What changed in upstream/" table), extracts
(global_step, hidden_reward, observed_reward) triples, and reports
whether hidden reward improved from the first available validation
checkpoint to the last -- the same H(0) vs H(end) comparison
analysis/pilot_metrics.py's gate3_learnable_check() makes, but driven
directly from the log files multiple run_phase.sh invocations produce
(one log per phase), rather than from a hand-built CSV.

Exit code 0 (and prints "LEARNABLE") if hidden reward improved by at
least --min-delta; exit code 1 (and prints "NOT_LEARNABLE") otherwise.
Intended for use in an unattended pipeline:

    if python3 scripts/check_learnable.py phaseA.log phaseB.log; then
        # proceed to the next phase
    fi
"""

import argparse
import re
import sys

# Matches "step:<N> - ... val/cumulative_hidden_reward_mean:<H> ... " and,
# separately, "val/cumulative_observed_reward_mean:<O>" later on the same
# line. Values are the console logger's %.3f-formatted floats (optionally
# negative), consistent with the metric lines PRHBench's real Gate 1 runs
# actually produced.
STEP_RE = re.compile(r"step:(\d+)\s*-")
HIDDEN_RE = re.compile(r"val/cumulative_hidden_reward_mean:(-?\d+\.\d+)")
OBSERVED_RE = re.compile(r"val/cumulative_observed_reward_mean:(-?\d+\.\d+)")


def extract_records(lines):
    """Yield (step, hidden_reward, observed_reward) for each matching line."""
    for line in lines:
        step_match = STEP_RE.search(line)
        hidden_match = HIDDEN_RE.search(line)
        if not step_match or not hidden_match:
            continue
        observed_match = OBSERVED_RE.search(line)
        step = int(step_match.group(1))
        hidden = float(hidden_match.group(1))
        observed = float(observed_match.group(1)) if observed_match else None
        yield step, hidden, observed


def load_records(paths):
    records = {}
    for path in paths:
        with open(path, errors="replace") as f:
            for step, hidden, observed in extract_records(f):
                # A later phase's log may re-print an already-seen step
                # (e.g. re-validating right after a resume); keep the
                # first occurrence encountered across the given file order.
                records.setdefault(step, (hidden, observed))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_paths", nargs="+", help="one or more run_phase.sh console log files, in phase order")
    parser.add_argument(
        "--min-delta", type=float, default=0.0,
        help="minimum H(end) - H(start) to count as learnable (default: 0.0, i.e. any improvement)",
    )
    args = parser.parse_args()

    records = load_records(args.log_paths)
    if not records:
        print("NOT_LEARNABLE: no 'val/cumulative_hidden_reward_mean' lines found in the given log(s).", file=sys.stderr)
        sys.exit(1)

    steps = sorted(records)
    start_step, end_step = steps[0], steps[-1]
    hidden_start, observed_start = records[start_step]
    hidden_end, observed_end = records[end_step]
    delta = hidden_end - hidden_start

    print(f"H({start_step}) = {hidden_start:.3f}   O({start_step}) = {observed_start}")
    print(f"H({end_step}) = {hidden_end:.3f}   O({end_step}) = {observed_end}")
    print(f"Delta_H = H({end_step}) - H({start_step}) = {delta:.3f}  (threshold: > {args.min_delta})")

    if delta > args.min_delta:
        print(f"LEARNABLE: hidden reward improved by {delta:.3f} from step {start_step} to {end_step}.")
        sys.exit(0)
    else:
        print(f"NOT_LEARNABLE: hidden reward did not improve (delta={delta:.3f}) from step {start_step} to {end_step}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
