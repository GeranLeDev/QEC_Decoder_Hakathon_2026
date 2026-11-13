"""Run your decoder the way the code check does on a local problem set: decode every shot, time it and report the LER.

Run from the repository root:

    python submission_template/run_local.py [--problem-set data/test_set] [--runs 5] [--threads N] [--shots N] [--out submission.csv]

A problem set is a folder made by data/generate_data.py. It holds the circuits X_circuit.stim and Z_circuit.stim and
one file <folder name>.csv (e.g. data/test_set/test_set.csv) with a row per shot: the columns id, basis, the detector events D0, D1, ... and, if available, the
true flips L0 to L3. The LER is only reported when the true flips are there: data/test_set (the default) has them,
but data/kaggle_public and data/kaggle_private (downloaded from Kaggle) do not, so there you only get the runtime
and the submission file. Run once per Kaggle folder, each with its own --out, and submit both files. You can make more sets with labels using
data/generate_data.py.

For each run, decode() is called once for the X shots and once for the Z shots, and the runtime of a run is the total
time of the two calls. The script reports the median over the runs, as we do on our timing machine (one NVIDIA B200
GPU with 48 CPU cores, where we set the thread counts to 48). Your own machine will be faster or slower, so treat the
number as a guide.
"""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Decode a local problem set with decoder.decode, time it and report the LER.")
parser.add_argument("--problem-set", default="data/train", help="problem set folder (default: data/test_set)")
parser.add_argument("--decoder", default=HERE, help="folder with your decoder.py (default: this folder)")
parser.add_argument("--runs", type=int, default=5, help="timed runs; the median is reported (default: 5)")
parser.add_argument("--threads", type=int, default=None, help="set OMP_NUM_THREADS and NUMBA_NUM_THREADS (we use 48)")
parser.add_argument("--shots", type=int, default=None, help="only use the first N shots of each basis (default: all)")
parser.add_argument("--out", default="submission.csv", help="submission file to write (default: submission.csv)")
args = parser.parse_args()

if args.threads:  # must be set before numpy, numba or torch are imported
    for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)

import numpy as np  # noqa: E402
import stim  # noqa: E402
from hackathon_2026 import load_kaggle_csv, logical_error_rate, save_submission  # noqa: E402

sys.path.insert(0, os.path.abspath(args.decoder))
from decoder import decode  # noqa: E402

data_path = os.path.join(args.problem_set, os.path.basename(os.path.normpath(args.problem_set)) + ".csv")
print(f"loading {data_path} ...")
data = load_kaggle_csv(data_path)
circuits, ids, detector_events, observable_flips = {}, {}, {}, {}
for b in "XZ":
    circuits[b] = stim.Circuit.from_file(os.path.join(args.problem_set, f"{b}_circuit.stim"))
    rows = np.flatnonzero(data["basis"] == b)[: args.shots]
    ids[b] = data["id"][rows]
    detector_events[b] = data["detector_events"][rows]
    if "observable_flips" in data:
        observable_flips[b] = data["observable_flips"][rows]
    print(f"{args.problem_set} {b}: {len(rows)} shots")


def run():
    """Decode every shot, one decode() call per basis. Returns ({basis: predictions}, seconds)."""
    predictions = {}
    start = time.perf_counter()
    for b in "XZ":
        out = np.asarray(decode(circuits[b], detector_events[b]))
        n = len(detector_events[b])
        assert out.shape == (n, 4), f"decode returned shape {out.shape} for {n} {b} shots, expected ({n}, 4)"
        assert np.isin(out, (0, 1)).all(), f"decode returned values other than 0 and 1 for the {b} shots"
        predictions[b] = out
    return predictions, time.perf_counter() - start


times, first = [], None
for i in range(args.runs):
    predictions, seconds = run()
    times.append(seconds)
    print(f"run {i + 1}: {seconds:.2f} s")
    if first is None:
        first = predictions
    elif not all(np.array_equal(first[b], predictions[b]) for b in "XZ"):
        print("  warning: predictions differ from run 1; fix your decoder's random seed so the code check can reproduce your submission")
print(f"runtime (median of {args.runs}): {statistics.median(times):.2f} s")

# Submission format: header "id,L0,L1,L2,L3", one row per shot with the ids from the problem set CSV, X shots then Z shots
save_submission(args.out, np.concatenate([ids[b] for b in "XZ"]), np.concatenate([first[b] for b in "XZ"]))
print(f"wrote {args.out}")

if len(observable_flips) < 2:
    print(f"no observable flips in {args.problem_set}, so no LER; use a set with labels such as data/test_set")
    sys.exit()
for b in "XZ":
    print(f"{b}: LER {logical_error_rate(first[b], observable_flips[b]):.5f}")
both_pred = np.concatenate([first[b] for b in "XZ"])
both_true = np.concatenate([observable_flips[b] for b in "XZ"])
print(f"both: LER {logical_error_rate(both_pred, both_true):.5f}")
