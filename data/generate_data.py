"""Sample labelled shots in the same CSV format as the Kaggle data.

Run from the repository root:

    python data/generate_data.py data/generated [--shots 100000] [--seed 1] [--rounds 5] [--noise-probability 0.002]
                                                [--basis X Z] [--exclude-observable-flips]

The output folder gets <folder name>.csv (e.g. data/generated/generated.csv), one <basis>_circuit.stim per basis and
args.json. Any existing files in that folder are deleted first.

The defaults rebuild the competition's circuits exactly (5 rounds, p = 0.002), so the output can be used as extra
training data. Other rounds or noise strengths are fine for training too; the file then has 14 x rounds detector
columns instead of 70.

With --exclude-observable-flips, <folder name>.csv holds only the detector columns (like
data/kaggle_public/kaggle_public.csv) and the
true observable flips go to solutions.csv in the submission format plus a Kaggle Usage column (id,L0,L1,...,Usage).
A random 30% of each basis's shots are marked Public and the other 70% Private.
"""
import argparse
import json
import os

import numpy as np
from hackathon_2026 import apply_circuit_noise, create_circuit, test_dataset

parser = argparse.ArgumentParser(description="Sample labelled shots in the Kaggle CSV format.")
parser.add_argument("out", help="output folder, e.g. data/generated")
parser.add_argument("--shots", type=int, default=100_000, help="shots per basis (default: 100000)")
parser.add_argument("--seed", type=int, default=None, help="random seed; basis X uses seed, Z uses seed + 1 (default: random)")
parser.add_argument("--rounds", type=int, default=5, help="rounds of error correction (default: 5, as in the competition)")
parser.add_argument("--noise-probability", type=float, default=0.002, help="SI1000 noise strength p (default: 0.002, as in the competition)")
parser.add_argument("--basis", nargs="+", default=["X", "Z"], choices=["X", "Z"], help="which experiments to sample (default: X Z)")
parser.add_argument("--exclude-observable-flips", action="store_true",
                    help="leave the observable flip columns out of <folder name>.csv and save them to solutions.csv instead, "
                         "with a Usage column marking 30%% of the shots Public and 70%% Private")
args = parser.parse_args()

PUBLIC_FRACTION = 0.3


def rows_as_text(bits):
    """(n, k) array of 0/1 -> n strings like '0,1,1,0'."""
    chars = np.asarray(bits, dtype=np.uint8) + np.uint8(ord("0"))
    return [",".join(row.tobytes().decode()) for row in chars]


os.makedirs(args.out, exist_ok=True)
for file in os.listdir(args.out):
    os.remove(os.path.join(args.out, file))

with open(os.path.join(args.out, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

width = len(str(args.shots - 1))
data_name = os.path.basename(os.path.normpath(args.out)) + ".csv"
data_file = open(os.path.join(args.out, data_name), "w")
solutions_file = None
if args.exclude_observable_flips:
    split_rng = np.random.default_rng(args.seed)
    solutions_file = open(os.path.join(args.out, "solutions.csv"), "w")
try:
    for k, basis in enumerate(args.basis):
        circuit = apply_circuit_noise(create_circuit(basis, args.rounds), p=args.noise_probability)
        with open(os.path.join(args.out, f"{basis}_circuit.stim"), "w") as g:
            g.write(str(circuit))
        seed = None if args.seed is None else args.seed + k
        detector_events, observable_flips = test_dataset(circuit, shots=args.shots, seed=seed)

        if k == 0:
            det_header = [f"D{i}" for i in range(detector_events.shape[1])]
            obs_header = [f"L{i}" for i in range(observable_flips.shape[1])]
            if args.exclude_observable_flips:
                data_file.write(",".join(["id", "basis"] + det_header) + "\n")
                solutions_file.write(",".join(["id"] + obs_header + ["Usage"]) + "\n")
            else:
                data_file.write(",".join(["id", "basis"] + det_header + obs_header) + "\n")

        if args.exclude_observable_flips:
            is_public = np.zeros(args.shots, dtype=bool)
            is_public[split_rng.permutation(args.shots)[:round(PUBLIC_FRACTION * args.shots)]] = True

        for i, (dets, obs) in enumerate(zip(rows_as_text(detector_events), rows_as_text(observable_flips))):
            shot_id = f"gen_{basis}_{i:0{width}d}"
            if args.exclude_observable_flips:
                data_file.write(f"{shot_id},{basis},{dets}\n")
                usage = "Public" if is_public[i] else "Private"
                solutions_file.write(f"{shot_id},{obs},{usage}\n")
            else:
                data_file.write(f"{shot_id},{basis},{dets},{obs}\n")
        print(f"{basis}: {args.shots} shots, {detector_events.shape[1]} detectors")
finally:
    data_file.close()
    if solutions_file is not None:
        solutions_file.close()
print(f"wrote {args.out}")
