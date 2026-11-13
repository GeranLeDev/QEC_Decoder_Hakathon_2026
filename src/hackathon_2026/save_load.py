import numpy as np

OBSERVABLE_COLUMNS = ["L0", "L1", "L2", "L3"]


def load_kaggle_csv(path):
    """
    Load a problem set CSV, e.g. data/kaggle_public/kaggle_public.csv or data/test_set/test_set.csv.

    Returns a dict with:
        "id": string array of shot ids,
        "basis": string array, "X" or "Z" for each shot,
        "detector_events": bool array of shape (shots, #detectors), columns D0, D1, ...,
        "observable_flips": bool array of shape (shots, 4), columns L0 to L3 (only if the file has them).
    """
    with open(path) as f:
        header = f.readline().strip().split(",")
    detector_columns = [i for i, name in enumerate(header) if name.startswith("D")]
    observable_columns = [header.index(name) for name in OBSERVABLE_COLUMNS if name in header]
    text = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(header.index("id"), header.index("basis")), dtype=str, ndmin=2)
    data = {
        "id": text[:, 0],
        "basis": text[:, 1],
        "detector_events": np.loadtxt(path, delimiter=",", skiprows=1, usecols=detector_columns, dtype=np.uint8, ndmin=2).astype(bool),
    }
    if observable_columns:
        data["observable_flips"] = np.loadtxt(path, delimiter=",", skiprows=1, usecols=observable_columns, dtype=np.uint8, ndmin=2).astype(bool)
    return data


def save_submission(path, ids, predictions):
    """
    Write a Kaggle submission file: a header row "id,L0,L1,L2,L3", then one row per test shot.

    The columns are always written in this order. Kaggle matches them by position, not by name, so a file with the
    columns in another order would be scored against the wrong observables without an error.

    Args:
        path: Where to write the file, e.g. "submission.csv".
        ids: The shot ids from the problem set's CSV (e.g. data/kaggle_public/kaggle_public.csv), in any order.
        predictions: 0/1 or bool array of shape (len(ids), 4), row i for ids[i].
    """
    predictions = np.asarray(predictions)
    assert predictions.ndim == 2 and predictions.shape[1] == len(OBSERVABLE_COLUMNS), f"predictions must have shape (shots, 4), got {predictions.shape}"
    assert len(ids) == len(predictions), f"{len(ids)} ids but {len(predictions)} rows of predictions"
    assert np.isin(predictions, (0, 1)).all(), "predictions must be 0 or 1"
    with open(path, "w") as f:
        f.write(",".join(["id", *OBSERVABLE_COLUMNS]) + "\n")
        for shot_id, row in zip(ids, predictions.astype(np.uint8)):
            f.write(f"{shot_id}," + ",".join(str(v) for v in row) + "\n")
