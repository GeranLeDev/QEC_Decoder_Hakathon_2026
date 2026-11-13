import numpy as np


def logical_error_rate(predicted_observable_flips: np.ndarray, actual_observable_flips: np.ndarray) -> float:
    """
    The leaderboard metric: the fraction of predicted observable flips that are wrong (lower is better).

    Every observable of every shot counts separately. With 0/1 predictions this equals the MAE that Kaggle
    computes.

    Args:
        predicted_observable_flips (np.ndarray): Predicted flips, shape (shots, 4), 0/1 or bool.
        actual_observable_flips (np.ndarray): True flips, same shape.

    Returns:
        float: The logical error rate (LER).
    """
    predicted = np.asarray(predicted_observable_flips)
    actual = np.asarray(actual_observable_flips)
    assert predicted.shape == actual.shape, f"The two inputs do not have the same shape. {predicted.shape} != {actual.shape}."
    return float(np.mean(predicted.astype(bool) != actual.astype(bool)))


def final_score(ler: float, runtime: float, t_cap: float, alpha: float) -> float:
    """
    The final score for the top 10 teams: LER * max(1, runtime / t_cap) ** alpha (lower is better).

    A decoder that finishes within t_cap keeps its LER; every doubling of the runtime beyond t_cap multiplies the
    score by 2 ** alpha. See the Evaluation section of the README for the values of t_cap and alpha.

    Args:
        ler (float): The logical error rate on the Private leaderboard.
        runtime (float): Total wall-clock time of the two decode calls, in seconds.
        t_cap (float): Target runtime in seconds.
        alpha (float): How steeply the score is penalised beyond t_cap.

    Returns:
        float: The final score.
    """
    return ler * max(1.0, runtime / t_cap) ** alpha
