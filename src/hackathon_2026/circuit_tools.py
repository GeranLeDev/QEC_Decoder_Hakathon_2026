import stim
import numpy as np

def problem_info(qec_circuit: stim.Circuit) -> dict[str, np.ndarray]:
    """Extracts the following properties from a QEC circuit:
    
    | Property | Description |
    |----------|-------------|
    | **H** | Parity check matrix |
    | **L** | Observable matrix |
    | **p** | Noise model |
    
    Args:
        qec_circuit (stim.Circuit): Noisy circuit to extract properties from.
    
    Returns:
        dict: A dictionary containing the parity check matrix, observable matrix, and noise model with keys `H`, `L`, and `p`.
    """
    dem = qec_circuit.detector_error_model()
    # Merge error instructions with the same (detectors, observables) effect into one column
    columns: dict[tuple[frozenset[int], frozenset[int]], int] = {}
    priors: list[float] = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        key = (frozenset(detectors), frozenset(observables))
        p = instruction.args_copy()[0]
        if key in columns:
            q = priors[columns[key]]
            priors[columns[key]] = q * (1 - p) + p * (1 - q)  # either one fires, not both
        else:
            columns[key] = len(priors)
            priors.append(p)
    H = np.zeros((dem.num_detectors, len(priors)), dtype=np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), dtype=np.uint8)
    for (detectors, observables), j in columns.items():
        H[list(detectors), j] = 1
        L[list(observables), j] = 1
    return {"H": H, "L": L, "p": np.asarray(priors)}

def test_dataset(qec_circuit: stim.Circuit, shots: int, seed: int = None):
    """Generates a test dataset of detector events and observable flips from a QEC circuit.

    Args:
        qec_circuit (stim.Circuit): Noisy circuit to sample from.
        shots (int): Number of shots to sample / experiments to run.
        seed (int, optional): Random seed. Pass one to get the same samples every time.

    Returns:
        tuple: `(detector_events, observable_flips)` as boolean arrays of shape
        `(shots, #detectors)` and `(shots, #observables)`.
    """
    sampler = qec_circuit.compile_detector_sampler(seed=seed)
    detector_events, observable_flips = sampler.sample(shots=shots, separate_observables=True)
    return detector_events, observable_flips