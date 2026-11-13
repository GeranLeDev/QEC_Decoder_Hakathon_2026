"""Your decoder for the code check.

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README).

Everything that happens inside decode() counts towards your runtime, including building matrices from the circuit
or loading trained weights. Training doesn't count, but it must be reproducible from the code you hand in.
"""
import numpy as np
import stim


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    # Replace this with your decoder. Predicting no flips gives an LER of about 0.29.
    return np.zeros((len(detector_events), 4), dtype=np.uint8)
