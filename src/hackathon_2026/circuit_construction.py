import stim, json
from pathlib import Path
import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(os.path.join(_DATA_DIR, "metadata_18_4_4.json")) as f:
    metadata = json.load(f)
with open(os.path.join(_DATA_DIR, "coord_18_4_4.stim")) as f:
    coord = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "syndrome_18_4_4.stim")) as f:
    syndrome = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "det_init_x.stim")) as f:
    det_init_x = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "det_init_z.stim")) as f:
    det_init_z = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "det_round.stim")) as f:
    det_round = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "det_meas_x.stim")) as f:
    det_meas_x = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "det_meas_z.stim")) as f:
    det_meas_z = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "observable_x.stim")) as f:
    observable_x = stim.Circuit(f.read())
with open(os.path.join(_DATA_DIR, "observable_z.stim")) as f:
    observable_z = stim.Circuit(f.read())
shift_time = stim.Circuit("SHIFT_COORDS(0, 0, 1)")
L_data_qubits  = [idx for name, idx in metadata["qubit_index"].items() if name[0] == "L"]
R_data_qubits  = [idx for name, idx in metadata["qubit_index"].items() if name[0] == "R"]
X_check_qubits = [idx for name, idx in metadata["qubit_index"].items() if name[0] == "X"]
Z_check_qubits = [idx for name, idx in metadata["qubit_index"].items() if name[0] == "Z"]

def create_circuit(basis: str, rounds: int) -> stim.Circuit:
    """Create the [[18,4,4]] QEC code from [Wang et al. (2026)](https://doi.org/10.1038/s41567-025-03157-4) with the given logical basis (shared for all four logical qubits) and number of rounds."""
    assert basis in ["X", "Z"]
    assert rounds >= 1

    circuit = stim.Circuit()
    circuit += coord
    # Initialize data qubits to |0> or |+> depending on the basis
    circuit.append_operation("R", L_data_qubits+R_data_qubits)
    if basis == "X":
        circuit.append_operation("Tick")
        circuit.append_operation("H", L_data_qubits+R_data_qubits)
    circuit += syndrome
    circuit += det_init_x if basis == "X" else det_init_z
    # Add rounds of syndrome extraction (SHIFT_COORDS gives detectors from round t the time coordinate t)
    circuit += (syndrome + shift_time + det_round) * (rounds - 1)
    # Logical measurement of data qubits in the chosen basis
    if basis == "X":
        circuit.append_operation("H", L_data_qubits+R_data_qubits)
        circuit.append_operation("Tick")
    circuit.append_operation("M", L_data_qubits+R_data_qubits)
    circuit += shift_time
    circuit += det_meas_x if basis == "X" else det_meas_z
    circuit += observable_x if basis == "X" else observable_z

    return circuit