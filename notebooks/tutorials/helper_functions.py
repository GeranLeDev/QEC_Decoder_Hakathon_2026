import stim
import matplotlib.pyplot as plt
import numpy as np

def remap_stim_circuit(circ: stim.Circuit, qubit_map: dict[int, int]) -> stim.Circuit:
    """Returns a copy of the given circuit with qubits remapped according to the given mapping."""
    new_circ = stim.Circuit()
    for op in circ:
        try:
            if op.name == "REPEAT":
                new_circ += remap_stim_circuit(op.body_copy(), qubit_map) * op.repeat_count
                continue
            if op.name in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
                new_circ.append(op)
                continue
            new_targets = [qubit_map[t.qubit_value] for t in op.targets_copy()]
            new_circ.append(op.name, new_targets, op.gate_args_copy())
        except BaseException as e:
            print(f"Error remapping operation: {op}")
            raise e
    return new_circ

def sort_stim_qubit_order(circ: stim.Circuit) -> stim.Circuit:
    """Returns a copy of the given circuit with qubits sorted in ascending order."""
    data_qubit_map = {}
    meas_qubit_map = {}
    num_data_qubits = 0
    num_meas_qubits = 0
    # Construct mappings
    for op in circ:
        if op.name == "QUBIT_COORDS":
            x, y = op.gate_args_copy()
            if x % 2 == 0 and y % 2 == 0:
                meas_qubit_map[op.targets_copy()[0].qubit_value] = num_meas_qubits
                num_meas_qubits += 1
            elif x % 2 == 1 and y % 2 == 1:
                data_qubit_map[op.targets_copy()[0].qubit_value] = num_data_qubits
                num_data_qubits += 1
            else:
                raise ValueError(f"Unexpected qubit coordinates: ({x}, {y})")
    global_qubit_map = {**data_qubit_map, **{k: v+num_data_qubits for k, v in meas_qubit_map.items()}}
    # Create new circuit
    new_circ = remap_stim_circuit(circ, global_qubit_map)
    return new_circ

def tanner_graph(
        check_matrix: np.ndarray,
        observable_matrix: np.ndarray,
        syndrome: np.ndarray,
        logical_flip: np.ndarray,
        posterior_prob: np.ndarray = None,
        title: str = "Tanner graph of the decoding problem"
) -> None:
    if posterior_prob is not None:
        predicted_error = posterior_prob > 0.5
        syndrome_from_predicted_error = (predicted_error @ check_matrix.T) % 2
        predicted_logical_flip = (predicted_error @ observable_matrix.T) % 2

    # Tanner graph
    vxs = np.linspace(0, 10, check_matrix.shape[1])  # Variable nodes (error locations)
    vys = np.zeros(check_matrix.shape[1])
    cxs = np.linspace(0, 10, check_matrix.shape[0])  # Check nodes (detectors)
    cys = np.ones(check_matrix.shape[0])
    oxs = np.linspace(0, 10, observable_matrix.shape[0]+2)[1:-1]  # Observable nodes
    oys = -np.ones(observable_matrix.shape[0])

    plt.figure(figsize=(15, 1), dpi=300)
    # Plot variable nodes
    if posterior_prob is not None:
        plt.scatter(vxs, vys, s=10, c=posterior_prob, label='Error locations', cmap='coolwarm', vmin=0, vmax=1)
        plt.colorbar()
    else:
        plt.scatter(vxs, vys, s=10, c='blue', label='Error locations')
    # Plot check nodes
    mask = syndrome == 1  # Highlight the detectors that fired
    plt.scatter(cxs[mask], cys[mask], s=10, c='red', label='Detectors')
    plt.scatter(cxs[~mask], cys[~mask], s=10, c='red', alpha=0.1)
    # Plot check nodes (from BP perspective)
    if posterior_prob is not None:
        mask = syndrome_from_predicted_error == 1  # Highlight the detectors that fired
        plt.scatter(cxs[mask], cys[mask]+0.2, s=10, marker='s', c='red', label='Detectors')
        plt.scatter(cxs[~mask], cys[~mask]+0.2, s=10, marker='s', c='red', alpha=0.1)
    # Plot observable nodes
    mask = logical_flip == 1  # Highlight the observables that flipped
    plt.scatter(oxs[mask], oys[mask], s=10, c='green', label='Observables')
    plt.scatter(oxs[~mask], oys[~mask], s=10, c='green', alpha=0.1)
    # Plot observable nodes (from BP perspective)
    if posterior_prob is not None:
        mask = predicted_logical_flip == 1  # Highlight the observables that flipped
        plt.scatter(oxs[mask], oys[mask]-.2, s=10, marker='s', c='green', label='Observables')
        plt.scatter(oxs[~mask], oys[~mask]-.2, s=10, marker='s', c='green', alpha=0.1)
    # Plot edges based on the check matrix
    for c, v in zip(*check_matrix.nonzero()):
        if posterior_prob is not None:
            plt.plot([vxs[v], cxs[c]], [vys[v], cys[c]], 'red', lw=1, zorder=-1, alpha=1 if predicted_error[v] else 0.01)
        else:
            plt.plot([vxs[v], cxs[c]], [vys[v], cys[c]], 'red', lw=1, zorder=-1, alpha=.2)
    # Plot edges based on the observable matrix
    for o, v in zip(*observable_matrix.nonzero()):
        if posterior_prob is not None:
            plt.plot([vxs[v], oxs[o]], [vys[v], oys[o]], 'green', lw=1, zorder=-1, alpha=1 if predicted_error[v] else 0.01)
        else:
            plt.plot([vxs[v], oxs[o]], [vys[v], oys[o]], 'green', lw=1, zorder=-1, alpha=.2)
    # Title
    if posterior_prob is not None:
        plt.title(f"{title}\n{'Valid decoding' if np.array_equal(check_matrix @ predicted_error, syndrome) else 'Invalid decoding'}\nPredicted logical flip: {predicted_logical_flip[0]}, Actual logical flip: {logical_flip[0].astype(int)}")
    else:
        plt.title(title)
    plt.axis('off')
    plt.show()