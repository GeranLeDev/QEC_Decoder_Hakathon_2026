import stim

def apply_circuit_noise(qec_circuit: stim.Circuit, p: float) -> stim.Circuit:
    """Apply the SI1000 circuit-level noise model from [Gidney et al. (2021)](https://doi.org/10.22331/q-2021-12-20-605) to the given QEC circuit.
    
    Applied per TICK-separated moment:
        two-qubit gates:    DEPOLARIZE2(p) after the gate
        single-qubit gates: DEPOLARIZE1(p/10) after the gate
        resets:             flip with probability 2p after the reset
        measurements:       flip with probability 5p before the measurement
        idle qubits:        DEPOLARIZE1(p/10) on qubits not used in the moment
        resonator idle:     DEPOLARIZE1(2p) on qubits not measured/reset in a moment containing a measurement/reset
                            (stacks with the idle noise above)
    """
    return _apply_circuit_noise(qec_circuit, p, in_use=set(range(qec_circuit.num_qubits)))

def _apply_circuit_noise(qec_circuit: stim.Circuit, p: float, in_use: set[int] | None = None) -> stim.Circuit:
    num_qubits = qec_circuit.num_qubits
    noisy_circuit = stim.Circuit()
    touched: set[int] = set()
    measured_or_reset: set[int] = set()
    if in_use is None:
        in_use: set[int] = set()

    def add_idle_noise():
        # Idle(p/10) and ResonatorIdle(2p) stack, matching the reference implementation in
        # https://github.com/Strilanc/honeycomb_threshold/blob/main/src/noise.py
        idle = sorted((set(range(num_qubits)) - touched) & in_use)
        if touched and idle:  # skip empty moments (e.g. after a trailing TICK)
            noisy_circuit.append("DEPOLARIZE1", idle, p / 10)
        waiting = sorted((set(range(num_qubits)) - measured_or_reset) & in_use)
        if measured_or_reset and waiting:
            noisy_circuit.append("DEPOLARIZE1", waiting, 2 * p)

    for instruction in qec_circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            noisy_body = _apply_circuit_noise(instruction.body_copy(), p, in_use)
            noisy_circuit.append(stim.CircuitRepeatBlock(instruction.repeat_count, noisy_body))
            continue
        qubits = [t.value for t in instruction.targets_copy() if t.is_qubit_target]
        match instruction.name:
            case "TICK":
                add_idle_noise()
                noisy_circuit.append(instruction)
                touched.clear()
                measured_or_reset.clear()
                continue
            case "CZ":
                noisy_circuit.append(instruction)
                noisy_circuit.append("DEPOLARIZE2", qubits, p)
            case "X" | "Y" | "Z" | "H":
                noisy_circuit.append(instruction)
                noisy_circuit.append("DEPOLARIZE1", qubits, p / 10)
            case "R":
                noisy_circuit.append(instruction)
                noisy_circuit.append("X_ERROR", qubits, 2 * p)
                measured_or_reset.update(qubits)
                in_use.update(qubits)
            case "M":
                noisy_circuit.append("X_ERROR", qubits, 5 * p)
                noisy_circuit.append(instruction)
                measured_or_reset.update(qubits)
                in_use.difference_update(qubits)
            case "QUBIT_COORDS" | "SHIFT_COORDS" | "DETECTOR" | "OBSERVABLE_INCLUDE":
                noisy_circuit.append(instruction)
                continue
            case _:
                raise ValueError(f"Warning: unknown instruction {instruction.name}, skipping noise addition.")
        touched.update(qubits)
    return noisy_circuit