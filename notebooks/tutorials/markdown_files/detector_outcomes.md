<div style="font-size: 1.18em; line-height: 1.3; border: 1px solid var(--jp-border-color2, #ccc); border-left: 6px solid #7e57c2; padding: 12px; border-radius: 6px; margin: 10px 0;">
<h4>🧠 How to understand the output above</h4>

The above output shows which detectors reacted at each round of the QECC. An `_` means no reaction, while a `1` means a detection / change in syndrome measurement. Generally speaking, a `1` in the same location on two consecutive rounds tells us a measurement error occurred on that syndrome and no errors on the data qubits the measurement is connected to. On the other hand, two `1`'s next to each other indicates a Pauli error on the data qubit shared by those detectors, since a Pauli error on a single data qubit activates is detected by two separate stabilizers (unless the error is on the edge of the code).

</div>