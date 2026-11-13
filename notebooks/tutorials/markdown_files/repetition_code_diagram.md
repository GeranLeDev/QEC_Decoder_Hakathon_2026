<div style="font-size: 1.18em; line-height: 1.3; border: 1px solid var(--jp-border-color2, #ccc); border-left: 6px solid #7e57c2; padding: 12px; border-radius: 6px; margin: 10px 0;">
<h4>🧠 How to read the quantum circuit</h4>

Here is a list of the quantum gates that appera in the above quantum circuit as well as the instructions that are <code>stim</code> specific.

- `R`: This will reset the qubit to the $\ket0$ state.
- `M`: This measures the qubit in the Z-basis. The measurement result is stored in the`rec[i]` variable.
- `MR`: This is a measurement of the qubit in the Z-basis followed by a reset to the $\ket0$ state.
- `●-⊕`: This is the CNOT gate, with the `●` being the control and the `⊕` being the target.
- `DETECTOR`: This returns the parity of one or more measurement records`rec[i]` (it is `1` if an odd number of `rec[i]` are `1`, and `0` otherwise).
- `OBS_INCLUDE(n)`: Like `DETECTOR`, but returns the logical observable instead of a stabilizer observable.
- `REP#`: Indicates that the part of the circuit inside the brackets `[]` should be repeated `#` times.

</div>