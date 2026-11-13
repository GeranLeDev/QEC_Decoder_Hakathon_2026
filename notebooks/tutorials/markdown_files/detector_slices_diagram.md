<div style="font-size: 1.18em; line-height: 1.3; border: 1px solid var(--jp-border-color2, #ccc); border-left: 6px solid #7e57c2; padding: 12px; border-radius: 6px; margin: 10px 0;">
<h4>🧠 How to read the diagram</h4>

The colored blocks in this diagram indicates the stabilizers of the code at each timestep. Each of the stabilizers is red or blue at the locations where it meets a physical qubit - red tells us that it stabilizes the physical qubit with a Pauli X while blue tells us that it stabilizes with a Pauli Z. At tick 0, we see that the data qubits which are reset to $\ket0$ (**$R$**) are stabilized with the color blue since $Z\ket0 = \ket0$, while the ones that are reset to $\ket+$ (**$R_X$**) are stabilized by the color red since $X\ket+ = \ket+$.

</div>