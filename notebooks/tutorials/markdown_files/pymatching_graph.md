<div style="font-size: 1.18em; line-height: 1.3">

The output of this cell is saved to a html file named `surface_code_circuit_decoder_graph.html` which allows you to pan around in 3d and get a better understanding of the graph. The file can be found in the same directory as this notebook and be opened in a web-browser.

</div>

<div style="font-size: 1.18em; line-height: 1.3; border: 1px solid var(--jp-border-color2, #ccc); border-left: 6px solid #7e57c2; padding: 12px; border-radius: 6px; margin: 10px 0;">
<h4>🧠 How to read the 3d graph</h4>

- The 3 dimensions are the two spatial dimensions of the surface code (seen directly in the above graph) and a time dimension moving away from the point of view in the above graph.
- The black dots in the code represent the parity check detectors placed throughout the code (the stabilizers).
- The lines that connect the dots represent the errors that can occur in the code based on the selected noise. Each error is either a connection between two detectors (black dots) or a connection between a detector and outside of the code (representing errors that will only trigger one syndrome measurement).
- The red lines are just like the black lines except that if an odd number of them is used in the decoding process it signals that the observable of the code has been flipped.

</div>