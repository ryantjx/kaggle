# Overview

$$\text{TM-score} = \max \left( \frac{1}{L_{\text{ref}}} \sum_{i=1}^{L_{\text{align}}} \frac{1}{1 + \left( \frac{d_i}{d_0} \right)^2 } \right)$$

where:
- Lref is the number of residues solved in the experimental reference structure ("ground truth").
- Lalign is the number of aligned residues.
- di is the distance between the ith pair of aligned residues, in Angstroms.
- d0 is a distance scaling factor in Angstroms, defined as:
$$d_0 = 0.6 (L_{\text{ref}} - 0.5)^{1/2}  - 2.5$$

for Lref ≥ 30; and d0 = 0.3, 0.4, 0.5, 0.6, or 0.7 for Lref <12, 12-15, 16-19, 20-23, or 24-29, respectively.

The rotation and translation of predicted structures to align with experimental reference structures are carried out by US-align. To match default settings, as used in the CASP competitions, the alignment will be sequence-independent.

For each target RNA sequence, you will submit 5 predictions and your final score will be the average of best-of-5 TM-scores of all targets. For a few targets, multiple slightly different structures have been captured experimentally; your predictions' scores will be based on the best TM-score compared to each of these reference structures.