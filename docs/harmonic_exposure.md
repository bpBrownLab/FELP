# Harmonic exposure

Harmonic exposure measures how much a FELP distribution changes under a smooth
continuation into unvisited cells of its CV atlas. It uses the reference
assignments and predicted probabilities, without a target trajectory.

## Discrete Dirichlet construction

Let $q_i$ be the normalized FELP probability on a complete regular grid.
Partition the cells into those visited by the original reference, $V$, and
those unvisited, $W$. Fix an amplitude at each visited cell:

$$
a_i=\sqrt{q_i},\qquad i\in V.
$$

Visited cells with $q_i=0$ remain fixed zero anchors. Score filtering cannot
turn an originally visited cell into an unvisited one.

Join cardinal neighbors along each axis, with conductance $c_{ij}=1/h_d^2$
for spacing $h_d$ along axis $d$. The continuation minimizes Dirichlet energy

$$
\mathcal E(a)=\frac12\sum_{\{i,j\}\in E}c_{ij}(a_i-a_j)^2
$$

subject to the fixed visited amplitudes. Differentiating with respect to each
free amplitude gives

$$
\sum_{j\sim i}c_{ij}(a_i-a_j)=0,\qquad i\in W.
$$

Thus each free amplitude is a conductance-weighted average of its neighbors.
With graph Laplacian $L$, the linear system is

$$
L_{WW}a_W=-L_{WV}a_V.
$$

Periodic axes wrap; nonperiodic boundaries include only edges inside the atlas.
There are no imposed zero anchors beyond the grid boundary. The solve checks
the residual, preservation of anchors, nonnegativity and maximum principle.

## From continued amplitudes to H

The fixed amplitudes satisfy $\sum_{i\in V}a_i^2=1$. Let the additional
squared amplitude on unvisited cells be

$$
E_W=\sum_{i\in W}a_i^2.
$$

Normalize the extended distribution as $\tilde q_i=a_i^2/(1+E_W)$.
On the visited cells this reduces the original mass by $E_W/(1+E_W)$; the
same mass is assigned to unvisited cells. Consequently,

$$
\begin{aligned}
H&=\operatorname{TV}(q,\tilde q) \\
&=\frac12\left[
\sum_{i\in V}\left(q_i-\frac{q_i}{1+E_W}\right)
+\sum_{i\in W}\frac{a_i^2}{1+E_W}\right] \\
&=\frac{E_W}{1+E_W}.
\end{aligned}
$$

The total-variation identity is exact for this continuation. The continuation
itself is a smoothness assumption: H neither estimates nor bounds the true
target population outside the sampled cells. H is zero when every cell was
visited, even if the sampling inside those cells is inadequate. Its value
depends on the CVs, grid domain, resolution and physical axis spacing.

## Worked example

The supplied [five-cell chain](../examples/harmonic_exposure.json) has

| Cell | Reference count | FELP probability | Fixed amplitude | Local ESS |
|---|---:|---:|---:|---:|
| `left` | 30 | 0.36 | 0.6 | 12 |
| `gap_left` | 0 | 0 | — | 0 |
| `visited_zero` | 1 | 0 | 0 | 1 |
| `gap_right` | 0 | 0 | — | 0 |
| `right` | 40 | 0.64 | 0.8 | 4 |

With unit spacing, the two unknown amplitudes are the averages of their
neighbors: 0.3 and 0.4. Hence $E_W=0.3^2+0.4^2=0.25$ and $H=0.2$.
The normalized continuation is $(0.288,0.072,0,0.128,0.512)$.
The optional unreliable-mass diagnostic is $\mathcal U=0.64$, because only
the right endpoint has positive target mass and local ESS below 10.

```bash
python -m felp harmonic-exposure \
  --input examples/harmonic_exposure.json \
  --output harmonic_result.json
```

This is an analytic example of the calculation, not a molecular benchmark.
The command refuses to overwrite an existing output file.

## Saved-array interface

| JSON field | Meaning |
|---|---|
| `schema_version` | Integer 1 |
| `state_ids` | Unique labels in the full physical grid's C order |
| `state_counts` | Original assigned reference counts, before score QC |
| `q` | Normalized final FELP probabilities in the same order |
| `grid_shape` | One or two positive integer axis lengths |
| `periodic` | Boolean wrapping flag for each axis |
| `steps` | Positive uniform physical spacing for each axis |
| `local_ess` | Optional local treated-weight ESS, for U with ESS < 10 |

Include empty cells, with zero q. Physical adjacency must come from the atlas;
sorting arbitrary state IDs does not recover it. Use original reference counts,
not counts reconstructed from the filtered score table.

`summarize` derives cell order from the atlas's CV intervals. Incomplete or
irregular geometry produces `H=null` with `status=unavailable_atlas`, rather
than zero. Periodicity comes from the explicitly requested atlas axes. The
full calculation is recorded under `overlap_diagnostics.harmonic_exposure` in
`run_summary.json`; U and H also appear in `manuscript_diagnostics`.

## Relation to the literature

The graph Dirichlet construction is established in Doyle and Snell,
*Random Walks and Electric Networks* (1984); Zhu, Ghahramani and Lafferty,
*Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions*,
ICML (2003), pp. 912–919; and Grady, *Random Walks for Image Segmentation*,
IEEE TPAMI (2006), 28, 1768–1783, DOI 10.1109/TPAMI.2006.233.

Those works support the graph extension. Extending square-root FELP
probabilities and reporting H is the application introduced in this work.
Here “harmonic” means satisfying the graph Laplace equation; it does not refer
to a vibrational approximation of molecular free energy.

See [harmonic.py](../felp/diagnostics/harmonic.py) for the implementation and
[the diagnostic guide](overlap_diagnostics.md) for interpreting H alongside U.
