# Interpreting the diagnostics

FELP reports weight concentration within sampled states and sensitivity to
unvisited CV cells separately. Neither calculation needs target MD. Independent
target trajectories can test the predictions, but are not diagnostic inputs.

| Quantity | What it measures | What it cannot establish |
|---|---|---|
| Global Pareto k | Tail shape of the observed importance weights | Coverage of configurations never sampled |
| Local ESS | Concentration of treated weights within one state | Independent trajectory sample count or score accuracy |
| U | Target mass in states with local ESS < 10 | Target probability outside the visited cells |
| H | Change under square-root harmonic continuation into unvisited cells | True missing target mass or coverage within each CV cell |

## Pareto k and score filtering

Large favorable scores can concentrate the exponential average on very few
frames. Under a generalized-Pareto tail model, $k<0.5$ corresponds to a finite
second moment. The manuscript uses the first global score cut whose fitted k
passes that criterion, then applies statewise PSIS.

This k describes the retained weights. It is not independent validation of the
cut: filtering was chosen using the same criterion. A stable retained tail may
still accompany a wrong landscape. See the
[protocol](manuscript_protocol.md#processing-settings) for how unresolved cuts
and constant identity weights are recorded.

## Local ESS and unreliable mass U

For the treated weights in state $i$,

$$
\mathrm{ESS}_i=\frac{(\sum_j w_{ij})^2}{\sum_j w_{ij}^2},
\qquad
\mathcal U=\sum_i q_i\,\mathbf 1[\mathrm{ESS}_i<10],
$$

where $q_i$ is the final predicted state probability. Equal weights give ESS
equal to the number of retained frames; one dominant weight gives ESS near 1.
The calculation measures weight concentration and does not correct trajectory
autocorrelation.

U focuses attention on states important to the target prediction. Poor local
estimates in negligible-probability states contribute little to it. Neighbor
pooling may stabilize a factor, but U still uses that state's local ESS.

A low global ESS fraction can also arise when a broad reference is reweighted
to a narrower target. That population shift is not automatically an error.
Likewise, amplification of a state's probability above its reference value
is an intended outcome of reweighting, not a failure criterion.

## Harmonic exposure H

H fixes $\sqrt q$ on the cells visited before QC and smoothly continues these
amplitudes over the unvisited cells of the original atlas. It is the
total-variation change after normalizing that continuation. A large H means
the prediction is sensitive to this particular extension of its sampled
support. The [derivation and example](harmonic_exposure.md) show the exact
calculation.

H is not a missing-mass estimate or bound. In particular, a basin with no
sampled connection to favorable observed probabilities can remain invisible
to a smooth continuation. H also cannot resolve omitted coordinates or rare
conformations inside an already visited cell. If every atlas cell was visited,
H is zero regardless of these remaining sampling problems.

The original atlas, counts, spacing and periodicity are part of the diagnostic.
Do not crop the atlas to occupied cells. An unavailable H for an incomplete
or irregular atlas must not be read as zero.

## Reading the outputs together

U and H describe different weaknesses. A prediction can have low U and high
H when its observed weights are stable but its sampled support is narrow.
It can have high U and low H when the CV grid is well covered but important
local averages depend on few frames. Low values of both leave scoring-model
error and unresolved configuration-space sampling as possible failure modes.

This distinction matters for the dipeptide examples: the broad GLY reference
can support narrower targets, whereas ALA→GLY and PRO→GLY require conformations
poorly represented by their references. Stable observed weights alone cannot
answer that coverage question. For proteins, a full-looking projection onto
one or two CVs likewise does not establish full configuration-space overlap.

In `run_summary.json`, read `reweighting_qc` for the global cut and
`overlap_diagnostics.manuscript_diagnostics` for U and H. The adjacent
`harmonic_exposure` object includes calculation status and grid information.
Per-state probabilities and ESS are in `microstate_summary.csv`.
