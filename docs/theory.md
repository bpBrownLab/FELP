# FELP theory and method

FELP estimates how a local perturbation redistributes probability over a
reference-sampled conformational landscape. The statistical identity below
motivates the estimator. Restrained endpoint preparation, solvent-model changes
and finite-sample regularization introduce approximations in its implementation.

## Restricted partition functions

Let $x$ denote a configuration, $U_0(x)$ the reference potential, $U_1(x)$ the
perturbed potential, and $\beta=(k_BT)^{-1}$. A state indicator $\chi_i(x)$
defines the restricted partition function

$$
Z_a(i)=\int \chi_i(x)e^{-\beta U_a(x)}\,dx,
\qquad
\pi_a(i)=\frac{Z_a(i)}{\sum_k Z_a(k)}.
$$

For potentials defined on a common configuration space and measure,

$$
\begin{aligned}
Z_1(i)
&=\int \chi_i(x)e^{-\beta U_0(x)}e^{-\beta[U_1(x)-U_0(x)]}\,dx \\
&=Z_0(i)\left\langle e^{-\beta\Delta U(x)}\right\rangle_{0,i},
\end{aligned}
$$

where $\Delta U(x)=U_1(x)-U_0(x)$ and the average is over the reference
equilibrium distribution conditioned on state $i$. Substitution gives

$$
\boxed{
\pi_1(i)=
\frac{\pi_0(i)\left\langle e^{-\beta\Delta U(x)}\right\rangle_{0,i}}
{\sum_k\pi_0(k)\left\langle e^{-\beta\Delta U(x)}\right\rangle_{0,k}}
}.
$$

This is the state-local Zwanzig relation used by FELP. It separates the
reference state population from the perturbation within that state. Applying
it to mutations with different atom sets requires a compatible mapping and
statistical measure; independently prepared endpoints are an approximation to
this identity.

## Finite-sample estimator

For $n_i$ evaluated frames in state $i$, define a common energy shift and weights

$$
s=\min_{i,j}\Delta U_{ij},
\qquad
w_{ij}=e^{-\beta(\Delta U_{ij}-s)},
\qquad
f_i=\frac{1}{n_i}\sum_{j=1}^{n_i}w_{ij}.
$$

Then

$$
\hat\pi_1(i)=\frac{\hat\pi_0(i)f_i}{\sum_k\hat\pi_0(k)f_k},
\qquad
F_1(i)=-k_BT\ln\hat\pi_1(i)+C.
$$

The global shift cancels in normalization and avoids overflow. Independent
statewise shifts would change the relative factors unless their scale were
restored. The constant $C$ sets the lowest finite free energy to zero. The code
uses energies in kJ/mol, so its numerical inverse temperature is $1/(RT)$.

An overall sequence-dependent energy constant also cancels. Thus the output is
a conformational population distribution and its relative free energies, not
an absolute mutation free energy.

## Reference populations and state definitions

The manuscript uses empirical reference populations,
$\hat\pi_0(i)=N_i/\sum_k N_k$, fixed before score-tail filtering. This prevents
the removal of a favorable energy outlier from also redefining the reference
state occupancy.

| State source | Reference population |
|---|---|
| Regular phi/psi grid | Raw reference counts |
| JSON/YAML atlas | Raw counts or supplied `reference_probability` values |
| MSM bundle | Raw counts or loaded stationary probabilities |

For subsampled scoring, the assignment pass and sampling manifest preserve the
full reference counts. Without supplied counts, the raw-count prior is computed
from eligible scored frames before tail filtering. Harmonic exposure separately
uses the original assignments, including frames whose score failed, to identify
visited cells. These distinctions matter when scores are missing; retain the
full reference assignments and counts with any subsampled calculation.

A positive loaded prior requires evaluated frames in that state before QC.
With a raw-count prior, a never-visited state has zero target probability even
after neighbor pooling. FELP cannot create a new conformational basin.

## Restrained endpoint scoring

Each reference frame supplies two branches: the reference sequence and the
mutant sequence. CV targets measured on the original frame are applied to both
branches. Constrained Cartesian FastRelax prepares the structures, followed by
OpenMM minimization and energy evaluation with
`amber/protein.ff14SBonlysc.xml` and `implicit/gbn2.xml`.

The recorded endpoint difference is more precisely

$$
\Delta E_{ij}=U_1(x_{1,ij})-U_0(x_{0,ij}),
$$

where the reference and mutant endpoint coordinates generally differ. The
`delta_u` field stores this quantity. It is a practical local score for the
exponential estimator, rather than a same-coordinate energy difference drawn
from an exact common-Hamiltonian ensemble.

The restraints keep the comparison near the assigned conformation, while the
mobility settings permit local accommodation of the mutation. Production uses
the mutation site's side chain, with system-specific exceptions described in
the [manuscript protocol](manuscript_protocol.md). Minimized structures do not
sample the thermal distribution of local side-chain states. The EGFR
[side-chain sensitivity calculation](../felp/sidechain_ensemble/README.md)
examines that approximation separately.

The reference MD in the paper uses explicit solvent, whereas endpoint scoring
uses implicit solvent. Agreement of weight diagnostics therefore does not
establish agreement between the scoring Hamiltonian and the reference ensemble.

## Stabilizing the observed weights

Exponential averaging can be dominated by a few favorable scores. The
`jctc2026` protocol applies three operations in a fixed order:

1. **Global score-tail cut.** Sort the eligible scores in ascending order and
   remove successive prefixes until the retained weights first give a finite
   ArviZ generalized-Pareto shape estimate $k<0.5$, retaining at least 20 scores.
   The cut is chosen once over the whole dataset, before state-local smoothing.
2. **Statewise PSIS.** Apply ArviZ 0.22.0 Pareto-smoothed importance sampling
   with `reff=1` within each retained state. Restore the state-factor scale after
   ArviZ's weight normalization so factors remain comparable across states.
3. **Neighbor pooling.** Blend each local factor with an ESS-weighted average
   of nearby factors, then normalize the resulting state probabilities.

In the generalized-Pareto tail model, $k<0.5$ corresponds to a finite second
moment. A fitted value describes the observed tail; it cannot establish the
moments or coverage of the unknown full target distribution. Selecting a cut
and smoothing the retained weights changes the finite-sample estimator. These
steps reduce weight concentration but do not remove exponential-average bias
or prove that deleted scores were physically incorrect.

If no cut passes, FELP retains the original sample and records unresolved QC.
Constant identity weights have an undefined tail fit. State-level fits remain
diagnostic and do not trigger another round of deletion.

## Neighbor-pool estimator

For eligible neighbors $\mathcal N(i)$, form

$$
\bar f_i=\frac{\sum_{j\in\mathcal N(i)}\mathrm{ESS}_j f_j}
{\sum_{j\in\mathcal N(i)}\mathrm{ESS}_j},
\qquad
f_i^{\mathrm{blend}}=\alpha_i f_i+(1-\alpha_i)\bar f_i,
\qquad
\alpha_i=\frac{\mathrm{ESS}_i}{\mathrm{ESS}_i+\tau}.
$$

The final distribution is proportional to $\hat\pi_0(i)f_i^{\mathrm{blend}}$.
An L-curve selects $\tau$ by balancing local log-factor misfit against graph
roughness. The assumption is that nearby states have similar perturbation
factors; excessive pooling can blur a genuine change across a basin boundary.
See the [illustrated pooling guide](neighbor_pool_reweighting.md) for the
stencils, objective functions and controls.

## Sampling diagnostics

Local weight ESS is

$$
\mathrm{ESS}_i=\frac{(\sum_j w_{ij})^2}{\sum_j w_{ij}^2}.
$$

The manuscript's unreliable mass is

$$
\mathcal U=\sum_i\hat\pi_1(i)\,\mathbf 1[\mathrm{ESS}_i<10].
$$

It uses local treated-weight ESS even after pooling. It measures how much of
the predicted target population rests on few effective observations. This
weight-based ESS does not correct for trajectory autocorrelation.

Harmonic exposure $H$ addresses a different question: how sensitive is the
predicted landscape to a smooth continuation into cells the reference never
visited? It holds $\sqrt{\hat\pi_1}$ fixed on visited cells and minimizes
Dirichlet energy over the remaining grid. If the continued squared amplitudes
sum to $E$ on unvisited cells, $H=E/(1+E)$. This is exactly the total-variation
distance to that normalized continuation. The
[harmonic-exposure guide](harmonic_exposure.md) gives the derivation.

Small $\mathcal U$ and $H$ are not sufficient conditions for an accurate
prediction. Both use the observed reference and reweighted data. They cannot
detect all missing configurations within a CV cell or determine the true target
probability in an unseen basin. A broad reference can cover a narrower target;
the reverse transformation can fail despite stable observed weights. The
[diagnostics guide](overlap_diagnostics.md) explains how to read these quantities
together.

## Implementation

The [source map](../felp/README.md) locates the scoring and analysis modules.
The main reweighting path is `prep.py` (priors, QC and local factors),
`engines/neighbor_pool.py` (pooling), and `engines/_finalize.py` (populations and
diagnostics) in [felp/reweighting](../felp/reweighting/README.md).
