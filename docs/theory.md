# FELP theory and method

This document tracks the estimator, diagnostics, and assumptions described in the manuscript. Advanced neighbor-pool options are documented separately in [neighbor_pool_reweighting.md](neighbor_pool_reweighting.md).

## Problem statement

FELP estimates how a local perturbation, usually a mutation, redistributes probability over a **reference-sampled** conformational landscape. The method starts from a reference ensemble with a prior over discrete microstates, evaluates restrained reference-to-perturbed energy differences for configurations in each state, and reweights the reference state probabilities.

FELP does not create new conformational basins. It estimates the perturbed distribution over the state space already sampled by the reference trajectory or supplied by an MSM bundle.

## Reference and perturbed state probabilities

Let $x$ denote a configuration, $U_0(x)$ the reference potential, $U_1(x)$ the perturbed potential, and $\beta = (k_B T)^{-1}$. A microstate indicator $\chi_i(x)$ defines the restricted partition function

$$
Z_a(i) = \int \chi_i(x) \exp[-\beta U_a(x)]\,dx
$$

for Hamiltonian $a$. The state probability is

$$
\pi_a(i) = \frac{Z_a(i)}{\sum_k Z_a(k)}
$$

The restricted Zwanzig identity gives

$$
\frac{Z_1(i)}{Z_0(i)}
= \left\langle \exp[-\beta \Delta U(x)] \right\rangle_{0,i},
\qquad
\Delta U(x) = U_1(x) - U_0(x)
$$

so the FELP target distribution is

$$
\pi_1(i) =
\frac{
  \pi_0(i)\left\langle \exp[-\beta \Delta U(x)] \right\rangle_{0,i}
}{
  \sum_k \pi_0(k)\left\langle \exp[-\beta \Delta U(x)] \right\rangle_{0,k}
}
$$

This is the central estimator in the manuscript.

## Finite-sample estimator

For evaluated frames $x_{ij}$ in state $i$, FELP computes stabilized Boltzmann weights

$$
s = \min_{i,j} \Delta U_{ij}
$$

$$
w_{ij} = \exp[-\beta(\Delta U_{ij} - s)]
$$

$$
f_i = \frac{1}{n_i}\sum_j w_{ij}
$$

The shift $s$ is global, so it cancels during normalization. The finite-sample estimate is

$$
\hat{\pi}_1(i) =
\frac{\hat{\pi}_0(i) f_i}
{\sum_k \hat{\pi}_0(k) f_k}
$$

$$
F_1(i) = -k_B T \ln \hat{\pi}_1(i) + C
$$

where $C$ shifts the lowest finite free energy to zero.

## Reference priors and microstates

FELP supports three state sources:

| Source | Reference prior |
|--------|-----------------|
| `grid` | Raw counts over assigned/evaluable reference frames |
| `file` | Loaded `reference_probability` values or raw counts |
| `msm_bundle` | Loaded MSM probabilities or raw counts |

The manuscript benchmarks used raw-count priors from MD trajectories. Loaded priors are useful when state definitions already carry an established stationary distribution.

By default, grid states are target-residue phi/psi bins. File-defined states can use other CV ranges, including distances. MSM bundles provide cluster or metastable states from a PyEMMA workflow.

## Restrained perturbation evaluation

For each reference frame, FELP builds reference and perturbed branches from the same starting pose. CV values measured in the reference pose become restraint targets. The same restraints are applied to both branches so the energy comparison remains local to the assigned state.

In the manuscript implementation, local structure preparation uses PyRosetta constrained FastRelax. Energies are then evaluated in OpenMM using ff14SB sidechain-only parameters with GBNeck2 implicit solvent. The recorded per-frame value is

$$
\Delta U_{ij} = U_1(x_{ij}) - U_0(x_{ij})
$$

Failed or non-finite frames are excluded before reweighting.

## Overlap and validity diagnostics

The local effective sample size is

$$
\mathrm{ESS}_i =
\frac{\left(\sum_j w_{ij}\right)^2}{\sum_j w_{ij}^2}
$$

Low $\mathrm{ESS}_i$ indicates that the local Zwanzig factor depends on too few configurations to be reliable.

FELP estimates the perturbed distribution only over the conformational state space sampled by the reference ensemble. With `raw_counts` priors, states never visited by the reference have $\hat{\pi}_0 = 0$, no evaluated frames, and $\hat{\pi}_1 = 0$. They are absent from the estimated landscape rather than flagged by an overlap diagnostic.

This is the key asymmetry in the manuscript: a broad reference can often predict a narrower target (GLY->ALA), but a narrow reference cannot recover a broader target (ALA->GLY) unless the needed conformations were sampled.

The manuscript summarizes unreliable target probability mass as

$$
\mathcal{U} = \sum_i \hat{\pi}_1(i)\,
\mathbf{1}\{\mathrm{ESS}_i < \epsilon_i\}
$$

The implementation extends this to include state-level QC flags:

$$
\mathcal{U} = \sum_i \hat{\pi}_1(i)\,
\mathbf{1}\{\mathrm{ESS}_i < \epsilon_i \; \mathrm{or}\; q_i = 1\}
$$

$$
\epsilon_i = \max(5, 0.1 n_i)
$$

where $q_i = 1$ marks states flagged for dominant-frame control after QC.

FELP also reports absolute low-ESS masses ($\mathrm{ESS}_i < 10$ and $\mathrm{ESS}_i < 50$) for continuity with earlier analyses. These are advisory rules of thumb. The scale-aware unreliable-target-mass field is the better single summary when frame counts differ across states.

Other overlap fields should be read in context:

| Metric | Use | Caveat |
|--------|-----|--------|
| `target_probability_unreliable_local_support` | Output field for unreliable target probability mass | Cannot see unsampled target-only basins |
| `target_probability_low_ess_mass_ess_lt_10` | Manuscript-style low-ESS mass summary | Absolute threshold; harsh for small states, lenient for large states |
| `max_normalized_weight` | Detects frame-level domination | Global, not state-local |
| `overlap_coefficient` | Describes similarity of reference and target state distributions | Low values can be benign for subset targets such as GLY->ALA |
| global ESS fraction | Detects global weight concentration | Can penalize valid concentration into a target subset |

No scalar diagnostic can prove that an unsampled target basin exists. To assess missing phase space, inspect reference occupancy, compare forward/reverse directions when possible, and use chemical knowledge about whether the target ensemble is broader than the reference.

## Neighbor-pool regularization

The manuscript uses neighbor pooling to stabilize noisy local factors on a discrete grid. For eligible neighbors $\mathcal{N}(i)$,

$$
\bar{f}_i =
\frac{\sum_{j \in \mathcal{N}(i)} \mathrm{ESS}_j f_j}
{\sum_{j \in \mathcal{N}(i)} \mathrm{ESS}_j}
$$

with $\bar{f}_i = f_i$ when no eligible neighbors are available.

$$
\alpha_i = \frac{\mathrm{ESS}_i}{\mathrm{ESS}_i + \tau}
$$

$$
f_i^{\mathrm{blend}} =
\alpha_i f_i + (1 - \alpha_i)\bar{f}_i
$$

Well-sampled states retain their local factors; low-ESS states borrow from adjacent states. $\tau$ is selected by an L-curve criterion over log-factor misfit and graph roughness unless a fixed value is supplied. See [neighbor_pool_reweighting.md](neighbor_pool_reweighting.md).

## Reweighting methods

The verified manuscript workflow uses the independent state-local Zwanzig estimator, optionally regularized with neighbor-pool blend.

All methods use the same prepared data: eligible frames grouped by microstate, reference probabilities, state-local Boltzmann factors, ESS values, and optional grid topology. Final target probabilities always have the form

$$
\pi_{\mathrm{target}}(i) =
\frac{
  \pi_{\mathrm{reference}}(i)\,\mathrm{reweight\_factor}(i)
}{Z}
$$

| Method | Status | Reweight factor |
|--------|--------|-----------------|
| `independent` | Baseline manuscript estimator | $f_i = \langle \exp[-\beta \Delta U]\rangle_i$ |
| `neighbor_pool` | Manuscript regularizer | ESS-weighted blend of $f_i$ with neighboring local factors |

Use `independent` as the baseline. Use `neighbor_pool` for sparse/noisy grid maps when local factors should vary smoothly over neighboring states. Neighbor pooling defaults to `cardinal` grid connectivity.

## Main limitations

FELP is reliable when the target distribution lies within the sampled reference state space and the important target-probability states have adequate local ESS. It can fail when the target requires unsampled conformations, when local overlap is poor, when states mix distinct basins, or when the implicit-solvent mutation scoring model is not appropriate for the physical question.

The manuscript benchmarks emphasize this asymmetry: GLY->ALA and GLY->PRO are well-posed because glycine is conformationally broad, while ALA->GLY fails because alanine does not sample the broad glycine landscape.

## OpenMM platform notes

Use `--cpu` or `--gpu` (or `--platform-name`) to select the OpenMM platform. On CPU, FELP defaults to one OpenMM thread per context to avoid oversubscription and NaN minimization on many-core hosts when running many workers. Override with `--cpu-threads N` when running few workers.
