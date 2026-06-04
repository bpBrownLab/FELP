# Neighbor-pool reweighting

Neighbor-pool reweighting is a **spatially coupled** extension of the state-local Zwanzig estimator. After per-bin local Boltzmann factors are computed from QC-filtered frames, the method adjusts those factors using information from **adjacent bins on the CV grid** before reconstructing target probabilities.

Covers the `neighbor_pool` backend in `felp summarize` / `felp run`: the **blend** solver, automatic **L-curve** τ selection, the legacy **GCV** score, and the optional **variational** solver.

## Role in the FELP pipeline

For each microstate (grid bin) `i`, FELP first computes the usual independent quantities:

| Symbol | Meaning |
|--------|---------|
| `f_i^local` | State-local mean Boltzmann factor (Zwanzig average with global log shift) |
| `ESS_i` | Effective sample size of the state-local weights |
| `π_i^ref` | Reference prior probability |

Neighbor pooling produces a **reweight factor** `f_i` that may differ from `f_i^local`. Target probabilities follow the same normalization as independent Zwanzig:

```text
Z = Σ_i π_i^ref · f_i
π_i^target = π_i^ref · f_i / Z
```

Plotting and diagnostics use the same `microstate_summary.csv` columns (`reweight_factor`, `target_probability`, etc.) as other backends.

## Grid topology

Coupling requires a neighbor graph on microstates:

1. **Preferred:** regular 1D/2D/3D CV-grid microstates (`bin_indices` or uniform `cv_ranges`).
   - **`chebyshev3` (default):** Chebyshev (L_inf) ball of radius 3 in bin-index space
     (6 neighbors in 1D, 48 in 2D, 124 in 3D interior cells).
   - **`cardinal`:** face-sharing bins (2 in 1D, 4 in 2D, 6 in 3D).
   - **`chebyshev` / `chebyshevN`:** Chebyshev ball with explicit radius `N`. `chebyshev1`
     adds diagonals (8 neighbors in 2D); `chebyshev2` adds the next shell (24 in 2D).
2. **Fallback:** k-nearest neighbors in CV space, capped by `--neighbor-pool-max-neighbors`
   (default 4). Used for irregular k-means MSM atlases; grid-connectivity flags are
   ignored. Grid-based MSM bundles (`msm-fit --clustering-method grid`) use the same
   rectilinear topology as file/grid FELP microstates.

The manuscript benchmarks used cardinal grid neighbors. The current CLI default is
`chebyshev3`; pass `--neighbor-pool-grid-connectivity cardinal` to reproduce the
manuscript neighbor stencil.

If no topology is available, neighbor-pool degrades to local factors and records a warning in `run_summary.json`.

## Solver: blend (default, recommended)

The **blend** solver is the manuscript estimator. It performs **per-bin** partial pooling toward an ESS-weighted average of **neighbors’ local factors** (not a global linear solve).

### Blending formula

```text
w_i = ESS_i / (ESS_i + τ)

f_i = w_i · f_i^local + (1 - w_i) · f̄_{N(i)}^local
```

where `f̄_{N(i)}^local` is the ESS-weighted mean of `f^local` over graph neighbors of `i` that have positive ESS and positive local factor.

### What τ means (blend)

**τ is not** the number of frames, samples, or grid bins. It is a **dimensionless pseudocount** on the same scale as `ESS_i`:

| τ vs ESS_i | Effect |
|------------|--------|
| τ ≪ ESS_i | `w_i → 1` → stay local |
| τ ≫ ESS_i | `w_i → 0` → trust neighbor pool |
| τ ~ ESS_i | comparable local vs pooled contribution |

Example: `ESS_i = 10`, `τ = 50` gives `(1 - w_i) = 50/60 ≈ 83%` weight on the neighbor pool (strong spatial smoothing).

τ controls **how much each bin trusts its own noisy local average versus its neighbors’ local averages**. It does not change who the neighbors are; the grid definition does.

### Recorded metadata (blend)

Per state (in aggregation metadata):

- `raw_local_factor`
- `pooled_neighbor_factor`
- `pooling_weight` (= `w_i`)
- `neighbor_state_ids`
- `local_ess`

Run-level (`run_summary.json` → `neighbor_pool`):

- `pool_solver`: `"blend"`
- `tau`, `tau_mode`
- τ-selection diagnostics (see below)

## Automatic τ selection (blend)

When `--neighbor-pool-tau` is omitted, τ is chosen by `--neighbor-pool-tau-mode`.

### L-curve (default: `lcurve`)

For each candidate τ on a log-spaced grid from `1` to `10 × max(ESS)` (30 points):

1. Apply the blend formula and compute  
   **misfit** = `Σ_i ESS_i · (log f_i^local - log f_i)²`
2. Compute **roughness** = `½ Σ_{(i,j) ∈ edges} (log f_i - log f_j)²` on the grid graph
3. Form the curve `(misfit(τ), roughness(τ))`

The chosen τ is the **corner** of this curve: the point farthest from the straight chord joining the endpoints, evaluated in **log₁₀(misfit)** vs **log₁₀(roughness)** space (Hansen L-curve heuristic).

**Why L-curve:** It balances fit and smoothness without an arbitrary multiplier (the legacy GCV score used `misfit + median(n_frames) × roughness`).

**Stored diagnostics:**

- `lcurve_scan`: list of `{tau, misfit, roughness}` for all candidates
- `lcurve_corner_index`, `lcurve_corner_misfit`, `lcurve_corner_roughness`

Use `lcurve_scan` to plot the tradeoff and verify a clear elbow. On very sparse or low-overlap datasets the curve may be shallow; in that case prefer a fixed τ chosen from a small sweep.

### GCV (`gcv`, legacy)

Minimizes:

```text
score(τ) = misfit(τ) + median(n_frames_per_bin) × roughness(τ)
```

over the same τ grid. Retained for backward compatibility and comparison with earlier runs. The `median(n_frames)` scale (~60 on dense trajectories) is **not** derived from ESS theory.

### ESS summaries (`mean_ess`, `median_ess`)

Set τ to the mean or median of positive bin ESS values (no optimization). Useful as a simple scale-aware default when L-curve/GCV are untrusted.

### Fixed τ (`--neighbor-pool-tau <value>`)

Skips automatic selection; `tau_mode` is recorded as `"fixed"`. Recommended when you want reproducible smoothing strength across projects (e.g. τ = 50 after a documented sweep).

## Optional: `--neighbor-pool-min-local-weight`

Default `0.0` (no effect). When `> 0`, after computing `f_i` the code enforces:

```text
w_i ← max(w_i, w_min)
f_i ← w_i · f_i^local + (1 - w_i) · f_i
```

This is a post-hoc guardrail to prevent extreme pooling in very low-ESS bins. It reuses the blend weight definition.

## Solver: variational (optional, experimental)

```bash
--neighbor-pool-solver variational
```

The variational solver replaces the per-bin blend with a **coupled Gaussian Markov random field** on log-factors `y_i = log f_i`:

```text
min_y  Σ_i ESS_i · (y_i - y_i^local)²  +  (1/τ) · ½ yᵀ L y
```

solved as `(diag(ESS) + L/τ) y = diag(ESS) y^local`, then `f_i = exp(y_i)`.

Here **large τ** means **weak** neighbor coupling (→ independent limit), opposite in spirit to blend τ.

Automatic τ (`--neighbor-pool-tau-mode variational`) minimizes the profile energy `F*(τ)` and often drives τ to the **upper search bound** on sparse, noisy data (e.g. chignolin), collapsing to independent Zwanzig. **Do not use variational auto-τ** unless you are explicitly studying that model.

Use **blend** for spatial smoothing of discrete maps. Use **variational** only with fixed τ and clear validation if exploring the coupled log-factor model.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--reweighting-methods neighbor_pool` | omit | Enable this backend (can combine with `independent`, `laplacian`, `frame`) |
| `--neighbor-pool-solver` | `blend` | `blend` or `variational` |
| `--neighbor-pool-tau` | *(omit)* | Fixed τ; skips auto selection |
| `--neighbor-pool-tau-mode` | `lcurve` | `lcurve`, `gcv`, `mean_ess`, `median_ess` (blend); `variational` (+ `gcv` alias) for variational solver |
| `--neighbor-pool-min-local-weight` | `0` | Minimum local blend weight |
| `--neighbor-pool-max-neighbors` | `4` | kNN cap for non-grid atlases |
| `--neighbor-pool-grid-connectivity` | `chebyshev3` | Grid stencil: `cardinal`, `chebyshev`, or `chebyshevN` (1D/2D/3D) |
| `--neighbor-pool-chebyshev-radius` | `3` | Radius when connectivity is `chebyshev` (ignored for `chebyshevN` aliases) |

Multi-method summarize writes each backend under `reweighting_<method>/` plus `reweighting_comparison.json`.

### Example: auto L-curve (blend)

```bash
python felp.py summarize \
  --raw-frame-results path/to/frame_results.csv \
  --output-dir path/to/out \
  --microstate-source file \
  --microstate-file path/to/auto_microstates.json \
  --prior-probability-mode raw_counts \
  --repartition \
  --temperature-kelvin 310 \
  --reweighting-qc-policy exclude-frames \
  --reweighting-methods independent neighbor_pool laplacian frame \
  --neighbor-pool-solver blend \
  --neighbor-pool-tau-mode lcurve
```

(`--neighbor-pool-solver blend` and `--neighbor-pool-tau-mode lcurve` are optional; they are the defaults.)

### Example: fixed τ = 50 (blend)

```bash
  --neighbor-pool-solver blend \
  --neighbor-pool-tau 50
```

### Example: compare L-curve vs GCV

Run two output directories differing only in `--neighbor-pool-tau-mode lcurve` vs `gcv`, then compare `reweighting_neighbor_pool/run_summary.json` (`tau`, `lcurve_scan` or `gcv_score`) and state-level FEL plots (`reweighted_*_state_free_energy_landscape.png`).

## Relation to other coupled backends

| Backend | Variable smoothed | Coupling |
|---------|-------------------|----------|
| **independent** | none | per-bin Zwanzig only |
| **neighbor_pool (blend)** | `f_i` | local ↔ neighbor **local** factors |
| **laplacian** | `ΔG_i` | MAP with `λ (k_B T)² Σ (∇ΔG)²` on grid |
| **frame** | global SNIS | continuous frame weights, then bin |

Neighbor-pool (blend) is the lightest-touch spatial regularizer: it does not re-optimize energies and preserves the Zwanzig interpretation of `f_i^local` while sharing information across the grid.

## Implementation

- Engine: `felp/reweighting/engines/neighbor_pool.py`
- Topology: `felp/reweighting/topology.py`
- Config: `NeighborPoolConfig` in `felp/core/models.py`
