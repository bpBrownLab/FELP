"""Harmonic exposure on a regular CV grid.

Visited cells fix sqrt(q), including visited cells with zero target probability.
Only originally unvisited cells are continued. This is a sensitivity to an
assumed smooth continuation, not an estimate of physical missing target mass.
The diagnostic uses a cardinal graph, independently of the pooling graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def dirichlet_matrix(shape, periodic, steps):
    """Weighted grid Laplacian with c_ij = 1 / axis_spacing**2.

    Nonperiodic boundaries have only their in-domain edges. On a two-cell
    periodic axis the two directions are parallel edges, as in the published
    finite-difference construction.
    """
    shape = tuple(shape)
    flags = np.asarray(periodic)
    steps = np.asarray(steps, dtype=float)
    if (not 1 <= len(shape) <= 2
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                   or n < 1 for n in shape)
            or flags.shape != (len(shape),) or flags.dtype.kind != "b"
            or steps.shape != (len(shape),) or not np.isfinite(steps).all()
            or np.any(steps <= 0)):
        raise ValueError("Require a uniform 1D/2D grid, boolean periodic flags and positive axis spacings")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        conductance = (1. / steps) ** 2
    if not np.isfinite(conductance).all() or np.any(conductance <= 0):
        raise ValueError("Axis spacings are outside the supported floating-point range")
    rows, cols, values = [], [], []
    for coord in np.ndindex(shape):
        i = int(np.ravel_multi_index(coord, shape))
        degree = 0.
        for axis, n in enumerate(shape):
            if n == 1:
                continue
            for offset in (-1, 1):
                other = list(coord)
                other[axis] += offset
                if flags[axis]:
                    other[axis] %= n
                elif not 0 <= other[axis] < n:
                    continue
                rows.append(i)
                cols.append(int(np.ravel_multi_index(tuple(other), shape)))
                values.append(-conductance[axis])
                degree += conductance[axis]
        rows.append(i)
        cols.append(i)
        values.append(degree)
    size = math.prod(shape)
    return coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()


def harmonic_exposure(counts, q, shape, periodic, steps):
    """Return H = E/(1+E) on a full Cartesian atlas flattened in C order.

    Counts must be from before score QC. The probability vector must be
    normalized on that original support. No target MD or target-specific
    physical constraints enter this calculation.
    """
    counts = np.asarray(counts, dtype=float)
    q = np.asarray(q, dtype=float)
    if (counts.ndim != 1 or not counts.size or q.shape != counts.shape
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts)) or not np.any(counts > 0)):
        raise ValueError("Require original nonnegative integral reference counts")
    if (not np.isfinite(q).all() or np.any(q < 0)
            or not np.isclose(q.sum(), 1., atol=1e-12, rtol=0)
            or np.any(q[counts == 0] > 0)):
        raise ValueError("Require normalized q on original reference support")
    laplacian = dirichlet_matrix(shape, periodic, steps)
    if laplacian.shape[0] != counts.size:
        raise ValueError("Counts and probabilities must include every grid cell")
    visited = counts > 0
    free = np.flatnonzero(~visited)
    anchors = np.sqrt(q)
    amplitude = anchors.copy()
    residual = 0.
    if len(free):
        scale = float(laplacian.diagonal().max())
        if not np.isfinite(scale) or scale <= 0:
            raise ArithmeticError("Invalid Dirichlet operator scale")
        scaled = laplacian / scale
        rhs = -(scaled @ anchors)[free]
        solution = splu(scaled[free][:, free].tocsc()).solve(rhs)
        if not np.isfinite(solution).all() or solution.min() < -1e-12:
            raise ArithmeticError("Harmonic continuation is not nonnegative")
        amplitude[free] = np.maximum(solution, 0.)
        residual = float(np.max(np.abs((scaled @ amplitude)[free])))
        if residual > 1e-10:
            raise ArithmeticError("Harmonic continuation failed stationarity")
        if (amplitude[free].min() < anchors[visited].min() - 1e-10
                or amplitude[free].max() > anchors[visited].max() + 1e-10):
            raise ArithmeticError("Harmonic continuation failed the maximum principle")
    if not np.array_equal(amplitude[visited], anchors[visited]):
        raise ArithmeticError("A visited amplitude was changed")
    extra = float(amplitude[free] @ amplitude[free])
    return {
        "H": extra / (1. + extra), "E": extra,
        "status": "computed" if len(free) else "no_unvisited_cells",
        "unvisited_state_count": int(len(free)),
        "stationarity_residual": residual,
        "physical_missing_mass_estimate": False,
    }


@dataclass(frozen=True)
class AtlasGrid:
    """Physical cell order established from full, uniform CV ranges."""

    state_ids: tuple[str, ...]
    cv_names: tuple[str, ...]
    shape: tuple[int, ...]
    steps: tuple[float, ...]


def grid_from_atlas(atlas):
    """Validate a complete equal-volume atlas; never bridge missing grid cells.

    Sorting opaque state IDs or compressing sparse bin indices can change the
    physical graph. Instead use the actual interval bounds on every axis.
    Irregular, incomplete and MSM atlases must supply an explicit grid through
    the standalone diagnostic, if a defensible grid exists.
    """
    definitions = atlas.definitions
    if not definitions:
        raise ValueError("The atlas has no states")
    names = tuple(sorted(definitions[0].cv_ranges))
    if not 1 <= len(names) <= 2:
        raise ValueError("Harmonic exposure requires a complete uniform 1D/2D CV atlas")
    if len({d.state_id for d in definitions}) != len(definitions):
        raise ValueError("Atlas state IDs must be unique")
    intervals = []
    for definition in definitions:
        if tuple(sorted(definition.cv_ranges)) != names:
            raise ValueError("All atlas states must have the same CV axes")
        bounds = tuple(tuple(map(float, definition.cv_ranges[name])) for name in names)
        if any(len(b) != 2 or not np.isfinite(b).all() or b[1] <= b[0] for b in bounds):
            raise ValueError("Atlas CV intervals must be finite with increasing bounds")
        intervals.append(bounds)
    axes = [sorted({bounds[i] for bounds in intervals}) for i in range(len(names))]
    steps = []
    for axis in axes:
        edges = np.asarray(axis)
        widths = edges[:, 1] - edges[:, 0]
        if (not np.allclose(widths, widths[0], rtol=1e-9, atol=1e-12)
                or not np.allclose(edges[:-1, 1], edges[1:, 0], rtol=1e-9, atol=1e-12)):
            raise ValueError("Atlas must contain contiguous, uniformly spaced intervals")
        steps.append(float(widths[0]))
    shape = tuple(map(len, axes))
    lookup = [{bounds: i for i, bounds in enumerate(axis)} for axis in axes]
    indexed = {}
    for definition, bounds in zip(definitions, intervals):
        index = tuple(lookup[i][b] for i, b in enumerate(bounds))
        if index in indexed:
            raise ValueError("Multiple states occupy the same physical grid cell")
        indexed[index] = definition.state_id
    if len(indexed) != math.prod(shape):
        raise ValueError("The atlas omits grid cells; H is unavailable, not zero")
    return AtlasGrid(tuple(indexed[index] for index in np.ndindex(shape)),
                     names, shape, tuple(steps))


def harmonic_exposure_for_atlas(atlas, counts, probabilities, periodic_cv_names=()):
    """Adapt FELP state mappings, reporting unavailable geometry explicitly."""
    try:
        grid = grid_from_atlas(atlas)
    except ValueError as exc:
        return {"H": None, "status": "unavailable_atlas", "reason": str(exc),
                "physical_missing_mass_estimate": False}
    unknown = set(periodic_cv_names) - set(grid.cv_names)
    if unknown:
        raise ValueError("Unknown periodic CVs: " + ", ".join(sorted(unknown)))
    periodic = tuple(name in periodic_cv_names for name in grid.cv_names)
    result = harmonic_exposure([counts.get(s, 0) for s in grid.state_ids],
                               [probabilities.get(s, 0.) for s in grid.state_ids],
                               grid.shape, periodic, grid.steps)
    result["grid"] = {"shape": list(grid.shape), "cv_names": list(grid.cv_names),
                      "steps": list(grid.steps), "periodic": list(periodic),
                      "connectivity": "cardinal", "flattening_order": "C"}
    return result


def diagnose_payload(payload):
    """Closed JSON interface for saved reference counts and FELP probabilities."""
    required = {"schema_version", "state_ids", "state_counts", "q", "grid_shape", "periodic", "steps"}
    if (not isinstance(payload, dict) or not required <= payload.keys()
            or payload.keys() - required - {"local_ess"}):
        raise ValueError("Expected schema_version, state_ids, state_counts, q, grid_shape, periodic, steps and optional local_ess")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Require schema_version=1")
    ids = payload["state_ids"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids) or len(ids) != len(payload["state_counts"])):
        raise ValueError("Require distinct nonempty state_ids in physical grid C order")
    result = harmonic_exposure(payload["state_counts"], payload["q"], payload["grid_shape"],
                               payload["periodic"], payload["steps"])
    result.update(schema_version=1, U=None, ess_floor=10,
                  interpretation="Sensitivity to assumed square-root harmonic continuation; not a coverage certificate")
    if "local_ess" in payload:
        ess = np.asarray(payload["local_ess"], dtype=float)
        if ess.shape != (len(ids),) or not np.isfinite(ess).all() or np.any(ess < 0):
            raise ValueError("local_ess must be finite, nonnegative and aligned with state_ids")
        result["U"] = float(np.asarray(payload["q"])[ess < 10].sum())
    return result
