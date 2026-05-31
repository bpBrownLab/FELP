#!/usr/bin/env python3
"""Render publication-quality FELP workflow schematic to PNG/SVG."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import graphviz
except ImportError:
    print("graphviz Python package required: pip install graphviz", file=sys.stderr)
    sys.exit(1)

HERE = Path(__file__).resolve().parent


def _base_graph(name: str, *, rankdir: str = "TB") -> graphviz.Digraph:
    return graphviz.Digraph(
        name,
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": rankdir,
            "splines": "true",
            "nodesep": "0.35",
            "ranksep": "0.48",
            "bgcolor": "white",
            "pad": "0.3",
            "fontsize": "14",
            "fontname": "Helvetica",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "14",
            "margin": "0.20,0.10",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "color": "#4a5568",
            "arrowsize": "0.65",
        },
    )


def build_vertical() -> graphviz.Digraph:
    g = _base_graph("FELP_workflow_vertical", rankdir="TB")

    with g.subgraph(name="cluster_s1") as c:
        c.attr(label="Stage 1: Reference ensemble & microstates", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("md", "Reference MD trajectory\n(topology + coordinates)", fillcolor="#ebf8ff", color="#2b6cb0")
        c.node("ms", "Microstate atlas\ngrid · JSON/YAML · MSM bundle", fillcolor="#f7fafc", color="#4a5568")
        c.edge("md", "ms")

    with g.subgraph(name="cluster_s2") as c:
        c.attr(label="Stage 2: Assignment & reference prior", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("load", "Load frames", fillcolor="#f7fafc")
        c.node("cv", "Measure CVs · assign microstate i", fillcolor="#f7fafc")
        c.node("prior", "Reference prior π₀(i)", fillcolor="#edf2f7", color="#2d3748")
        c.edges([("load", "cv"), ("cv", "prior")])

    with g.subgraph(name="cluster_s3") as c:
        c.attr(label="Stage 3: Symmetric restrained perturbation (parallel workers)", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("conv", "ParmEd structure → PyRosetta pose", fillcolor="#f7fafc")
        c.node("rest", "Reference-frame restraints\n(backbone dihedrals · distance CVs)", fillcolor="#fffaf0", color="#c05621")

        with c.subgraph(name="cluster_sym") as s:
            s.attr(label="Symmetric PyRosetta preparation (same relax/repack protocol)", style="dashed,rounded", color="#c05621", fontcolor="#c05621", fontsize="12")
            s.node("ref_mut", "Reference branch:\nenforce / mutate original identity\n+ FastRelax or repack", fillcolor="#fffaf0", color="#c05621")
            s.node("pert_mut", "Perturbed branch:\nmutate target residue\n+ FastRelax or repack", fillcolor="#fffaf0", color="#c05621")

        with c.subgraph(name="cluster_openmm") as o:
            o.attr(label="OpenMM reweighting Hamiltonian", style="dashed,rounded", color="#276749", fontcolor="#276749", fontsize="12")
            o.node("ff", "AMBER ff14SBonlysc\n+ GBNeck2 implicit solvent", fillcolor="#f0fff4", color="#276749")
            o.node("u0", "Restrained minimization → U₀\n(reference branch)", fillcolor="#f0fff4", color="#276749")
            o.node("u1", "Restrained minimization → U₁\n(perturbed branch)", fillcolor="#f0fff4", color="#276749")
            o.node("du", "ΔU = U₁ − U₀", fillcolor="#e6fffa", color="#234e52")
            o.edges([("u0", "ff"), ("u1", "ff"), ("ff", "du")])

        c.edges([("conv", "rest"), ("rest", "ref_mut"), ("rest", "pert_mut")])
        c.edge("ref_mut", "u0")
        c.edge("pert_mut", "u1")

        c.node("rosetta_score", "Rosetta scoring (alt.)\nref2015 + constraints", fillcolor="#fffaf0", color="#c05621", style="rounded,filled,dashed")
        c.edge("pert_mut", "rosetta_score", style="dashed", label="optional backend")

    with g.subgraph(name="cluster_s4") as c:
        c.attr(label="Stage 4: State-local reweighting & diagnostics", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("fi", "Local factor fᵢ = ⟨exp(−βΔU)⟩₀,ᵢ", fillcolor="#faf5ff", color="#6b46c1")
        c.node("ess", "ESSᵢ · frame QC · 𝒰", fillcolor="#faf5ff", color="#6b46c1")
        c.node("pool", "Optional: neighbor pooling + L-curve τ\nfᵢ^blend = αᵢ fᵢ + (1−αᵢ) f̄ᵢ", fillcolor="#faf5ff", color="#9f7aea", style="rounded,filled,dashed")
        c.node("pi", "π₁(i) ∝ π₀(i) · fᵢ\n(normalize over states)", fillcolor="#edf2f7", color="#2d3748")
        c.node("out", "Perturbed landscape F₁(i)\nplots · diagnostics", fillcolor="#e2e8f0", color="#2d3748")
        c.edges([("fi", "ess"), ("ess", "pool"), ("pool", "pi"), ("pi", "out")])
        c.edge("fi", "pi", style="dashed", label="default: independent")

    g.edge("ms", "load")
    g.edge("prior", "conv")
    g.edge("du", "fi")
    g.edge("rosetta_score", "fi", style="dashed")
    g.edge("prior", "pi", style="dashed", constraint="false", color="#718096")
    return g


def build_wide() -> graphviz.Digraph:
    """Landscape layout for full-width LaTeX figures."""
    g = _base_graph("FELP_workflow_wide", rankdir="LR")

    with g.subgraph(name="cluster_s1") as c:
        c.attr(label="1 · Reference", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("md", "Reference MD\n(topology + coords)", fillcolor="#ebf8ff", color="#2b6cb0")
        c.node("ms", "Microstate atlas\ngrid · file · MSM", fillcolor="#f7fafc", color="#4a5568")
        c.edge("md", "ms")

    with g.subgraph(name="cluster_s2") as c:
        c.attr(label="2 · Assign", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("cv", "CVs · assign i\nπ₀(i)", fillcolor="#f7fafc")
        c.node("prior", "π₀(i)", fillcolor="#edf2f7", color="#2d3748")
        c.edge("cv", "prior")

    with g.subgraph(name="cluster_s3") as c:
        c.attr(label="3 · Symmetric perturbation (per frame)", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("conv", "ParmEd →\nPyRosetta", fillcolor="#f7fafc")
        c.node("rest", "Reference-frame\nrestraints", fillcolor="#fffaf0", color="#c05621")
        c.node("ref_mut", "Ref: mutate /\nenforce original", fillcolor="#fffaf0", color="#c05621")
        c.node("pert_mut", "Pert: mutate\ntarget", fillcolor="#fffaf0", color="#c05621")
        c.node(
            "openmm",
            "OpenMM (ff14SBonlysc + GBNeck2)\nrestrained min → U₀, U₁\nΔU = U₁ − U₀",
            fillcolor="#f0fff4",
            color="#276749",
        )
        c.edges([("conv", "rest"), ("rest", "ref_mut"), ("rest", "pert_mut")])
        c.edge("ref_mut", "openmm")
        c.edge("pert_mut", "openmm")

    with g.subgraph(name="cluster_s4") as c:
        c.attr(label="4 · Reweight", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="15", fontname="Helvetica-Bold")
        c.node("fi", "fᵢ = ⟨e^{−βΔU}⟩\nESS · QC · 𝒰", fillcolor="#faf5ff", color="#6b46c1")
        c.node("pool", "Optional:\nneighbor pool · L-curve", fillcolor="#faf5ff", color="#9f7aea", style="rounded,filled,dashed")
        c.node("pi", "π₁(i) ∝ π₀(i) · fᵢ\nF₁(i)", fillcolor="#e2e8f0", color="#2d3748")
        c.edge("fi", "pool")
        c.edge("pool", "pi")
        c.edge("fi", "pi", style="dashed", label="default")

    g.edge("ms", "cv")
    g.edge("prior", "conv")
    g.edge("openmm", "fi")
    g.edge("prior", "pi", style="dashed", constraint="false", color="#718096")
    return g


def build_vertical_large_font() -> graphviz.Digraph:
    """Vertical layout with larger font size, equivalent to the wide version."""
    g = graphviz.Digraph(
        "FELP_workflow_vertical_large",
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "splines": "true",
            "nodesep": "0.45",
            "ranksep": "0.55",
            "bgcolor": "white",
            "pad": "0.4",
            "fontsize": "18",
            "fontname": "Helvetica-Bold",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "16",
            "margin": "0.25,0.15",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "13",
            "color": "#4a5568",
            "arrowsize": "0.8",
        },
    )

    with g.subgraph(name="cluster_s1") as c:
        c.attr(label="1 · Reference", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="18", fontname="Helvetica-Bold")
        c.node("md", "Reference MD\n(topology + coords)", fillcolor="#ebf8ff", color="#2b6cb0")
        c.node("ms", "Microstate atlas\ngrid · file · MSM", fillcolor="#f7fafc", color="#4a5568")
        c.edge("md", "ms")

    with g.subgraph(name="cluster_s2") as c:
        c.attr(label="2 · Assign", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="18", fontname="Helvetica-Bold")
        c.node("cv", "CVs · assign i\nπ₀(i)", fillcolor="#f7fafc")
        c.node("prior", "π₀(i)", fillcolor="#edf2f7", color="#2d3748")
        c.edge("cv", "prior")

    with g.subgraph(name="cluster_s3") as c:
        c.attr(label="3 · Symmetric perturbation (per frame)", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="18", fontname="Helvetica-Bold")
        c.node("conv", "ParmEd →\nPyRosetta", fillcolor="#f7fafc")
        c.node("rest", "Reference-frame\nrestraints", fillcolor="#fffaf0", color="#c05621")
        c.node("ref_mut", "Ref: mutate /\nenforce original", fillcolor="#fffaf0", color="#c05621")
        c.node("pert_mut", "Pert: mutate\ntarget", fillcolor="#fffaf0", color="#c05621")
        c.node(
            "openmm",
            "OpenMM (ff14SBonlysc + GBNeck2)\nrestrained min → U₀, U₁\nΔU = U₁ − U₀",
            fillcolor="#f0fff4",
            color="#276749",
        )
        c.edges([("conv", "rest"), ("rest", "ref_mut"), ("rest", "pert_mut")])
        c.edge("ref_mut", "openmm")
        c.edge("pert_mut", "openmm")

    with g.subgraph(name="cluster_s4") as c:
        c.attr(label="4 · Reweight", style="rounded", color="#2c5282", fontcolor="#2c5282", fontsize="18", fontname="Helvetica-Bold")
        c.node("fi", "fᵢ = ⟨e^{−βΔU}⟩\nESS · QC · 𝒰", fillcolor="#faf5ff", color="#6b46c1")
        c.node("pool", "Optional:\nneighbor pool · L-curve", fillcolor="#faf5ff", color="#9f7aea", style="rounded,filled,dashed")
        c.node("pi", "π₁(i) ∝ π₀(i) · fᵢ\nF₁(i)", fillcolor="#e2e8f0", color="#2d3748")
        c.edge("fi", "pool")
        c.edge("pool", "pi")
        c.edge("fi", "pi", style="dashed", label="default")

    g.edge("ms", "cv")
    g.edge("prior", "conv")
    g.edge("openmm", "fi")
    g.edge("prior", "pi", style="dashed", constraint="false", color="#718096")
    return g


def _render(g: graphviz.Digraph, stem: str, *, dpi: int = 300) -> Path:
    g.graph_attr["dpi"] = str(dpi)
    path = Path(g.render(filename=stem, directory=str(HERE), cleanup=True))
    print(f"Wrote {path}")
    return path


def render() -> None:
    _render(build_vertical(), "felp_workflow_schematic", dpi=300)
    _render(build_wide(), "felp_workflow_schematic_wide", dpi=300)
    _render(build_vertical_large_font(), "felp_workflow_schematic_vertical", dpi=300)

    svg = build_wide()
    svg.format = "svg"
    svg_path = Path(svg.render(filename="felp_workflow_schematic_wide", directory=str(HERE), cleanup=False))
    print(f"Wrote {svg_path}")

    svg_vert = build_vertical_large_font()
    svg_vert.format = "svg"
    svg_vert_path = Path(svg_vert.render(filename="felp_workflow_schematic_vertical", directory=str(HERE), cleanup=False))
    print(f"Wrote {svg_vert_path}")

    mmd = HERE / "felp_workflow_schematic.mmd"
    if mmd.exists():
        mmd_png = HERE / "felp_workflow_schematic_mermaid.png"
        cmd = [
            "npx", "-y", "@mermaid-js/mermaid-cli",
            "-i", str(mmd), "-o", str(mmd_png),
            "-b", "white", "-w", "3000", "-H", "2000", "-s", "2",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            print(f"Wrote {mmd_png}")
        except Exception as exc:
            print(f"Mermaid render skipped ({exc}).", file=sys.stderr)


if __name__ == "__main__":
    render()
