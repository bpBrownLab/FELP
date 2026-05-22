# FELP workflow schematic assets

Publication workflow diagram for the manuscript.

## Files

| File | Use |
|------|-----|
| `felp_workflow_schematic.mmd` | Editable Mermaid source |
| `felp_workflow_schematic.png` | Tall vertical layout (Graphviz) |
| `felp_workflow_schematic_300dpi.png` | Same, 300 DPI |
| `felp_workflow_schematic_wide.png` | **Recommended for LaTeX** — landscape, full `\linewidth` |
| `felp_workflow_schematic_wide.svg` | Vector version (Inkscape / Illustrator) |
| `render_felp_workflow_schematic.py` | Regenerate all PNG/SVG outputs |

## LaTeX

```latex
\includegraphics[width=\linewidth]{figures/felp_workflow_schematic_wide.png}
```

For the boxed figure style used elsewhere in the manuscript:

```latex
\setlength{\fboxrule}{0.95pt}\setlength{\fboxsep}{4pt}
\fbox{%
  \begin{minipage}{0.95\linewidth}
    \centering
    \includegraphics[width=0.98\linewidth]{figures/felp_workflow_schematic_wide.png}
    \caption{...}
  \end{minipage}
}
```

## Regenerate

```bash
pip install graphviz   # Python bindings; system `dot` must also be installed
python3 manuscript/figures/render_felp_workflow_schematic.py
```

## Mermaid only

Edit `felp_workflow_schematic.mmd`, then:

- Paste into [Mermaid Live Editor](https://mermaid.live) and export PNG/SVG, or
- `npx -y @mermaid-js/mermaid-cli -i felp_workflow_schematic.mmd -o out.png -b white -w 2800 -s 2`

## Pipeline summary (for caption text)

1. **Reference & microstates** — MD trajectory; partition into grid, file, or MSM states.
2. **Assignment** — measure CVs, assign frames, build π₀(i).
3. **Symmetric perturbation** — ParmEd ↔ PyRosetta; reference-frame restraints; **reference branch** enforces/mutates original identity and **perturbed branch** mutates target (same relax/repack); OpenMM scores both with **ff14SBonlysc + GBNeck2** → ΔU.
4. **Reweighting** — state-local Zwanzig factors, ESS/QC; **optional** neighbor pooling + L-curve (default: independent fᵢ); π₁(i) and F₁(i).
