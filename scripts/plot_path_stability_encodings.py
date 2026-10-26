"""Path stability for LR and balls-and-urns under all three positional encodings.

A 2x3 grid: rows are the two families, columns the encoding. Every panel draws
one curve per prompt realisation, so panels are directly comparable, and the y
axis is shared within a row because the two families sit at different scales.

The comparison is about what happens past the training context. With no
positional encoding the model reads no position, so the trained weights carry
unchanged to N=500. A learned table has no rows beyond the trained context and
is extended by repeating the last trained position, and RoPE supplies rotation
angles for positions never seen in training. Divergence between the columns
after N=223 is therefore a property of the encoding, not of PMC.

    python3 scripts/plot_path_stability_encodings.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from plotting.paper_style import apply_paper_style

SOURCE = REPO / "outputs/pmc_diagnostics/jellyfish_m32_n500/path_stability_per_prompt.csv"
FAMILIES = [("lr", "Linear regression"), ("bau", "Balls-and-urns")]
ENCODINGS = [("learned", "Learned absolute"), ("rope", "RoPE"), ("none", "No positional encoding")]
CONTEXT = 223


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO.parent / "icl2latex" / "figures")
    parser.add_argument("--mark-context", action="store_true",
                        help="draw a rule at the in-context horizon")
    args = parser.parse_args()

    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))

    apply_paper_style(5.5)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.4), sharex=True, sharey="row")
    fig.subplots_adjust(left=.135, right=.99, bottom=.13, top=.90, hspace=.22, wspace=.10)

    for row, (family, family_label) in enumerate(FAMILIES):
        for col, (encoding, encoding_label) in enumerate(ENCODINGS):
            ax = axes[row, col]
            subset = [r for r in rows
                      if r["family"] == family and r["pos_encoding"] == encoding]
            groups: dict[tuple[str, str], list[dict]] = {}
            for r in subset:
                groups.setdefault((r["setup"], r["prompt_idx"]), []).append(r)
            for curve in groups.values():
                curve.sort(key=lambda r: float(r["T"]))
                ax.plot([float(r["T"]) for r in curve],
                        [float(r["scaled_l1"]) for r in curve], color="black", lw=.5)
            if args.mark_context:
                ax.axvline(CONTEXT, color=".55", ls="--", lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.14, lw=.5)
            if row == 0:
                ax.set_title(encoding_label, fontsize=7)
            if row == len(FAMILIES) - 1:
                ax.set_xlabel("Generated observations $N$")
            if col == 0:
                ax.set_ylabel(family_label + "\nScaled $L_1$", linespacing=2.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "reliability_path_stability_encodings.pdf"
    fig.savefig(out, metadata={"Creator": "plot_path_stability_encodings.py"})
    plt.close(fig)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"wrote {out}\n  sha256 {digest}\n  input  {SOURCE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
