# PMC supplementary explorer

An interactive supplement to *What does a Bayes-filtered transformer believe?
A predictive Monte Carlo approach*.

The page plots the **recovered distributions themselves**, computed in the
browser from the PMC sample clouds — not images of plots. Pick a task family, a
task diversity `M`, and a conditioning, and every coordinate of the latent task
is drawn as a density and a CDF against the two references the paper compares
against.

The build scripts read from local checkouts of the code repository
(`../pfn-memorising-generalising`) and the paper sources (`../icl2latex`), which
sit beside this folder. The page itself links only to published copies: the
paper on OpenReview, and the supplement and run records in
[afiq-aswadi/bft-pmc](https://github.com/afiq-aswadi/bft-pmc).

## Viewing it

Open `index.html` — double-clicking works. Nothing is `fetch`ed; the data
arrives as `<script>` globals, and per-config bundles are loaded on demand by
injecting a `<script>` tag, so the offline path is the same as the served one.

To serve it the way GitHub Pages does, run this from this folder:

```bash
python3 -m http.server 8731
```

then open <http://localhost:8731/>.

## What each section does

**01 Explorer** — the real thing. For each configuration the browser holds three
sample clouds over the latent task and bins them live:

| Layer | What it is | Source |
| --- | --- | --- |
| PMC (gold) | finite-rollout estimate of the implicit task distribution | `<kind>_drift_paths.npz` → `estimates_at_grid` / `thetas_at_grid` |
| Memorizing reference (green) | empirical task prior, or its posterior given the prompt | `<kind>_baselines.npz` → `baseline_memorising` |
| Generalizing reference (blue) | population prior, or its posterior given the prompt | `<kind>_baselines.npz` → `baseline_generalising` |

The dotted blue line is the analytic generalizing marginal for the selected
conditioning: Gaussian for LR, Beta for the categorical coordinates. Prior and
posterior parameters are exported separately. The rollout-length slider shows
finite-rollout sensitivity; neither a short rollout nor a stable estimate proves
recovery of a particular reference. The displayed coordinate distances are
binned, one-dimensional Wasserstein-1 estimates, distinct from full-vector ED/SW.

Configurations with PMC bundles: LR at M = 32/128/256, balls-and-urns at
32/256/512, Markov at 16/32/64/128 — each with prior and posterior, nine rollout
lengths, 512 displayed draws per distribution (subsampled from 1024). Each posterior
configuration uses one in-distribution prompt of length 32. Markov shows the 4×4
corner of the 10×10 transition matrix. All displayed Markov runs use position
scale 256/544; M=32 uses `markov_m32_n500_pi544` rather than the older 256/1024 run.
Lengths beyond 224 pairs (LR posterior) or 223 observations (categorical posterior)
exceed the training context. Prior limits are 256 pairs and 255 observations.

**02 Comparisons** — independent family and metric selectors. ED and SW are shown
separately. Prior predictive comparisons concern prediction space, not recovered
latent priors. Markov training dynamics displays the current paper figure: the
previous `id_kl` / `ood_kl` monitor is not a reference-posterior comparison, and the
available latent-metric CSV conflicts with the paper's stated ED ranking. Do not
restore interactive curves until the figure's numeric source has been verified.

**03 Figure archive** — web-sized marginal figures across the available task
diversities, with conditioning labels. The full-resolution originals stay in the
authors' workspace.

The supplementary TeX still labels some LR and Markov marginal prompts as length
8, while the main paper and selected diagnostic runs use 32. Archive captions do
not infer a prompt length from these conflicting descriptions. The paper sources
were not edited as part of the webpage copy revision.

## Deploying

The site is published from the `gh-pages` branch of
[afiq-aswadi/bft-pmc](https://github.com/afiq-aswadi/bft-pmc); see
[DEPLOY.md](DEPLOY.md).

## Rebuilding

```bash
python3 build_pmc.py     # sample clouds -> data/cfg_*.js   (~4 MB, seconds)
python3 build_site.py    # figures + metric series          (~25 MB, minutes)
```

Needs `numpy` and `pillow`. `build_site.py --no-figures` refreshes only the data.

Sample values are quantised to uint16 against a per-coordinate range; the
round-trip error is ~5e-6, far below a pixel. LR and balls-and-urns read the
no-positional-encoding runs, per the 2026-09-20 correction in the code
repository's [`docs/figure_provenance.md`](https://github.com/afiq-aswadi/bft-pmc/blob/main/docs/figure_provenance.md) —
`paper_data/{lr,bau}/sweep/` is superseded and deliberately unused.

## Layout

```
index.html            page shell (asset URLs are content-stamped on build)
assets/styles.css     monochrome editorial theme, light only
assets/plot.js        canvas primitives: binning, density/CDF, Beta & normal, axes
assets/app.js         explorer, sweep charts, figure archive, maths
vendor/katex/         KaTeX, vendored woff2-only so maths works offline
build_pmc.py          PMC sample clouds   -> data/cfg_*.js, data/pmc_index.*
build_site.py         figures + metrics   -> figures/, data/figures.*, data/series.*
data/*.json           citable data export
data/*.js             the same payloads as globals, for file:// use
figures/<family>/     web-sized WebP copies of the paper's renders
```

### A note on the y-axis

The memorizing posterior can concentrate on one or several training tasks. Its
histogram peak height depends on bin width. Panels scale to PMC and the
generalizing reference and allow tall memorizing bars to clip — the same rule as
the code repository's
[`bft_pmc/plotting/marginal_cell.py`](https://github.com/afiq-aswadi/bft-pmc/blob/main/bft_pmc/plotting/marginal_cell.py).

### Maths

Formulas are written as `data-tex` attributes and typeset by KaTeX, vendored
under `vendor/katex/` (woff2 only, ~590 KB) so the page needs no network. They
inherit the colour and font size of the surrounding text rather than KaTeX's
1.21em default.

Every formula slot is pre-filled with a Unicode fallback before KaTeX runs, so
if the library ever fails to load the maths degrades to `Π<sub>M</sub>` rather
than disappearing. `build_site.py` stamps every local asset URL with a content
hash for the same reason: a cached script can otherwise pair with a newer page.

### Plot conventions

- The y-axis is computed once per coordinate, over **all** layers and **all**
  rollout lengths. Neither the layer toggles nor the rollout slider can move
  it, so dragging the slider shows PMC sharpening rather than a curve of
  constant height against a shifting scale.
- Archive figures are capped at 880px wide and 78vh tall. They are print
  figures whose type is sized for a few inches of paper; stretched to the full
  column their labels become poster-sized. Click enlarges the web-sized copy.
- The left margin of a line chart is measured from the widest tick label plus
  the rotated axis title, rather than being a fixed guess.

### Colour

Text uses four levels of emphasis — heading/value, running text, label, and
switched-off — declared once in `:root` as `--ink` … `--ink-4`. Nothing in the
stylesheet picks its own grey.

The dark control rail mirrors those levels as `--on-dark-1…3`, but only level 2
has a value of its own; levels 1 and 3 reuse `--paper` and `--ink-3`, because a
second near-identical grey is noise, not hierarchy.

The levels are assigned by *role*, not by taste, and the rule is worth keeping
to when editing:

| Level | Used for |
| --- | --- |
| 1 | headings, numbers, the selected item |
| 2 | every sentence — lead, intros, captions, notes — and choosable text |
| 3 | every label — eyebrows, control names, column headings, metadata |
| 4 | hints and switched-off state, nothing else |

A caption is a sentence, so it sits at level 2 like any other prose and is set
apart by size, not by colour.

The dark rail is the deliberate exception: its text is white throughout, so the
panel reads as one surface rather than four shades of grey. On near-black, a
mid grey does not read as secondary — only as dirty white.

That includes switched-off curves, so the swatch alone carries on/off: it drops
to 18% opacity rather than merely dimming.

Colour otherwise belongs to the data: goldenrod, green and blue are reserved
for PMC, Π_M and Π_∞ and are never used for interface chrome. Rendered, the
page uses six greys and those three.
