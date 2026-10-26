#!/usr/bin/env python3
"""Export the PMC sample clouds that the explorer plots live.

`outputs/pmc_diagnostics/<config>/` holds, for a given family, task pool size M
and conditioning, three sample clouds over the latent task:

    <kind>_drift_paths.npz   estimates_at_grid / thetas_at_grid
                             (rollouts, rollout_lengths, coords) -- what PMC
                             recovered from the trained model, at each rollout
                             length, so convergence can be animated.
    <kind>_baselines.npz     baseline_memorising / baseline_generalising
                             (rollouts, coords) -- the two reference posteriors
                             the paper compares against, plus the finite task
                             pool and the population hyper-parameters.

All three are plotted the same way in the browser (histogram / KDE / ECDF), so
this writes them in one uniform shape. Values are quantised to uint16 against a
per-coordinate range and base64'd: a few hundred KB per config instead of a few
MB of JSON, with far more precision than a 1-pixel-wide histogram bin needs.

One file per config, each assigning into `window.PMC_DATA`, so the page can
load a config on demand with a <script> tag and still work over file://.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent
CODE = SITE.parent / "pfn-memorising-generalising"
DIAG = CODE / "outputs/pmc_diagnostics"
# The published page links each configuration's run settings to the copy of
# its run record in the public code repository.
RECORDS_URL = (
    "https://github.com/afiq-aswadi/bft-pmc/blob/main/"
    "paper_data/camera_ready/pmc_diagnostics"
)

N_ROLLOUTS = 512  # subsample; ample for a 60-bin histogram or an ECDF
MARKOV_BLOCK = 4  # show a KxK corner of the 10x10 transition matrix


# Which diagnostics bundle backs each (family, M). The `_n500` runs carry nine
# rollout lengths out to 500 rather than six out to 223, so prefer them.
# LR and BAU use the no-positional-encoding runs, matching the paper's sweeps.
CONFIGS = [
    ("lr", 32, "lr_m32_none_n500"),
    ("lr", 128, "lr_m128_none_n500"),
    ("lr", 256, "lr_m256_none_n500"),
    ("bau", 32, "bau_m32_none_n500"),
    ("bau", 256, "bau_m256_none_n500"),
    ("bau", 512, "bau_m512_none_n500"),
    ("markov", 16, "markov_m16_n500"),
    ("markov", 32, "markov_m32_n500_pi544"),
    ("markov", 64, "markov_m64_n500"),
    ("markov", 128, "markov_m128_n500"),
]

# `estimates_at_grid` for LR and Markov, `thetas_at_grid` for BAU.
PMC_KEY = {"lr": "estimates_at_grid", "bau": "thetas_at_grid", "markov": "estimates_at_grid"}

FAMILY = {
    "lr": {
        "label": "Linear regression",
        "latent": "Regression coefficient",
        "coord_prefix": "w",
        "support": None,  # weights live on the whole line
    },
    "bau": {
        "label": "Balls-and-urns",
        "latent": "Color probability",
        "coord_prefix": "p",
        "support": [0.0, 1.0],
    },
    "markov": {
        "label": "Markov chains",
        "latent": "Transition probability",
        "coord_prefix": "Q",
        "support": [0.0, 1.0],
    },
}


def q16(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> str:
    """Quantise (..., C) float data to uint16 per coordinate, then base64 it."""
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    q = np.round((arr - lo) / span * 65535.0)
    q = np.clip(q, 0, 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def coord_labels(family: str, n: int) -> list[str]:
    if family == "markov":
        k = MARKOV_BLOCK
        return [f"Q({i}→{j})" for i in range(k) for j in range(k)]
    p = FAMILY[family]["coord_prefix"]
    return [f"{p}{i + 1}" for i in range(n)]


def coord_tex(family: str, n: int) -> list[str]:
    """The same names as LaTeX, for KaTeX to typeset in the page."""
    if family == "markov":
        k = MARKOV_BLOCK
        return [f"Q_{{{i},{j}}}" for i in range(k) for j in range(k)]
    p = FAMILY[family]["coord_prefix"]
    return [f"{p}_{{{i + 1}}}" for i in range(n)]


def select_coords(family: str, arr: np.ndarray) -> np.ndarray:
    """Flatten to (..., C), keeping a readable corner of the Markov matrix."""
    if family != "markov":
        return arr
    n = arr.shape[-1]
    k = int(round(n**0.5))
    if k * k != n:
        return arr
    view = arr.reshape(*arr.shape[:-1], k, k)[..., :MARKOV_BLOCK, :MARKOV_BLOCK]
    return view.reshape(*arr.shape[:-1], MARKOV_BLOCK * MARKOV_BLOCK)


def reference_curves(family: str, baselines, n_coords: int, kind: str) -> list | None:
    """Analytic population-prior or generalizing-posterior marginals.

    BAU and Markov place a Dirichlet on the task, so each coordinate's marginal
    is Beta(alpha_c, sum(alpha) - alpha_c) -- for Markov, the sum is over that
    row of the transition matrix. LR's population prior is Gaussian, so each
    marginal is Normal(mean_c, sqrt(cov_cc)).
    """
    files = set(baselines.files)
    if family == "lr" and kind == "prior":
        return [{"kind": "normal", "mu": 0.0, "sigma": 1.0} for _ in range(n_coords)]
    if family == "lr" and {"generalising_mean", "generalising_covariance"} <= files:
        mean = np.asarray(baselines["generalising_mean"]).ravel()
        cov = np.asarray(baselines["generalising_covariance"])
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        return [{"kind": "normal", "mu": float(mean[c]), "sigma": float(sd[c])}
                for c in range(min(n_coords, mean.size))]
    if "alpha" in files:
        alpha = np.array(baselines["alpha"], dtype=float, copy=True)
        if kind == "posterior":
            tokens = np.asarray(baselines["prompt_tokens"], dtype=int).ravel()
            if family == "markov":
                # Match the paper's transition-only likelihood; the initial state is given.
                np.add.at(alpha, (tokens[:-1], tokens[1:]), 1)
            else:
                alpha += np.bincount(tokens, minlength=alpha.size)
        if family == "markov" and alpha.ndim == 2:
            k = MARKOV_BLOCK
            out = []
            for i in range(min(k, alpha.shape[0])):
                row = alpha[i].sum()
                for j in range(min(k, alpha.shape[1])):
                    out.append({"kind": "beta", "a": float(alpha[i, j]),
                                "b": float(row - alpha[i, j])})
            return out
        total = alpha.sum()
        return [{"kind": "beta", "a": float(alpha[c]), "b": float(total - alpha[c])}
                for c in range(min(n_coords, alpha.size))]
    return None


def build_config(family: str, m: int, dirname: str) -> dict | None:
    base = DIAG / dirname
    if not base.exists():
        return None

    conditionings: dict = {}
    coords_n = None
    lengths: list[int] = []
    pool_out = None
    record_path = base / "run_record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    run_config = record.get("config", {})
    run_context = record.get("context", {})
    prompt_length = int(run_config.get("prompt_length", 32))

    for kind, ui in (("prior", "prior"), ("posterior", "posterior")):
        bl_path = base / f"{kind}_baselines.npz"
        dp_path = base / f"{kind}_drift_paths.npz"
        if not (bl_path.exists() and dp_path.exists()):
            continue
        bl, dp = np.load(bl_path), np.load(dp_path)
        key = PMC_KEY[family]
        if key not in dp.files:
            continue

        pmc = select_coords(family, np.asarray(dp[key], dtype=np.float64))
        mem = select_coords(family, np.asarray(bl["baseline_memorising"], dtype=np.float64))
        gen = select_coords(family, np.asarray(bl["baseline_generalising"], dtype=np.float64))

        r = min(N_ROLLOUTS, pmc.shape[0], mem.shape[0], gen.shape[0])
        pmc, mem, gen = pmc[:r], mem[:r], gen[:r]
        coords_n = pmc.shape[-1]
        lengths = [int(v) for v in dp["rollout_lengths"]]

        # One range per coordinate, spanning all three clouds and every rollout
        # length, so a coordinate's axis never moves as the animation plays.
        stack = np.concatenate([pmc.reshape(-1, coords_n), mem, gen], axis=0)
        lo = np.percentile(stack, 0.2, axis=0)
        hi = np.percentile(stack, 99.8, axis=0)
        pad = np.where(hi - lo > 1e-9, 0.12 * (hi - lo), 0.05)
        lo, hi = lo - pad, hi + pad
        support = FAMILY[family]["support"]
        if support is not None:
            lo = np.maximum(lo, support[0])
            hi = np.minimum(hi, support[1])

        conditionings[ui] = {
            "lo": [round(float(v), 6) for v in lo],
            "hi": [round(float(v), 6) for v in hi],
            "n": int(r),
            "pmc": q16(pmc, lo, hi),          # (r, lengths, coords)
            "mem": q16(mem, lo, hi),          # (r, coords)
            "gen": q16(gen, lo, hi),          # (r, coords)
            "reference": reference_curves(family, bl, coords_n, kind),
            "prompt_length": 0 if kind == "prior" else prompt_length,
            "context_limit": (256 if family == "lr" else 255)
                - (0 if kind == "prior" else prompt_length),
        }

        if pool_out is None and "task_pool" in bl.files:
            pool = select_coords(family, np.asarray(bl["task_pool"], dtype=np.float64)
                                 .reshape(int(bl["task_pool"].shape[0]), -1))
            # The pool is the memorising support; keep it small and exact.
            pool_out = [[round(float(v), 5) for v in row] for row in pool[:512]]

    if not conditionings or coords_n is None:
        return None

    return {
        "family": family,
        "M": m,
        "source": dirname,
        "label": f"{FAMILY[family]['label']} · M={m}",
        "latent": FAMILY[family]["latent"],
        "coords": coord_labels(family, coords_n),
        "coords_tex": coord_tex(family, coords_n),
        "n_coords": coords_n,
        "rollout_lengths": lengths,
        "support": FAMILY[family]["support"],
        "pool": pool_out,
        "position_encoding": "RoPE" if family == "markov" else "NoPE",
        "position_scale": run_context.get("rope_position_scale", 1.0),
        "source_url": f"{RECORDS_URL}/{dirname}/run_record.json",
        "conditionings": conditionings,
    }


def main() -> None:
    out_dir = SITE / "data"
    out_dir.mkdir(exist_ok=True)
    index = []
    total = 0

    for family, m, dirname in CONFIGS:
        cfg = build_config(family, m, dirname)
        key = f"{family}_m{m}"
        if cfg is None:
            print(f"SKIP {key}: {dirname} not usable")
            continue
        blob = json.dumps(cfg, separators=(",", ":"))
        path = out_dir / f"cfg_{key}.js"
        path.write_text(
            f"(window.PMC_DATA=window.PMC_DATA||{{}})[{json.dumps(key)}]={blob};\n"
        )
        size = path.stat().st_size
        total += size
        index.append({
            "key": key,
            "family": family,
            "M": m,
            "label": cfg["label"],
            "latent": cfg["latent"],
            "coords": cfg["coords"],
            "coords_tex": cfg["coords_tex"],
            "rollout_lengths": cfg["rollout_lengths"],
            "conditionings": sorted(cfg["conditionings"]),
            "file": f"data/cfg_{key}.js",
            "bytes": size,
            "version": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        })
        print(f"  {key:14s} {cfg['n_coords']:3d} coords  "
              f"{len(cfg['rollout_lengths'])} lengths  {size / 1e3:7.0f} KB")

    manifest = {
        "configs": index,
        "families": {k: {"label": v["label"], "latent": v["latent"],
                         "support": v["support"]} for k, v in FAMILY.items()},
        "n_rollouts": N_ROLLOUTS,
        "markov_block": MARKOV_BLOCK,
    }
    blob = json.dumps(manifest, indent=1)
    (out_dir / "pmc_index.json").write_text(blob)
    (out_dir / "pmc_index.js").write_text(f"window.PMC_INDEX = {blob};\n")
    print(f"\n{len(index)} configs, {total / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
