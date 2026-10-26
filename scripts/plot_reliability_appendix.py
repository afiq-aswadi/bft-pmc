"""Redraw the reliability appendix and supplement from saved diagnostic CSVs.

No checkpoints or model generation are needed. Run from any directory:
    python3 scripts/plot_reliability_appendix.py
The default destination is the sibling icl2latex checkout. Every panel's
CSV and run-record hashes are saved in reliability_figure_manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from plotting.paper_style import apply_paper_style

ROOT = REPO / "outputs/pmc_diagnostics"
COLORS = {"memorising": "#2ca02c", "generalising": "#1f77b4"}
NAMES = {"lr": "Linear regression", "bau": "Balls-and-urns", "markov": "1-Markov"}
MANIFEST = []


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def provenance(path):
    record = path.parent / "run_record.json"
    result = {"path": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if record.exists():
        result["run_record"] = str(record.relative_to(REPO))
        result["run_record_sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    return result


def make_axes(nrows):
    apply_paper_style(5.5)
    fig, axes = plt.subplots(nrows, 2, figsize=(5.5, 0.65 + 1.35 * nrows), squeeze=False)
    fig.subplots_adjust(left=.15, right=.985, bottom=.15 if nrows==2 else .10, top=.91, hspace=.48, wspace=.36)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.14, lw=.5)
        ax.tick_params(length=2.5, width=.6)
    return fig, axes


def legend(fig, kind, mark_context=True):
    handles = [Line2D([], [], color=c, marker="o", ms=3, lw=1.15, label=n.capitalize().replace("ising", "izing")) for n, c in COLORS.items()]
    if kind == "length" and mark_context:
        handles += [Line2D([], [], color=".4", ls="--", lw=.8, label="$N=223$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, 1), ncol=len(handles), frameon=False, columnspacing=1.2, handlelength=1.7)


def finish(fig, name, inputs, panels, destination):
    out = destination / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Creator": "plot_reliability_appendix.py"})
    plt.close(fig)
    MANIFEST.append({"figure": "figures/" + name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "inputs": [provenance(p) for p in dict.fromkeys(inputs)], "panels": panels})


def curves(specs, source, kind, name, destination, mark_context=True):
    # mark_context draws the N=223 training-context line; runs that stop at
    # N=223 pass False, since the line would sit on their last point.
    fig, axes = make_axes(len(specs))
    panels, inputs = [], []
    for i, (label, run) in enumerate(specs):
        path = ROOT / run / f"{source}_recovery_vs_{kind}.csv"
        data = rows(path)
        inputs.append(path)
        xkey = "rollout_length" if kind == "length" else "num_rollouts"
        x = np.array([float(r[xkey]) for r in data])
        for j, metric in enumerate(["ed", "sw"]):
            ax = axes[i, j]
            panel = {"label": label, "source": source, "metric": metric, "x": x.tolist(), "series": {}}
            for ref, color in COLORS.items():
                key = f"dist/{metric}_vs_baseline_{ref}"
                if kind == "count":
                    y = np.array([float(r[key + "/mean"]) for r in data])
                    sd = np.array([float(r[key + "/sd"]) for r in data])
                    ax.errorbar(x[:-1], y[:-1], yerr=sd[:-1], fmt="none", ecolor=color, lw=.65, capsize=1.4)
                    panel["series"][ref + "_sd"] = sd.tolist()
                else:
                    y = np.array([float(r[key]) for r in data])
                ax.plot(x, y, color=color, marker="o", ms=2.5, lw=1.05)
                panel["series"][ref] = y.tolist()
            ax.set_xscale("log", base=2)
            ticks = [8, 32, 128, 500] if kind == "length" and max(x)>223 else ([8, 32, 128, 223] if kind=="length" else [16, 64, 256, 1024])
            ax.set_xticks(ticks, [str(v) for v in ticks])
            ax.set_xlabel("Rollout length $N$" if kind == "length" else "Rollout count $R$")
            ax.set_ylabel((label + "\n" if j == 0 else "") + ("Energy distance" if metric == "ed" else "Sliced Wasserstein-1"), linespacing=2.0)
            if kind == "length":
                if mark_context:
                    ax.axvline(223, color=".4", ls="--", lw=.8)
                values = [v for key, vals in panel["series"].items() for v in vals]
                if min(values) > 0 and max(values) / min(values) > 30:
                    ax.set_yscale("log")
            else:
                ax.axhline(0, color=".65", lw=.5, zorder=0)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            panels.append(panel)
    legend(fig, kind, mark_context)
    finish(fig, name, inputs, panels, destination)


def aggregate(run, name, destination, xkey):
    path = ROOT / run / ("recovery_vs_dimension.csv" if xkey == "latent_dim" else "recovery_vs_task_diversity.csv")
    data = rows(path)
    fig, axes = make_axes(2)
    panels=[]
    for i, source in enumerate(["prior", "posterior"]):
        subset = sorted((r for r in data if r["source"] == source), key=lambda r: float(r[xkey]))
        x = [float(r[xkey]) for r in subset]
        for j, metric in enumerate(["ed", "sw"]):
            ax=axes[i,j]
            panel={"source":source,"metric":metric,"x":x,"series":{}}
            for ref,color in COLORS.items():
                y=[float(r[f"{metric}_{ref}"]) for r in subset]
                ax.plot(x,y,color=color,marker="o",ms=3,lw=1.2)
                panel["series"][ref]=y
            ax.set_xscale("log",base=2)
            ax.set_xticks(x,[str(int(v)) for v in x])
            ax.set_xlabel("Free latent parameters" if xkey=="latent_dim" else "Task diversity $M$")
            ax.set_ylabel((source.capitalize()+"\n" if j==0 else "")+("Energy distance" if metric=="ed" else "Sliced Wasserstein-1"), linespacing=2.0)
            ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3),useMathText=True)
            panels.append(panel)
    legend(fig,"aggregate")
    finish(fig,name,[path],panels,destination)


def jellyfish(destination):
    # NoPE only for LR and BAU, matching every other figure in the appendix.
    # All three roll out to N=500. LR and BAU read no position, so the longer
    # rollout uses the trained weights unchanged; Markov is RoPE and reaches
    # 500 only by rescaling its positions into the trained range (rotary
    # position interpolation, scale 256/544). That makes the Markov panel a
    # different model setting from the Markov results elsewhere in the paper,
    # which the caption states: its recovery error is worse than the
    # in-context run's, so this panel shows that the paths settle, not that
    # the estimate is good.
    extended=ROOT/"jellyfish_m32_n500/path_stability_per_prompt.csv"
    sources={"lr":(extended,"none"),"bau":(extended,"none"),"markov":(extended,"markov")}
    data={path:rows(path) for path in (extended,)}
    apply_paper_style(5.5)
    # One standalone figure per family rather than three stacked panels.
    for family in ["lr","bau","markov"]:
        fig,ax=plt.subplots(1,1,figsize=(5.5,2.1))
        fig.subplots_adjust(left=.13,right=.985,bottom=.21,top=.96)
        path,encoding=sources[family]
        subset=[r for r in data[path] if r["family"]==family and r["pos_encoding"]==encoding]
        groups={}
        for r in subset:
            # Prompt key is recorded explicitly in the tidy input.
            key=(r["setup"],r["prompt_idx"])
            groups.setdefault(key,[]).append(r)
        for rs in groups.values():
            rs.sort(key=lambda r:float(r["T"]))
            ax.plot([float(r["T"]) for r in rs],[float(r["scaled_l1"]) for r in rs],color="black",lw=.55)
        ax.set_xlabel("Generated observations $N$")
        ax.set_ylabel("Scaled $L_1$")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.14,lw=.5)
        finish(fig,f"reliability_path_stability_{family}.pdf",[extended],[],destination)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--paper-dir",type=Path,default=REPO.parent/"icl2latex")
    args=parser.parse_args()
    dest=args.paper_dir
    # All three families at N=500. Markov gets there by rotary position
    # interpolation (scale 256/544), the same run as the path-stability
    # figure, so every Markov panel in D.1-D.3 is one model setting. Its
    # posterior sits far from the memorizing reference at every length, which
    # the in-context run does not: that is a property of the rescaled model,
    # and the text says so rather than reading it as a PMC result.
    #
    # The `_pools` runs carry --independent-count-pools: the count curve draws
    # a fresh pool at every count and every repeat instead of subsampling one
    # saved pool of 1024, so D.3's R=1024 endpoint has an error bar rather than
    # a single point. Length and count both come from the same run, so each
    # family shows one prompt across D.2 and D.3; do not mix a length curve
    # from a non-pools run with a count curve from a pools run.
    #
    # Their posterior numbers do NOT match the non-pools runs and are not meant
    # to. In pools mode analyze_source never calls rng.choice, so the numpy
    # stream sits elsewhere when the posterior source draws its prompt, and
    # balls-and-urns picks a different urn while linear regression picks a
    # different task. 1-Markov is unaffected: its prompt comes from the dataset.
    #
    # Linear regression uses the H100 run. The A100 twin (same arguments, same
    # seed) agrees to a relative 3e-4 and is kept as a hardware cross-check;
    # the earlier T4 runs differ by the same amount, which is architecture-
    # specific GEMM summation order compounded along 500 autoregressive steps.
    primary=[("Linear regression", "lr_m32_none_n500_pools_h100"),("Balls-and-urns","bau_m32_none_n500_pools"),("1-Markov","markov_m32_n500_pi544_pools")]
    for source in ["prior","posterior"]:
        for kind in ["length","count"]:
            # No N=223 rule on the two length figures: the cutoff is stated in
            # the text, and the line crowded curves that already carry it.
            curves(primary,source,kind,f"reliability_{kind}_{source}.pdf",dest,mark_context=False)
    for run,name,xkey in [
        ("bau_dimension_sweep_n500","reliability_dimension_bau.pdf","latent_dim"),
        ("markov_dimension_sweep_n500","reliability_dimension_markov.pdf","latent_dim"),
        ("lr_task_diversity_none_n500","reliability_taskdiv_lr.pdf","num_tasks"),
        ("bau_task_diversity_none_n500","reliability_taskdiv_bau.pdf","num_tasks"),
        ("markov_task_diversity_n500","reliability_taskdiv_markov.pdf","num_tasks"),
    ]:aggregate(run,name,dest,xkey)
    # Appendix D.5 repeats the rollout-length analysis at each task diversity,
    # on the same in-context runs as the aggregate figures above.
    # LR and BAU read no position, so their N=500 runs reuse the trained
    # weights and are the ones plotted here. Markov has no N=500 run at these
    # task diversities and stays in-context at N=223.
    for family,ms in [("lr",[32,128,256]),("bau",[32,256,512]),("markov",[16,32,64,128])]:
        # M=32 is the pi544 run; the other task diversities were submitted
        # later under the plain _n500 name. Both use scale 256/544.
        markov_run=lambda m: "markov_m32_n500_pi544" if m==32 else f"markov_m{m}_n500"
        specs=[(f"$M={m}$", markov_run(m) if family=="markov" else f"{family}_m{m}_none_n500") for m in ms]
        curves(specs,"posterior","length",f"reliability_taskdiv_length_{family}.pdf",dest,mark_context=False)
    jellyfish(dest)
    supplement=[]
    for family,ms,horizons in [("lr",[32,128,256],[223,500]),("bau",[32,256,512],[223,500]),("markov",[16,32,64,128],[223])]:
        for horizon in horizons:
            specs=[(f"$M={m}$", f"markov_m{m}_ctx256" if family=="markov" else f"{family}_m{m}_none"+("_n500" if horizon==500 else "")) for m in ms]
            for source in ["prior","posterior"]:
                for kind in ["length","count"]:
                    name=f"reliability_supp_{family}_n{horizon}_{source}_{kind}.pdf"
                    curves(specs,source,kind,name,dest)
                    supplement.append({"file":name,"family":family,"horizon":horizon,"source":source,"kind":kind,"ms":ms})
    for source in ["prior","posterior"]:
        for kind,horizon in [("length",500),("count",223),("count",500)]:
            specs=[(f"{'LR' if family=='lr' else 'BAU'} / {'learned' if enc=='learned' else 'RoPE'}",f"{family}_m32_{enc}"+("_n500" if horizon==500 else "")) for family in ["lr","bau"] for enc in ["learned","rope"]]
            name=f"reliability_supp_encodings_n{horizon}_{source}_{kind}.pdf"
            curves(specs,source,kind,name,dest)
            supplement.append({"file":name,"family":"encodings","horizon":horizon,"source":source,"kind":kind,"ms":[32]})
    manifest={"generator":"scripts/plot_reliability_appendix.py","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"figures":MANIFEST,"supplement":supplement}
    (dest/"reliability_figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Generated {len(MANIFEST)} figures and provenance manifest in {dest}")


if __name__=="__main__":
    main()
