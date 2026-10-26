/* PMC supplementary explorer.

   Data arrives as <script> globals (PMC_INDEX, FIGURES, SERIES) and per-config
   files loaded on demand, so the page works over file:// where fetch() of a
   local file is blocked. */
(function () {
  "use strict";

  const INDEX = window.PMC_INDEX || { configs: [], families: {} };
  const FIG = window.FIGURES || { families: {}, marginals: {}, context: {}, other: {}, slices: {} };
  const SER = window.SERIES || { sweep: {}, dynamics: {}, prior: {} };
  const P = window.Plot;

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };

  const FAMILY_ORDER = ["lr", "bau", "markov"];
  const FAMILY_SHORT = { lr: "Linear regression", bau: "Balls-and-urns", markov: "Markov chains" };
  const COND_LABEL = { prior: "Prior", posterior: "Posterior" };

  // ====================================================================
  // maths — KaTeX, vendored locally so it also works over file://
  // ====================================================================
  /** A readable HTML rendering of the small TeX subset this page uses.
      It is written into every formula slot *before* KaTeX runs, so that if
      KaTeX is missing — blocked, failed to load, or a stale cached script —
      the maths degrades to Unicode instead of vanishing from the page. */
  function texFallback(src) {
    return String(src)
      .replace(/\\mathrm\{([^}]*)\}/g, "$1")
      .replace(/\\Pi/g, "\u03a0")
      .replace(/\\theta/g, "\u03b8")
      .replace(/\\infty/g, "\u221e")
      .replace(/\\ell/g, "ℓ")
      .replace(/\\cdot/g, "·")
      .replace(/\\mid/g, " | ")
      .replace(/\^\{([^}]*)\}/g, "<sup>$1</sup>")
      .replace(/\\to/g, "\u2192")
      .replace(/\\[!,;: ]/g, "")
      .replace(/_\{([^}]*)\}/g, "<sub>$1</sub>")
      .replace(/_(\S)/g, "<sub>$1</sub>")
      .replace(/\s*=\s*/g, " = ");
  }

  /** Markup for an inline formula; rendered by `typeset` once in the DOM. */
  function tex(src) {
    return `<span data-tex="${src.replace(/"/g, "&quot;")}">${texFallback(src)}</span>`;
  }

  /** Render every unrendered [data-tex] under `root`. */
  function typeset(root) {
    (root || document).querySelectorAll("[data-tex]").forEach((n) => {
      if (n.dataset.texDone === "1") return;
      if (!n.innerHTML.trim()) n.innerHTML = texFallback(n.dataset.tex);
      if (typeof katex === "undefined") return; // keep the fallback on screen
      try {
        katex.render(n.dataset.tex, n, { throwOnError: false, displayMode: false });
        n.dataset.texDone = "1";
      } catch (e) {
        n.innerHTML = texFallback(n.dataset.tex);
      }
    });
  }

  // ====================================================================
  // config loading (one <script> per config, cached)
  // ====================================================================
  const loading = {};
  function loadConfig(key) {
    if (window.PMC_DATA && window.PMC_DATA[key]) return Promise.resolve(window.PMC_DATA[key]);
    if (loading[key]) return loading[key];
    const entry = INDEX.configs.find((c) => c.key === key);
    if (!entry) return Promise.reject(new Error("unknown config " + key));
    loading[key] = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      // size stands in for a content version: a rebuilt bundle gets a new URL
      // rather than being served from a stale cache
      s.src = `${entry.file}?v=${entry.version || entry.bytes || "1"}`;
      s.onload = () => {
        const d = window.PMC_DATA && window.PMC_DATA[key];
        d ? resolve(d) : reject(new Error("empty config " + key));
      };
      s.onerror = () => reject(new Error("failed to load " + entry.file));
      document.head.appendChild(s);
    });
    return loading[key];
  }

  /* uint16 -> Float64Array, undoing the per-coordinate quantisation. */
  const decoded = new WeakMap();
  function decode(block, nCoords, lo, hi) {
    let cache = decoded.get(block);
    if (cache) return cache;
    const out = {};
    ["pmc", "mem", "gen"].forEach((k) => {
      if (!block[k]) return;
      const bin = atob(block[k]);
      const u16 = new Uint16Array(bin.length / 2);
      for (let i = 0; i < u16.length; i++) {
        u16[i] = bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8);
      }
      const f = new Float64Array(u16.length);
      for (let i = 0; i < u16.length; i++) {
        const c = i % nCoords;
        f[i] = lo[c] + (u16[i] / 65535) * (hi[c] - lo[c]);
      }
      out[k] = f;
    });
    decoded.set(block, out);
    return out;
  }

  /** Pull one coordinate's samples out of a flat (rows, coords) buffer. */
  function column(flat, nCoords, coord, rows, stride0) {
    const out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) out[r] = flat[r * stride0 + coord];
    return out;
  }

  // ====================================================================
  // explorer state
  // ====================================================================
  const S = {
    family: "lr",
    M: null,
    cond: "posterior",
    lenIdx: null, // set to the longest rollout once a config is loaded
    coord: 0,
    layers: { pmc: true, mem: true, gen: true },
    cfg: null,
    playing: false,
    timer: null,
  };

  const configsFor = (fam) => INDEX.configs.filter((c) => c.family === fam).sort((a, b) => a.M - b.M);
  const configKey = () => `${S.family}_m${S.M}`;

  // ====================================================================
  // sample extraction for the current selection
  // ====================================================================
  const binCache = new Map();

  /** Binned counts for one (coord, layer) at the current rollout length. */
  function layerBins(cfg, cond, coord, lenIdx, bins) {
    const key = `${cfg.family}|${cfg.M}|${cond}|${coord}|${lenIdx}|${bins}`;
    if (binCache.has(key)) return binCache.get(key);

    const block = cfg.conditionings[cond];
    const C = cfg.n_coords, L = cfg.rollout_lengths.length, n = block.n;
    const d = decode(block, C, block.lo, block.hi);
    const lo = block.lo[coord], hi = block.hi[coord];

    // pmc is (rows, lengths, coords); mem and gen are (rows, coords)
    const pmcCol = new Float64Array(n);
    for (let r = 0; r < n; r++) pmcCol[r] = d.pmc[(r * L + lenIdx) * C + coord];

    const out = {
      lo, hi,
      pmc: P.bin(pmcCol, lo, hi, bins),
      mem: P.bin(column(d.mem, C, coord, n, C), lo, hi, bins),
      gen: P.bin(column(d.gen, C, coord, n, C), lo, hi, bins),
      meanPmc: mean(pmcCol),
      sdPmc: sd(pmcCol),
    };
    binCache.set(key, out);
    return out;
  }

  function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function sd(a) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2; return Math.sqrt(s / a.length); }

  function activeLayers(b) {
    return ["pmc", "mem", "gen"]
      .filter((k) => S.layers[k])
      .map((k) => ({ key: k, counts: b[k].counts, n: b[k].n }));
  }

  /** Every layer, toggled on or not. */
  function allLayers(b) {
    return ["pmc", "mem", "gen"].map((k) => ({ key: k, counts: b[k].counts, n: b[k].n }));
  }

  const yScaleCache = new Map();

  /**
   * The y maximum for one coordinate, fixed for the whole panel.
   *
   * Taken over the PMC cloud at *every* rollout length plus the generalising
   * reference, so neither the layer toggles nor the rollout slider can move
   * the axis — dragging the slider then shows PMC sharpening, rather than a
   * curve of constant height against a shifting scale.
   *
   * Pi_M is excluded for the usual reason: at small M it is nearly a point
   * mass whose height is an artefact of the bin width.
   */
  function yScaleFor(cfg, cond, coord, bins) {
    const key = `${cfg.family}|${cfg.M}|${cond}|${coord}|${bins}`;
    if (yScaleCache.has(key)) return yScaleCache.get(key);

    const block = cfg.conditionings[cond];
    const C = cfg.n_coords, L = cfg.rollout_lengths.length, n = block.n;
    const d = decode(block, C, block.lo, block.hi);
    const lo = block.lo[coord], hi = block.hi[coord];

    let max = 0;
    const consider = (binned) => {
      const vals = P.smooth(P.density(binned.counts, binned.n, lo, hi), 1);
      for (let i = 0; i < vals.length; i++) if (vals[i] > max) max = vals[i];
    };

    const col = new Float64Array(n);
    for (let li = 0; li < L; li++) {
      for (let r = 0; r < n; r++) col[r] = d.pmc[(r * L + li) * C + coord];
      consider(P.bin(col, lo, hi, bins));
    }
    consider(P.bin(column(d.gen, C, coord, n, C), lo, hi, bins));

    const ref = block.reference ? block.reference[coord] : null;
    if (ref) {
      const probe = [];
      for (let i = 0; i < 160; i++) {
        const v = P.analyticAt(ref, lo + ((hi - lo) * i) / 159, "density");
        if (Number.isFinite(v)) probe.push(v);
      }
      probe.sort((a, b) => a - b);
      if (probe.length) max = Math.max(max, probe[Math.floor(probe.length * 0.99)]);
    }

    const out = max > 0 ? max * 1.14 : 1;
    yScaleCache.set(key, out);
    return out;
  }

  /** Wasserstein-1 between two binned distributions, in units of the axis. */
  function w1(a, bb, lo, hi) {
    const n = a.counts.length, w = (hi - lo) / n;
    const ca = P.cdf(a.counts, a.n), cb = P.cdf(bb.counts, bb.n);
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.abs(ca[i] - cb[i]) * w;
    return s;
  }

  // ====================================================================
  // controls
  // ====================================================================
  function segment(host, items, current, onPick) {
    host.innerHTML = "";
    items.forEach((it) => {
      const b = el("button", null, it.label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(it.key === current));
      if (it.disabled) b.disabled = true;
      b.addEventListener("click", () => onPick(it.key));
      host.appendChild(b);
    });
  }

  function press(host, attr, value) {
    host.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset[attr] === value)));
  }

  function referenceTex(layer, cond = S.cond) {
    if (cond === "prior") return layer === "mem" ? "\\Pi_M" : "\\Pi_\\infty";
    return `\\Pi^{\\mathrm{${layer}}}(\\cdot\\mid c)`;
  }

  function renderConditioning() {
    $("c-cond").querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.textContent === COND_LABEL[S.cond])));
    ["mem", "gen"].forEach((layer) => {
      const node = $(`${layer}-label`);
      node.innerHTML = tex(referenceTex(layer));
      node.parentElement.setAttribute("aria-label", `${layer === "mem" ? "Memorizing" : "Generalizing"} reference`);
      typeset(node);
    });
  }

  function renderControls() {
    segment($("c-family"),
      FAMILY_ORDER.filter((f) => configsFor(f).length).map((f) => ({ key: f, label: FAMILY_SHORT[f] })),
      S.family,
      (f) => { S.family = f; S.M = configsFor(f)[0].M; S.coord = 0; select(); });

    segment($("c-m"),
      configsFor(S.family).map((c) => ({ key: String(c.M), label: String(c.M) })),
      String(S.M),
      (m) => { S.M = Number(m); select(); });

    const conds = S.cfg ? Object.keys(S.cfg.conditionings) : ["prior", "posterior"];
    segment($("c-cond"),
      ["prior", "posterior"].filter((k) => conds.includes(k)).map((k) => ({ key: k, label: COND_LABEL[k] })),
      S.cond,
      (k) => { S.cond = k; draw(); });

    const slider = $("c-len");
    const lens = S.cfg ? S.cfg.rollout_lengths : [0];
    const idx = S.lenIdx == null ? lens.length - 1 : Math.min(S.lenIdx, lens.length - 1);
    slider.max = String(lens.length - 1);
    slider.value = String(idx);
    $("c-len-val").textContent = String(lens[idx]);
  }

  $("c-len").addEventListener("input", (e) => {
    S.lenIdx = Number(e.target.value);
    if (S.cfg) $("c-len-val").textContent = String(S.cfg.rollout_lengths[S.lenIdx]);
    stopPlay();
    draw();
  });

  $("c-layers").querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      const k = b.dataset.layer;
      // keep at least one layer on
      const on = Object.values(S.layers).filter(Boolean).length;
      if (S.layers[k] && on === 1) return;
      S.layers[k] = !S.layers[k];
      b.setAttribute("aria-pressed", String(S.layers[k]));
      draw();
    });
  });

  // ---- rollout-length animation
  function stopPlay() {
    S.playing = false;
    clearInterval(S.timer);
    $("c-play").setAttribute("aria-pressed", "false");
    $("c-play").textContent = "▶";
  }
  function startPlay() {
    if (!S.cfg) return;
    S.playing = true;
    $("c-play").setAttribute("aria-pressed", "true");
    $("c-play").textContent = "❚❚";
    S.timer = setInterval(() => {
      S.lenIdx = (S.lenIdx + 1) % S.cfg.rollout_lengths.length;
      $("c-len").value = String(S.lenIdx);
      $("c-len-val").textContent = String(S.cfg.rollout_lengths[S.lenIdx]);
      draw();
    }, 620);
  }
  $("c-play").addEventListener("click", () => (S.playing ? stopPlay() : startPlay()));

  // ====================================================================
  // drawing the explorer
  // ====================================================================
  const CELL_BINS = 44, FOCUS_BINS = 56;

  function drawCells() {
    const host = $("cells");
    const cfg = S.cfg;
    host.querySelectorAll(".cell").forEach((cell, i) => {
      const canvas = cell.querySelector("canvas");
      const { ctx, w, h } = P.fit(canvas);
      const b = layerBins(cfg, S.cond, i, S.lenIdx, CELL_BINS);
      P.marginal(ctx, { x: 0, y: 0, w, h }, {
        lo: b.lo, hi: b.hi, mode: "density", compact: true,
        layers: activeLayers(b), yMax: yScaleFor(cfg, S.cond, i, CELL_BINS), bars: true,
      });
      const closer = w1(b.pmc, b.mem, b.lo, b.hi) <= w1(b.pmc, b.gen, b.lo, b.hi) ? "mem" : "gen";
      const tag = cell.querySelector(".tag");
      // Re-typesetting on every animation frame is wasteful, so only redraw the
      // tag when the nearer reference actually changes.
      if (tag && tag.dataset.cur !== `${S.cond}|${closer}`) {
        tag.dataset.cur = `${S.cond}|${closer}`;
        tag.textContent = closer === "mem" ? "Mem. closer" : "Gen. closer";
        tag.title = `Closer reference for this coordinate: ${closer === "mem" ? "memorizing" : "generalizing"} (binned Wasserstein-1)`;
        typeset(tag);
      }
    });
  }

  function buildCells() {
    const host = $("cells");
    host.innerHTML = "";
    const cfg = S.cfg;
    cfg.coords.forEach((name, i) => {
      const cell = el("button", "cell");
      cell.type = "button";
      cell.setAttribute("aria-pressed", String(i === S.coord));
      const label = cfg.coords_tex ? tex(cfg.coords_tex[i]) : name;
      cell.innerHTML = `<canvas></canvas><figcaption><span>${label}</span><span class="tag"></span></figcaption>`;
      cell.addEventListener("click", () => {
        S.coord = i;
        host.querySelectorAll(".cell").forEach((c, j) =>
          c.setAttribute("aria-pressed", String(j === i)));
        drawFocus();
      });
      host.appendChild(cell);
    });
    typeset(host);
    $("multiples-label").innerHTML = cfg.family === "markov"
      ? `16 selected entries of ${tex("Q")} · upper-left 4 × 4 block of the 10 × 10 matrix`
      : cfg.family === "lr" ? `All 8 coefficients of ${tex("w")}` : `All 12 color probabilities in ${tex("p")}`;
    typeset($("multiples-label"));
  }

  function drawFocus() {
    const cfg = S.cfg;
    const canvas = $("focus-canvas");
    const stacked = canvas.clientWidth < 640;
    canvas.style.height = stacked ? "600px" : "";
    const { ctx, w, h } = P.fit(canvas);
    const b = layerBins(cfg, S.cond, S.coord, S.lenIdx, FOCUS_BINS);
    const block = cfg.conditionings[S.cond];
    const ref = block.reference ? block.reference[S.coord] : null;

    // density left, CDF right — the user asked for both, and the CDF is where
    // a memorising staircase separates from a generalising curve
    const gap = 26;
    const panelWidth = stacked ? w : (w - gap) / 2;
    const panelHeight = stacked ? (h - gap) / 2 : h;
    const common = {
      lo: b.lo, hi: b.hi, layers: activeLayers(b),
      yMax: yScaleFor(cfg, S.cond, S.coord, FOCUS_BINS),
      analytic: ref, showAnalytic: S.layers.gen && !!ref,
      xLabel: cfg.coords_tex
        ? cfg.coords_tex[S.coord].replace(/_\{([\d,]+)\}/g,
          (_, indices) => indices.replace(/\d/g, (digit) => "₀₁₂₃₄₅₆₇₈₉"[Number(digit)]))
        : cfg.coords[S.coord],
    };
    P.marginal(ctx, { x: 0, y: 0, w: panelWidth, h: panelHeight }, { ...common, mode: "density", bars: true, title: "Density" });
    P.marginal(ctx, { x: stacked ? 0 : panelWidth + gap, y: stacked ? panelHeight + gap : 0, w: panelWidth, h: panelHeight }, { ...common, mode: "cdf", title: "CDF" });

    const coordTex = cfg.coords_tex ? tex(cfg.coords_tex[S.coord]) : cfg.coords[S.coord];
    $("focus-title").innerHTML =
      `${FIG.families[cfg.family] ? FIG.families[cfg.family].label : cfg.family} ` +
      `· ${tex(`M=${cfg.M}`)} · ${coordTex}`;
    const length = cfg.rollout_lengths[S.lenIdx];
    const conditioning = S.cond === "posterior"
      ? `Posterior · In-distribution prompt · ${tex(`\\ell=${block.prompt_length}`)}`
      : "Prior · No observed prompt";
    $("focus-sub").innerHTML = `${conditioning} · ${tex(`N=${length}`)} · ${b.pmc.n} displayed samples`;
    typeset($("focus-sub"));
    const encoding = cfg.position_encoding === "RoPE"
      ? `RoPE position scale: ${Number(cfg.position_scale).toFixed(3)} (256/544).`
      : "No positional encoding.";
    const unit = cfg.family === "lr" ? "pairs" : "observations";
    $("context-note").textContent = `${encoding} ${length > block.context_limit ? "Beyond" : "Within"} training context. Limit: ${block.context_limit} generated ${unit} for this prompt.`;
    typeset($("focus-title"));

    // ---- numeric panel
    const dMem = w1(b.pmc, b.mem, b.lo, b.hi);
    const dGen = w1(b.pmc, b.gen, b.lo, b.hi);
    const rows = $("stat-rows");
    rows.innerHTML = "";
    const add = (color, k, v) => {
      const r = el("div", "stat-fig");
      r.innerHTML = `<span class="v">${v}</span>` +
        `<span class="k"><i style="background:${color}"></i>${k}</span>`;
      rows.appendChild(r);
    };
    add(P.COLORS.pmc, "PMC mean", P.fmt(b.meanPmc));
    add(P.COLORS.pmc, "PMC s.d.", P.fmt(b.sdPmc));
    add(P.COLORS.mem, "Wasserstein-1 · memorizing", P.fmt(dMem));
    add(P.COLORS.gen, "Wasserstein-1 · generalizing", P.fmt(dGen));

    const memCloser = dMem <= dGen;
    $("stat-note").innerHTML =
      `Closer reference for this coordinate: ` +
      `<strong>` +
      `${memCloser ? "memorizing" : "generalizing"}</strong>. Distances approximate one-dimensional ` +
      `Wasserstein-1 using binned sample CDFs. This marginal comparison does not determine agreement ` +
      `between the full joint distributions. <a href="${cfg.source_url}">Run settings</a>.`;
    typeset(rows);
  }

  function draw() {
    if (!S.cfg) return;
    renderConditioning();
    drawCells();
    drawFocus();
  }

  function select() {
    const host = $("cells");
    host.innerHTML = '<p class="loading">Loading the recovered samples…</p>';
    renderControls();
    loadConfig(configKey()).then((cfg) => {
      S.cfg = cfg;
      if (!cfg.conditionings[S.cond]) S.cond = Object.keys(cfg.conditionings)[0];
      // Open on the longest recorded rollout; this is a diagnostic, not a convergence claim.
      if (S.lenIdx == null) S.lenIdx = cfg.rollout_lengths.length - 1;
      S.lenIdx = Math.min(S.lenIdx, cfg.rollout_lengths.length - 1);
      S.coord = Math.min(S.coord, cfg.n_coords - 1);
      renderControls();
      buildCells();
      draw();
    }).catch((err) => {
      host.innerHTML = `<p class="loading">Could not load this configuration (${err.message}).</p>`;
    });
  }

  // ====================================================================
  // hero teaser: the same coordinate at three diversities
  // ====================================================================
  function drawTeaser() {
    const cells = $("teaser-cells").querySelectorAll(".teaser-cell");
    const cfgs = configsFor("lr").slice(0, 3);
    if (!cfgs.length) return;
    $("teaser-coord").innerHTML = `Linear regression · ${tex("w_1")}`;
    typeset($("teaser-coord"));
    Promise.all(cfgs.map((c) => loadConfig(c.key))).then((loaded) => {
      loaded.forEach((cfg, i) => {
        const cell = cells[i];
        if (!cell) return;
        const canvas = cell.querySelector("canvas");
        const { ctx, w, h } = P.fit(canvas);
        const cond = cfg.conditionings.posterior ? "posterior" : Object.keys(cfg.conditionings)[0];
        const li = cfg.rollout_lengths.length - 1;
        const b = layerBins(cfg, cond, 0, li, CELL_BINS);
        P.marginal(ctx, { x: 0, y: 0, w, h }, {
          lo: b.lo, hi: b.hi, mode: "density", compact: true, bars: true,
          yMax: yScaleFor(cfg, cond, 0, CELL_BINS),
          layers: ["pmc", "mem", "gen"].map((k) => ({ key: k, counts: b[k].counts, n: b[k].n })),
        });
        const dMem = w1(b.pmc, b.mem, b.lo, b.hi), dGen = w1(b.pmc, b.gen, b.lo, b.hi);
        const closer = dMem <= dGen ? "mem" : "gen";
        const caps = cell.querySelectorAll(".teaser-cap span");
        caps[0].innerHTML = tex(`M=${cfg.M}`);
        caps[1].textContent = closer === "mem" ? "Mem. closer" : "Gen. closer";
        caps[1].title = "Closer reference for this coordinate (binned Wasserstein-1)";
        typeset(caps[0]);
      });
    }).catch(() => {});
  }

  // ====================================================================
  // section 02 — sweep curves
  // ====================================================================
  const SW = { kind: "sweep", family: "lr", cond: null, metric: "ED", scale: "log", hidden: new Set(), frame: null, series: [] };

  function styleFor(label) {
    const mem = /memori[sz]/i.test(label);
    const gen = /generali[sz]/i.test(label);
    return { color: mem ? P.COLORS.mem : gen ? P.COLORS.gen : "#111110", dash: /^SW/.test(label) ? [5, 4] : [] };
  }

  const sweepData = () => {
    const data = SW.kind === "sweep"
      ? (SER.sweep[SW.family] || {})[SW.cond] || {}
      : (SER[SW.kind] || {})[SW.family] || {};
    return Object.fromEntries(Object.entries(data).filter(([label]) =>
      SW.kind === "prior" || label.startsWith(SW.metric)));
  };

  const usesPaperDynamics = () => SW.family === "markov" && SW.kind === "dynamics";

  function sweepMeta() {
    const fam = (FIG.families[SW.family] || {}).label || FAMILY_SHORT[SW.family];
    if (SW.kind === "sweep") return {
      title: `${fam} — distance vs task diversity`,
      x: "Task diversity M (log)", y: SW.metric === "ED" ? "Squared energy distance" : "Sliced Wasserstein-1",
      note: `Smaller distances indicate closer agreement with a reference. A crossover depends on the metric and evaluation setting. Markers identify values of task diversity ${tex("M")} available in the explorer.`,
    };
    if (SW.kind === "dynamics") return {
      title: `${fam} — distance vs training step`,
      x: "Training step (log)", y: SW.metric === "ED" ? "Squared energy distance" : "Sliced Wasserstein-1",
      note: SW.family === "markov"
        ? `The paper’s ${tex("M=8")} figure compares predictions (symmetrized KL) and posteriors (sliced Wasserstein-1). Rows show in-distribution and out-of-distribution prompts. Select the figure to enlarge it.`
        : `A single run at ${tex("M=32")}, averaged over in-distribution prompts. Transient generalization describes an initial approach to the generalizing reference followed by a later shift toward the memorizing reference.`,
    };
    return {
      title: `${fam} — prior predictive comparison`,
      x: "Task diversity M (log)", y: SW.family === "lr" ? "Δ MSE vs reference" : "KL vs reference",
      note: "Predictions made without an observed prompt. Agreement in prediction space does not uniquely identify the latent prior: different priors can produce identical mean predictions or next-token predictive distributions.",
    };
  }

  function renderSweepControls() {
    segment($("s-family"), FAMILY_ORDER.map((family) => ({ key: family, label: FAMILY_SHORT[family] })),
      SW.family, (family) => { SW.family = family; SW.cond = null; SW.hidden.clear(); renderSweepControls(); drawSweep(); });
    $("s-metric-wrap").hidden = SW.kind === "prior" || usesPaperDynamics();
    segment($("s-metric"), [
      { key: "ED", label: "Squared energy distance" },
      { key: "SW", label: "Sliced Wasserstein-1" },
    ], SW.metric, (metric) => { SW.metric = metric; SW.hidden.clear(); renderSweepControls(); drawSweep(); });
    $("s-series").parentElement.hidden = usesPaperDynamics();
    $("s-scale").parentElement.hidden = usesPaperDynamics();
    const wrap = $("s-cond-wrap");
    if (SW.kind !== "sweep") { wrap.hidden = true; }
    else {
      wrap.hidden = false;
      const avail = Object.keys(SER.sweep[SW.family] || {});
      const srcs = (FIG.families[SW.family] || { sources: [] }).sources || [];
      const items = srcs.filter((s) => avail.includes(s.key)).map((s) => ({ key: s.key, label: s.label }));
      if (!items.some((i) => i.key === SW.cond)) {
        const post = items.find((i) => (srcs.find((s) => s.key === i.key) || {}).kind === "posterior");
        SW.cond = (post || items[0] || {}).key || null;
      }
      segment($("s-cond"), items, SW.cond, (k) => { SW.cond = k; renderSeriesList(); drawSweep(); });
    }
    renderSeriesList();
  }

  function renderSeriesList() {
    const host = $("s-series");
    host.innerHTML = "";
    const labels = Object.keys(sweepData());
    if (!labels.length) { host.appendChild(el("p", "note", "No series for this selection.")); return; }
    labels.forEach((label) => {
      const st = styleFor(label);
      // SW series are drawn dashed, so the swatch is dashed too — otherwise the
      // legend claims two curves look alike when they do not.
      const swatch = st.dash.length
        ? `<i class="dashed" style="border-top-color:${st.color}"></i>`
        : `<i style="background:${st.color}"></i>`;
      const b = el("button", null, `${swatch}${label}`);
      b.type = "button";
      b.setAttribute("aria-pressed", String(!SW.hidden.has(label)));
      b.addEventListener("click", () => {
        SW.hidden.has(label) ? SW.hidden.delete(label) : SW.hidden.add(label);
        renderSeriesList(); drawSweep();
      });
      host.appendChild(b);
    });
  }

  function drawSweep() {
    const meta = sweepMeta();
    $("sweep-title").textContent = meta.title;
    const srcs = (FIG.families[SW.family] || { sources: [] }).sources || [];
    $("sweep-sub").textContent = SW.kind === "sweep" && SW.cond
      ? (srcs.find((s) => s.key === SW.cond) || {}).label || "" : "";
    $("sweep-note").innerHTML = meta.note;
    typeset($("sweep-note"));

    const paperView = usesPaperDynamics();
    $("sweep-canvas").parentElement.hidden = paperView;
    $("sweep-paper-figure").hidden = !paperView;
    $("sweep-paper-figure").innerHTML = "";
    if (paperView) {
      SW.series = []; SW.frame = null;
      $("sweep-readout").classList.remove("on");
      const meta = (FIG.other.markov || []).find((figure) => figure.slug === "dynamics");
      if (meta) {
        const img = el("img");
        img.src = meta.src; img.alt = "Markov chains: task diversity M = 8, training dynamics from the paper";
        img.addEventListener("click", () => openLb(meta.src, img.alt));
        $("sweep-paper-figure").appendChild(img);
      }
      return;
    }

    SW.series = Object.entries(sweepData())
      .filter(([l]) => !SW.hidden.has(l))
      .map(([label, points]) => ({ label, points, ...styleFor(label) }));

    if (SW.scale === "log" && SW.series.some((series) => series.points.some((p) => p[1] <= 0))) {
      $("sweep-note").appendChild(document.createTextNode(" Non-positive estimates are omitted on the log scale. Select Linear to display them."));
    }

    SW.frame = P.lines($("sweep-canvas"), {
      series: SW.series,
      logX: true, logY: SW.scale === "log",
      xLabel: meta.x, yLabel: meta.y,
      xFormat: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)),
      marks: SW.kind === "dynamics" ? null : configsFor(SW.family).map((c) => c.M),
    });
  }

  $("s-kind").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    SW.kind = b.dataset.kind; SW.hidden.clear();
    press($("s-kind"), "kind", SW.kind);
    renderSweepControls(); drawSweep();
  }));
  $("s-scale").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    SW.scale = b.dataset.scale; press($("s-scale"), "scale", SW.scale); drawSweep();
  }));

  (function wireSweepHover() {
    const canvas = $("sweep-canvas"), out = $("sweep-readout");
    canvas.addEventListener("mousemove", (e) => {
      if (!SW.frame || !SW.series.length) return;
      const r = canvas.getBoundingClientRect();
      const target = SW.frame.invX(e.clientX - r.left);
      const lines = SW.series.map((s) => {
        let best = null;
        s.points.forEach((p) => {
          const d = Math.abs(Math.log10(Math.max(p[0], 1e-9)) - Math.log10(Math.max(target, 1e-9)));
          if (!best || d < best.d) best = { d, p };
        });
        return best ? `${s.label}: ${P.fmt(best.p[1])}` : null;
      }).filter(Boolean);
      if (!lines.length) return;
      const near = SW.series[0].points.reduce((a, p) =>
        Math.abs(Math.log10(p[0]) - Math.log10(target)) < Math.abs(Math.log10(a) - Math.log10(target)) ? p[0] : a,
        SW.series[0].points[0][0]);
      const position = SW.kind === "dynamics" ? `Step ${near}` : tex(`M=${near}`);
      out.innerHTML = `<strong>${position}</strong><br>${lines.join("<br>")}`;
      typeset(out);
      out.classList.add("on");
    });
    canvas.addEventListener("mouseleave", () => out.classList.remove("on"));
  })();

  // ====================================================================
  // section 03 — figure archive
  // ====================================================================
  const AR = { family: "lr", source: null, mIdx: 6, extra: 0 };

  const lb = $("lightbox"), lbImg = $("lightbox-img"), lbCap = $("lightbox-cap");
  function openLb(src, cap) {
    lbImg.src = src; lbImg.alt = cap || ""; lbCap.textContent = cap || "";
    lb.classList.add("on"); document.body.style.overflow = "hidden";
  }
  function closeLb() {
    lb.classList.remove("on"); lbImg.removeAttribute("src"); document.body.style.overflow = "";
  }
  lb.addEventListener("click", closeLb);
  $("lightbox-close").addEventListener("click", closeLb);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && lb.classList.contains("on")) closeLb(); });

  const archFamilies = () => Object.keys(FIG.families)
    .filter((f) => Object.keys(FIG.marginals).some((k) => k.startsWith(f + "|")));

  function archSpec() { return FIG.families[AR.family]; }

  function renderArch() {
    const spec = archSpec();
    if (!spec) return;
    segment($("a-family"), archFamilies().map((f) => ({ key: f, label: FAMILY_SHORT[f] || FIG.families[f].label })),
      AR.family, (f) => {
        AR.family = f;
        const s = FIG.families[f].sources;
        AR.source = (s.find((x) => x.kind === "posterior") || s[0]).key;
        AR.mIdx = Math.min(AR.mIdx, FIG.families[f].diversity.length - 1);
        renderArch();
      });

    if (!spec.sources.some((s) => s.key === AR.source)) {
      AR.source = (spec.sources.find((s) => s.kind === "posterior") || spec.sources[0]).key;
    }
    segment($("a-source"), spec.sources.map((s) => ({ key: s.key, label: s.label.replace("Posterior · ", "") })),
      AR.source, (k) => { AR.source = k; renderArch(); });

    const d = spec.diversity;
    const slider = $("a-m");
    slider.max = String(d.length - 1);
    AR.mIdx = Math.min(AR.mIdx, d.length - 1);
    slider.value = String(AR.mIdx);
    $("a-m-val").innerHTML = tex(`M=${d[AR.mIdx]}`);
    typeset($("a-m-val"));
    drawArch();
  }

  function drawArch() {
    const spec = archSpec();
    const m = spec.diversity[AR.mIdx];
    const cell = FIG.marginals[`${AR.family}|${m}|${AR.source}`];
    const host = $("a-figure");
    host.innerHTML = "";
    if (!cell) { host.innerHTML = '<p class="loading">No figure for this combination.</p>'; }
    else {
      const metas = cell.grid ? [cell.grid] : [cell.density, cell.cdf].filter(Boolean);
      metas.forEach((meta) => {
        const img = el("img");
        img.src = meta.src;
        const source = spec.sources.find((item) => item.key === AR.source);
        const view = cell.grid ? "Marginal densities and CDFs" : meta === cell.density ? "Marginal densities" : "Marginal CDFs";
        img.alt = `${spec.label} · M = ${m} · ${source.label} · ${view}`;
        img.loading = "lazy";
        img.addEventListener("click", () => openLb(meta.src, img.alt));
        fitFigure(img, meta);
        host.appendChild(img);
        const caption = el("p", "figure-source");
        caption.innerHTML = `${spec.label} · ${tex(`M=${m}`)} · ${source.label} · ${view}. `;
        if (meta.source_url) {
          const link = el("a", null, "Source figure"); link.href = meta.source_url; caption.appendChild(link);
        }
        host.appendChild(caption);
        typeset(caption);
      });
      host.appendChild(el("p", "arch-hint", "Click to enlarge"));
    }

    drawExtras();
  }

  /** Size one archive figure: dense grids get the column, summaries get capped. */
  function fitFigure(img, meta) {
    if (meta && meta.dense) img.classList.add("dense");
    const setNat = () => {
      if (img.naturalWidth) img.style.setProperty("--nat", img.naturalWidth + "px");
    };
    if (img.complete) setNat();
    else img.addEventListener("load", setNat, { once: true });
  }

  /** The cross-M and prediction-space figures, one at a time, at full width. */
  function archExtras() {
    return (FIG.context[`${AR.family}|${AR.source}`] || [])
      .concat(FIG.other[AR.family] || []);
  }

  function drawExtras() {
    const extras = archExtras();
    const host = $("a-extra-figure");
    const wrap = document.querySelector(".arch-extra");
    if (!extras.length) {
      wrap.hidden = true;
      return;
    }
    wrap.hidden = false;
    AR.extra = Math.min(AR.extra, extras.length - 1);
    segment(
      $("a-extra"),
      extras.map((e, i) => ({ key: String(i), label: e.label })),
      String(AR.extra),
      (k) => { AR.extra = Number(k); drawExtras(); }
    );
    const meta = extras[AR.extra];
    host.innerHTML = "";
    const img = el("img");
    img.src = meta.src;
    img.alt = meta.label;
    img.loading = "lazy";
    img.addEventListener("click", () => openLb(meta.src, meta.label));
    fitFigure(img, meta);
    host.appendChild(img);
    host.appendChild(el("p", "arch-hint", "Click to enlarge"));
    if (meta.source_url) {
      const source = el("p", "figure-source");
      const link = el("a", null, "Source figure"); link.href = meta.source_url;
      source.appendChild(link); host.appendChild(source);
    }
  }

  $("a-m").addEventListener("input", (e) => {
    AR.mIdx = Number(e.target.value);
    $("a-m-val").innerHTML = tex(`M=${archSpec().diversity[AR.mIdx]}`);
    typeset($("a-m-val"));
    drawArch();
  });

  // ====================================================================
  // boot
  // ====================================================================
  function boot() {
    typeset(document); // the static formulas in the page shell

    if (!INDEX.configs.length) {
      $("cells").innerHTML = '<p class="loading">No PMC bundles were exported. Run build_pmc.py.</p>';
    } else {
      S.family = "lr";
      S.M = configsFor("lr")[0].M;
      select();
      drawTeaser();
    }

    SW.family = S.family;
    renderSweepControls();
    drawSweep();

    if (archFamilies().length) {
      AR.family = archFamilies()[0];
      const s = FIG.families[AR.family].sources;
      AR.source = (s.find((x) => x.kind === "posterior") || s[0]).key;
      renderArch();
    }

    let t;
    window.addEventListener("resize", () => {
      clearTimeout(t);
      t = setTimeout(() => { draw(); drawSweep(); drawTeaser(); }, 150);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
