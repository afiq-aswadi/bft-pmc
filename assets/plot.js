/* Canvas drawing primitives for the PMC explorer.

   Everything a panel shows comes from one pass of binning: counts over a fixed
   range give the density (counts / n / width) and the CDF (cumulative counts /
   n) at once. That keeps a 16-panel redraw cheap enough to animate, and means
   the density and the CDF of a layer can never disagree. */
window.Plot = (function () {
  "use strict";

  const COLORS = { pmc: "#c8901a", mem: "#2f9e41", gen: "#1f77b4" };
  const LABELS = { pmc: "PMC", mem: "Πₘ", gen: "Π∞" };
  const INK = "#222222", INK2 = INK, INK3 = INK, GRID = "#edebe7", AXIS = "#cbc7c0";
  const SANS = "Arial, Helvetica, sans-serif";

  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  /** Bin `values` (a Float64Array view) into `bins` over [lo, hi]. */
  function bin(values, lo, hi, bins) {
    const counts = new Float64Array(bins);
    const span = hi - lo;
    if (!(span > 0)) return { counts, n: 0 };
    const scale = bins / span;
    let n = 0;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v < lo || v > hi) continue;
      let k = ((v - lo) * scale) | 0;
      if (k >= bins) k = bins - 1;
      counts[k] += 1;
      n++;
    }
    return { counts, n };
  }

  /** Three-tap smoothing, so a 48-bin histogram reads as a curve. */
  function smooth(a, passes) {
    let cur = a;
    for (let p = 0; p < (passes || 1); p++) {
      const out = new Float64Array(cur.length);
      for (let i = 0; i < cur.length; i++) {
        const l = cur[i - 1] !== undefined ? cur[i - 1] : cur[i];
        const r = cur[i + 1] !== undefined ? cur[i + 1] : cur[i];
        out[i] = 0.25 * l + 0.5 * cur[i] + 0.25 * r;
      }
      cur = out;
    }
    return cur;
  }

  function density(counts, n, lo, hi) {
    const bins = counts.length, w = (hi - lo) / bins;
    const out = new Float64Array(bins);
    if (!n || !(w > 0)) return out;
    for (let i = 0; i < bins; i++) out[i] = counts[i] / (n * w);
    return out;
  }

  function cdf(counts, n) {
    const out = new Float64Array(counts.length);
    let acc = 0;
    for (let i = 0; i < counts.length; i++) { acc += counts[i]; out[i] = n ? acc / n : 0; }
    return out;
  }

  // ---------------------------------------------------------------- maths
  const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];

  function lgamma(z) {
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1;
    let x = 0.99999999999980993;
    for (let i = 0; i < LANCZOS.length; i++) x += LANCZOS[i] / (z + i + 1);
    const t = z + LANCZOS.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  const lnB = (a, b) => lgamma(a) + lgamma(b) - lgamma(a + b);

  function betaPdf(x, a, b) {
    if (x <= 0 || x >= 1) return 0;
    return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lnB(a, b));
  }
  function normPdf(x, mu, sd) {
    const z = (x - mu) / sd;
    return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
  }
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const normCdf = (x, mu, sd) => 0.5 * (1 + erf((x - mu) / (sd * Math.SQRT2)));

  function betacf(x, a, b) {
    const EPS = 3e-14, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 250; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function betaCdf(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const f = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lnB(a, b));
    return x < (a + 1) / (a + b + 2)
      ? (f * betacf(x, a, b)) / a
      : 1 - (Math.exp(b * Math.log(1 - x) + a * Math.log(x) - lnB(a, b)) * betacf(1 - x, b, a)) / b;
  }

  function analyticAt(ref, x, mode) {
    if (!ref) return null;
    if (ref.kind === "normal") {
      return mode === "cdf" ? normCdf(x, ref.mu, ref.sigma) : normPdf(x, ref.mu, ref.sigma);
    }
    return mode === "cdf" ? betaCdf(x, ref.a, ref.b) : betaPdf(x, ref.a, ref.b);
  }

  // ------------------------------------------------------------ formatting
  function fmt(v, digits) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
    if (digits != null) return v.toFixed(digits);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }

  function ticks(lo, hi, count) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return out;
  }

  // ------------------------------------------------------- the marginal cell
  /**
   * One panel: density or CDF, with up to three layers over a shared x-range.
   * `layers` is [{key, counts, n}]; bars draws the PMC histogram behind.
   */
  function marginal(ctx, rect, o) {
    const compact = !!o.compact;
    // Compact panels carry no axes, but each coordinate has its own range, so
    // they still print the two endpoints — otherwise the shape is readable and
    // the position is not.
    const pad = compact
      ? { l: 7, r: 7, t: 7, b: 22 }
      : { l: 54, r: 22, t: 14, b: 42 };
    const x0 = rect.x + pad.l, y0 = rect.y + pad.t;
    const iw = rect.w - pad.l - pad.r, ih = rect.h - pad.t - pad.b;
    if (iw <= 4 || ih <= 4) return;

    const { lo, hi, mode } = o;
    const bins = o.layers.length ? o.layers[0].counts.length : 1;

    // y-scale: CDF is always [0,1]; density takes the tallest smoothed curve,
    // with the analytic peak clipped at its 99th percentile so a Beta that
    // spikes at an endpoint cannot flatten everything else.
    let yMax = 1;
    const series = o.layers.map((L) => {
      const vals = mode === "cdf"
        ? cdf(L.counts, L.n)
        : smooth(density(L.counts, L.n, lo, hi), 1);
      return { key: L.key, vals, n: L.n };
    });
    if (mode !== "cdf" && o.yMax > 0) {
      // A scale computed once for this coordinate, over every layer and every
      // rollout length. Nothing the controls do may move the axis.
      yMax = o.yMax;
    } else if (mode !== "cdf") {
      const scaleSource = (o.scaleLayers || o.layers).map((L) => ({
        key: L.key,
        n: L.n,
        vals: smooth(density(L.counts, L.n, lo, hi), 1),
      }));
      // Pi_M collapses toward a point mass at low diversity, so its density
      // spike is position-carrying, not height-carrying: including it would
      // flatten every other curve to the axis. Scale to PMC and the population
      // instead -- the same rule as plotting/marginal_cell.py -- and let the
      // memorising spike clip.
      const scaleBy = scaleSource.filter((s) => s.key !== "mem" && s.n);
      const considered = scaleBy.length ? scaleBy : scaleSource;
      yMax = 0;
      considered.forEach((s) => {
        for (let i = 0; i < s.vals.length; i++) if (s.vals[i] > yMax) yMax = s.vals[i];
      });
      if (o.analytic) {
        const probe = [];
        for (let i = 0; i < 160; i++) {
          const v = analyticAt(o.analytic, lo + ((hi - lo) * i) / 159, "density");
          if (Number.isFinite(v)) probe.push(v);
        }
        probe.sort((a, b) => a - b);
        if (probe.length) yMax = Math.max(yMax, probe[Math.floor(probe.length * 0.99)]);
      }
      yMax = yMax > 0 ? yMax * 1.14 : 1;
    }

    const X = (v) => x0 + ((v - lo) / (hi - lo)) * iw;
    const Y = (v) => y0 + ih - (v / yMax) * ih;
    const binX = (i) => x0 + ((i + 0.5) / bins) * iw;

    // ---- grid + axes (full panels only)
    if (!compact) {
      ctx.save();
      ctx.font = `13px ${SANS}`;
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ticks(0, yMax, mode === "cdf" ? 4 : 4).forEach((t) => {
        if (t < 0 || t > yMax) return;
        const y = Math.round(Y(t)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + iw, y); ctx.stroke();
        ctx.fillStyle = INK3;
        ctx.fillText(fmt(t, mode === "cdf" ? 2 : null), x0 - 7, y);
      });
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      // One label per ~110px: a 6-character value like "-0.600" needs the room,
      // and these panels get narrow when two sit side by side.
      const nx = Math.max(2, Math.min(6, Math.round(iw / 110)));
      ticks(lo, hi, nx).forEach((t) => {
        if (t < lo || t > hi) return;
        const x = Math.round(X(t)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + ih); ctx.stroke();
        ctx.fillStyle = INK3;
        ctx.fillText(fmt(t), x, y0 + ih + 6);
      });
      ctx.strokeStyle = AXIS;
      ctx.beginPath();
      ctx.moveTo(x0 + 0.5, y0); ctx.lineTo(x0 + 0.5, y0 + ih + 0.5); ctx.lineTo(x0 + iw, y0 + ih + 0.5);
      ctx.stroke();
      ctx.restore();
    }

    // ---- PMC histogram bars behind the curves
    const pmcLayer = o.layers.find((L) => L.key === "pmc");
    if (o.bars && pmcLayer && mode !== "cdf" && pmcLayer.n) {
      const d = density(pmcLayer.counts, pmcLayer.n, lo, hi);
      ctx.fillStyle = "rgba(200,144,26,0.22)";
      const bw = iw / bins;
      for (let i = 0; i < bins; i++) {
        if (d[i] <= 0) continue;
        const y = Y(d[i]);
        ctx.fillRect(x0 + i * bw, y, Math.max(bw - 0.5, 0.5), y0 + ih - y);
      }
    }

    // ---- the analytic reference, subordinate to the sampled curves
    if (o.analytic && o.showAnalytic) {
      ctx.save();
      ctx.strokeStyle = COLORS.gen;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < 200; i++) {
        const x = lo + ((hi - lo) * i) / 199;
        const v = analyticAt(o.analytic, x, mode);
        if (!Number.isFinite(v)) { started = false; continue; }
        const py = Y(Math.min(v, yMax * 1.3));
        if (!started) { ctx.moveTo(X(x), py); started = true; } else ctx.lineTo(X(x), py);
      }
      ctx.stroke();
      ctx.restore();
    }

    // ---- the layers
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    series.forEach((s) => {
      if (!s.n) return;
      ctx.strokeStyle = COLORS[s.key];
      ctx.lineWidth = compact ? 1.3 : 1.8;
      ctx.beginPath();
      if (mode === "cdf") {
        ctx.moveTo(x0, Y(0));
        for (let i = 0; i < bins; i++) {
          const x = x0 + ((i + 1) / bins) * iw;
          ctx.lineTo(x, Y(s.vals[i]));
        }
      } else {
        ctx.moveTo(binX(0), Y(s.vals[0]));
        for (let i = 1; i < bins; i++) ctx.lineTo(binX(i), Y(s.vals[i]));
      }
      ctx.stroke();
    });
    ctx.restore();

    // ---- compact panels: a hairline baseline and the two endpoints
    if (compact) {
      ctx.save();
      ctx.strokeStyle = "#dedbd6";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y0 + ih + 0.5);
      ctx.lineTo(x0 + iw, y0 + ih + 0.5);
      ctx.stroke();
      ctx.fillStyle = INK;
      ctx.font = `13px ${SANS}`;
      ctx.textBaseline = "top";
      ctx.textAlign = "left";
      ctx.fillText(fmt(lo), x0, y0 + ih + 3);
      ctx.textAlign = "right";
      ctx.fillText(fmt(hi), x0 + iw, y0 + ih + 3);
      ctx.restore();
    }

    // ---- axis titles
    if (!compact) {
      ctx.save();
      ctx.fillStyle = INK2;
      ctx.font = `14px ${SANS}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      if (o.xLabel) ctx.fillText(o.xLabel, x0 + iw / 2, rect.y + rect.h - 4);
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle = INK;
      ctx.font = `600 14px ${SANS}`;
      if (o.title) ctx.fillText(o.title, x0 + 6, y0 + 3);
      ctx.restore();
    }

    return { X, Y, x0, y0, iw, ih, yMax, series, bins };
  }

  // --------------------------------------------------------- the line chart
  function lines(canvas, o) {
    const { ctx, w, h } = fit(canvas);

    const pts = o.series.flatMap((s) => s.points);
    if (!pts.length) {
      ctx.fillStyle = INK3; ctx.font = `13px ${SANS}`; ctx.textAlign = "center";
      ctx.fillText("No data for this selection.", w / 2, h / 2);
      return null;
    }

    const logX = o.logX !== false, logY = !!o.logY;
    const xs = pts.map((p) => p[0]).filter((v) => !logX || v > 0);
    const ys = pts.map((p) => p[1]).filter((v) => !logY || v > 0);
    if (!xs.length || !ys.length) return null;

    const fx = logX ? Math.log10 : (v) => v, fy = logY ? Math.log10 : (v) => v;
    let x0 = fx(Math.min(...xs)), x1 = fx(Math.max(...xs));
    let y0 = fy(Math.min(...ys)), y1 = fy(Math.max(...ys));
    if (x1 - x0 < 1e-9) { x0 -= 0.5; x1 += 0.5; }
    const padY = (y1 - y0) * 0.08 || Math.abs(y1) * 0.1 || 1;
    y0 -= padY; y1 += padY;
    if (!logY && y0 > 0) y0 = 0;

    const yT = logY
      ? Array.from({ length: Math.ceil(y1) - Math.floor(y0) + 1 },
          (_, i) => Math.pow(10, Math.floor(y0) + i)).filter((v) => fy(v) >= y0 && fy(v) <= y1)
      : ticks(y0, y1, 5);

    // The left margin is measured, not guessed: a rotated axis title plus the
    // widest tick label it must clear. Guessing collides as soon as a label
    // turns into something like "1.0e-4".
    const TITLE_BAND = o.yLabel ? 28 : 0;
    ctx.font = `13px ${SANS}`;
    const widestTick = Math.max(...yT.map((t) => ctx.measureText(fmt(t)).width), 0);
    const pad = {
      l: Math.round(TITLE_BAND + widestTick + 14),
      r: 24, t: 24, b: 52,
    };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 10 || ih <= 10) return null;

    const X = (v) => pad.l + ((fx(v) - x0) / (x1 - x0)) * iw;
    const Y = (v) => pad.t + ih - ((fy(v) - y0) / (y1 - y0)) * ih;

    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yT.forEach((t) => {
      const y = Math.round(Y(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iw, y); ctx.stroke();
      ctx.fillStyle = INK3; ctx.fillText(fmt(t), pad.l - 8, y);
    });

    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const xT = logX
      ? Array.from({ length: Math.floor(x1) - Math.ceil(x0) + 1 },
          (_, i) => Math.pow(10, Math.ceil(x0) + i))
      : ticks(x0, x1, 6);
    (xT.length > 1 ? xT : [Math.pow(10, x0), Math.pow(10, x1)]).forEach((t) => {
      const x = Math.round(X(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ih); ctx.stroke();
      ctx.fillStyle = INK3;
      ctx.fillText(o.xFormat ? o.xFormat(t) : fmt(t), x, pad.t + ih + 8);
    });

    ctx.strokeStyle = AXIS;
    ctx.beginPath();
    ctx.moveTo(pad.l + 0.5, pad.t); ctx.lineTo(pad.l + 0.5, pad.t + ih + 0.5); ctx.lineTo(pad.l + iw, pad.t + ih + 0.5);
    ctx.stroke();

    ctx.fillStyle = INK2; ctx.font = `14px ${SANS}`;
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    if (o.xLabel) ctx.fillText(o.xLabel, pad.l + iw / 2, h - 13);
    if (o.yLabel) {
      // Baseline sits 19px from the left edge; with "alphabetic" the glyphs
      // rise away from the ticks, leaving the measured band clear.
      ctx.save();
      ctx.translate(19, pad.t + ih / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(o.yLabel, 0, 0);
      ctx.restore();
    }

    ctx.lineWidth = 1.8; ctx.lineJoin = "round";
    o.series.forEach((s) => {
      ctx.strokeStyle = s.color;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let open = false;
      s.points.forEach((p) => {
        if ((logX && p[0] <= 0) || (logY && p[1] <= 0)) { open = false; return; }
        const x = X(p[0]), y = Y(p[1]);
        if (!open) { ctx.moveTo(x, y); open = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      if (s.points.length <= 24) {
        ctx.fillStyle = s.color;
        s.points.forEach((p) => {
          if ((logX && p[0] <= 0) || (logY && p[1] <= 0)) return;
          ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.5, 0, Math.PI * 2); ctx.fill();
        });
      }
    });

    // diversities that the explorer can open, marked on the axis
    if (o.marks && o.marks.length) {
      ctx.save();
      ctx.strokeStyle = INK3;
      ctx.fillStyle = INK3;
      ctx.setLineDash([2, 3]);
      ctx.lineWidth = 1;
      ctx.font = `13px ${SANS}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      o.marks.forEach((m) => {
        if (fx(m) < x0 || fx(m) > x1) return;
        const x = Math.round(X(m)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, pad.t + ih); ctx.lineTo(x, pad.t + 6); ctx.stroke();
        ctx.fillText(String(m), x, pad.t + 4);
      });
      ctx.restore();
    }

    return {
      X, Y, pad, iw, ih,
      invX: (px) => (logX
        ? Math.pow(10, x0 + ((px - pad.l) / iw) * (x1 - x0))
        : x0 + ((px - pad.l) / iw) * (x1 - x0)),
    };
  }

  return { fit, bin, smooth, density, cdf, marginal, lines, fmt, ticks,
           analyticAt, COLORS, LABELS };
})();
