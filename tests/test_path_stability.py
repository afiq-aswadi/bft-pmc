from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.path_stability as ps
from analysis.rollouts import rollout_bundle_path, save_rollout_bundle


def _write_lr_bundle(
    path: Path,
    *,
    prompt_len: int = 2,
    n_prompts: int = 2,
    n_rollouts: int = 3,
    total_len: int = 6,
    dim: int = 2,
) -> Path:
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((n_prompts, n_rollouts, total_len, dim))
    ys = rng.standard_normal((n_prompts, n_rollouts, total_len))
    written = save_rollout_bundle(
        rollout_bundle_path(path),
        {"rollout_x": xs, "rollout_y": ys},
        prompt_len=np.int64(prompt_len),
        prompt_source="memorising",
        num_tasks=np.int64(4),
        step=np.int64(7),
    )
    assert written is not None
    return written


def _write_bau_bundle(path: Path, *, prompt_len: int = 2) -> Path:
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 3, size=(2, 3, 6)).astype(np.int16)
    written = save_rollout_bundle(
        rollout_bundle_path(path),
        {"rollout_tokens": tokens},
        prompt_tokens=tokens[:, 0, :prompt_len],
        step=np.int64(4),
        prompt_source="data_memorising",
    )
    assert written is not None
    return written


def _write_markov_bundle(path: Path, *, key: str = "rollout_states") -> Path:
    rng = np.random.default_rng(2)
    states = rng.integers(0, 3, size=(2, 3, 6)).astype(np.int16)
    written = save_rollout_bundle(
        rollout_bundle_path(path),
        {key: states},
        prompt_tokens=states[:, 0, :2],
        n_chains=np.int64(8),
    )
    assert written is not None
    return written


def test_infer_family_reads_the_trace_key() -> None:
    assert ps.infer_family({"rollout_x": np.zeros(1), "rollout_y": np.zeros(1)}) == "lr"
    assert ps.infer_family({"rollout_tokens": np.zeros(1)}) == "bau"
    assert ps.infer_family({"rollout_states": np.zeros(1)}) == "markov"
    assert ps.infer_family({"rollout_posterior_states": np.zeros(1)}) == "markov"
    with pytest.raises(ValueError, match="No known rollout traces"):
        ps.infer_family({"model_samples": np.zeros(1)})


def test_load_traces_normalises_shapes_and_metadata(tmp_path: Path) -> None:
    lr_path = _write_lr_bundle(tmp_path / "lr.npz")
    traces = ps.load_traces(lr_path)
    assert traces.family == "lr"
    assert traces.prompt_len == 2
    assert (traces.n_prompts, traces.n_rollouts, traces.total_len) == (2, 3, 6)
    assert traces.metadata["prompt_source"] == "memorising"
    assert traces.metadata["num_tasks"] == 4

    # a single-prompt bundle gains a leading prompt axis
    single = save_rollout_bundle(
        tmp_path / "single_rollouts.npz",
        {"rollout_x": np.zeros((3, 6, 2)), "rollout_y": np.zeros((3, 6))},
        prompt_len=np.int64(2),
    )
    assert single is not None
    assert ps.load_traces(single).n_prompts == 1

    markov_path = _write_markov_bundle(tmp_path / "mk.npz")
    markov_traces = ps.load_traces(markov_path)
    # prompt_len falls back to the stored prompt tokens
    assert markov_traces.family == "markov"
    assert markov_traces.prompt_len == 2
    assert (
        ps.load_traces(
            _write_markov_bundle(
                tmp_path / "mk_pmc.npz", key="rollout_posterior_states"
            )
        ).family
        == "markov"
    )

    bare = save_rollout_bundle(
        tmp_path / "bare_rollouts.npz", {"rollout_tokens": np.zeros((1, 6))}
    )
    assert bare is not None
    with pytest.raises(ValueError, match="neither prompt_len nor prompt_tokens"):
        ps.load_traces(bare)


def test_prefix_grid_starts_at_n_and_ends_at_the_rollout_length() -> None:
    np.testing.assert_array_equal(ps.prefix_grid(2, 6, 1), [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(ps.prefix_grid(2, 7, 2), [2, 4, 6, 7])
    with pytest.raises(ValueError, match="stride"):
        ps.prefix_grid(2, 6, 0)
    with pytest.raises(ValueError, match="prompt_len must be positive"):
        ps.prefix_grid(0, 6, 1)
    with pytest.raises(ValueError, match="total_len must exceed"):
        ps.prefix_grid(6, 6, 1)


def test_lr_theta_path_matches_least_squares_on_the_prefix() -> None:
    rng = np.random.default_rng(3)
    xs = rng.standard_normal((1, 1, 5, 2))
    ys = rng.standard_normal((1, 1, 5))
    grid = np.array([3, 5])
    paths = ps._lr_theta_paths(xs, ys, grid)
    for index, prefix in enumerate(grid):
        expected = np.linalg.lstsq(xs[0, 0, :prefix], ys[0, 0, :prefix], rcond=None)[0]
        np.testing.assert_allclose(paths[0, 0, index], expected, atol=1e-8)


def test_bau_and_markov_theta_paths_match_their_estimators() -> None:
    tokens = np.array([[[0, 1, 1, 2]]])
    frequencies = ps._bau_theta_paths(tokens, np.array([2, 4]))
    np.testing.assert_allclose(frequencies[0, 0, 0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(frequencies[0, 0, 1], [0.25, 0.5, 0.25])

    states = np.array([[[0, 1, 1, 0]]])
    matrices = ps._markov_theta_paths(states, np.array([4]), num_states=2)
    counts = np.full((2, 2), ps.MARKOV_SMOOTHING)
    counts[0, 1] += 1
    counts[1, 1] += 1
    counts[1, 0] += 1
    expected = counts / counts.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(matrices[0, 0, 0], expected.reshape(-1))


def test_scaled_l1_paths_start_at_zero(tmp_path: Path) -> None:
    curves = ps.analyse_bundle(_write_lr_bundle(tmp_path / "lr.npz"), setup="lr")
    assert curves.grid[0] == curves.prompt_len
    np.testing.assert_allclose(curves.expected[0], 0.0)
    assert np.all(curves.per_prompt[:, 0] == 0.0)
    assert curves.theta_reference.shape == (2, 2)
    assert curves.theta_terminal.shape == (2, 3, 2)


def test_theta_paths_dispatches_per_family(tmp_path: Path) -> None:
    for writer, name in (
        (_write_bau_bundle, "bau.npz"),
        (_write_markov_bundle, "mk.npz"),
    ):
        traces = ps.load_traces(writer(tmp_path / name))
        grid = ps.prefix_grid(traces.prompt_len, traces.total_len, 1)
        paths = ps.theta_paths(traces, grid)
        assert paths.shape[:3] == (2, 3, len(grid))


def test_curves_to_frames_produces_tidy_tables(tmp_path: Path) -> None:
    curve = ps.analyse_bundle(_write_lr_bundle(tmp_path / "lr.npz"), setup="a/lr")
    aggregate, per_prompt = ps.curves_to_frames([curve])
    assert set(aggregate["setup"]) == {"a/lr"}
    assert aggregate["T"].min() == 0
    assert aggregate["p"].iloc[0] == 2
    assert len(per_prompt) == len(aggregate) * 2
    assert aggregate.loc[aggregate["T"] == 0, "scaled_l1"].item() == 0.0

    # the per-prompt table must line up with the (prompt, prefix) curve matrix
    for prompt_index in range(curve.per_prompt.shape[0]):
        for prefix_index, prefix in enumerate(curve.grid):
            row = per_prompt[
                (per_prompt["prompt_idx"] == prompt_index) & (per_prompt["N"] == prefix)
            ]
            assert row["scaled_l1"].item() == pytest.approx(
                curve.per_prompt[prompt_index, prefix_index]
            )
    # label columns stay categorical so millions of rows cost codes, not strings
    assert str(per_prompt["setup"].dtype) == "category"


def test_plot_path_stability_writes_png_and_pdf(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "setup": ["a", "a", "b", "b"],
            "T": [0, 1, 0, 1],
            "scaled_l1": [0.0, 0.2, 0.0, 0.3],
        }
    )
    out_path = ps.plot_path_stability(frame, tmp_path / "figures" / "curve.png")
    assert out_path.is_file()
    assert out_path.with_suffix(".pdf").is_file()
    with pytest.raises(ValueError, match="No trajectories"):
        ps.plot_path_stability(frame.iloc[:0], tmp_path / "empty.png")


def test_plot_by_family_colours_each_encoding_and_panels_each_family(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "setup": ["lr-rope/a", "lr-rope/a", "bau-none/b", "bau-none/b", "markov/c"],
            "family": ["lr", "lr", "bau", "bau", "markov"],
            "pos_encoding": ["rope", "rope", "none", "none", "markov"],
            "T": [0, 1, 0, 1, 0],
            "scaled_l1": [0.0, 0.2, 0.0, 0.3, 0.0],
        }
    )
    out_path = ps.plot_path_stability_by_family(frame, tmp_path / "fig" / "byfam.png")
    assert out_path.is_file()
    assert out_path.with_suffix(".pdf").is_file()
    # an unmapped encoding still draws, in the default grey
    assert ps.ENCODING_COLOURS["rope"] != ps.ENCODING_COLOURS["none"]
    with pytest.raises(ValueError, match="No trajectories"):
        ps.plot_path_stability_by_family(frame.iloc[:0], tmp_path / "empty.png")

    unmapped = frame.assign(pos_encoding="alibi")
    assert ps.plot_path_stability_by_family(unmapped, tmp_path / "u.png").is_file()


def test_per_encoding_figures_are_standalone_and_share_family_scales(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "setup": ["lr-rope/a", "lr-rope/a", "lr-none/b", "lr-none/b", "markov/c"],
            "family": ["lr", "lr", "lr", "lr", "markov"],
            "pos_encoding": ["rope", "rope", "none", "none", "markov"],
            "T": [0, 1, 0, 1, 0],
            "scaled_l1": [0.0, 0.9, 0.0, 0.1, 0.0],
        }
    )
    written = ps.plot_path_stability_per_encoding(frame, tmp_path, stem="ps")
    assert [path.name for path in written] == [
        "ps_rope.png",
        "ps_none.png",
        "ps_markov.png",
    ]
    assert all(path.is_file() for path in written)

    # the shared limit comes from the loudest encoding, not each file's own data
    limits = ps.family_ylims(frame)
    assert limits["lr"] == pytest.approx((0.0, 0.9 * 1.05))
    # a family that never moves still gets a usable axis
    assert limits["markov"] == (0.0, 1.0)

    with pytest.raises(ValueError, match="No trajectories"):
        ps.plot_path_stability_per_encoding(frame.iloc[:0], tmp_path)


def test_input_parsing_and_bundle_discovery(tmp_path: Path) -> None:
    assert ps.parse_input("rope=/tmp/run") == ("rope", Path("/tmp/run"))
    assert ps.parse_input("/tmp/run") == ("run", Path("/tmp/run"))

    bundle = _write_lr_bundle(tmp_path / "nested" / "lr.npz")
    assert ps.discover_bundles(bundle) == [bundle]
    assert ps.discover_bundles(tmp_path) == [bundle]


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="at least one --inputs"):
        ps.PathStabilityConfig().validate()
    with pytest.raises(ValueError, match="stride"):
        ps.PathStabilityConfig(inputs=("a",), stride=0).validate()
    with pytest.raises(ValueError, match="realisations"):
        ps.PathStabilityConfig(inputs=("a",), realisations=-1).validate()


def test_collect_curves_skips_prior_bundles_and_reports_empties(tmp_path: Path) -> None:
    prior_only = tmp_path / "prior"
    prior_only.mkdir()
    written = save_rollout_bundle(
        prior_only / "T4_prior_rollouts.npz",
        {"rollout_x": np.zeros((2, 4, 2)), "rollout_y": np.zeros((2, 4))},
        prompt_len=np.int64(0),
    )
    assert written is not None
    with pytest.raises(RuntimeError, match="prior-mode"):
        ps.collect_curves(ps.PathStabilityConfig(inputs=(str(prior_only),)))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No \\*_rollouts.npz"):
        ps.collect_curves(ps.PathStabilityConfig(inputs=(str(empty),)))


def test_collect_curves_survives_bundles_that_vanish_or_truncate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sweep written or cleaned up concurrently must not abort the run."""
    samples = tmp_path / "samples"
    samples.mkdir()
    good = _write_lr_bundle(samples / "T4_memorising_L2.npz")

    # half-written bundle: a real npz truncated mid-archive
    truncated = samples / "T8_memorising_L2_rollouts.npz"
    payload = good.read_bytes()
    truncated.write_bytes(payload[: len(payload) // 2])
    # listed by the initial walk, deleted before it was opened
    removed = samples / "T16_memorising_L2_rollouts.npz"

    monkeypatch.setattr(ps, "discover_bundles", lambda root: [good, truncated, removed])
    curves = ps.collect_curves(ps.PathStabilityConfig(inputs=(str(samples),)))

    assert [curve.setup for curve in curves] == ["samples/T4_memorising_L2"]
    errors = capsys.readouterr().err
    assert "2 bundle(s) were unreadable" in errors
    assert truncated.name in errors and removed.name in errors

    monkeypatch.setattr(ps, "discover_bundles", lambda root: [truncated])
    with pytest.raises(RuntimeError, match="prior-mode or unreadable"):
        ps.collect_curves(ps.PathStabilityConfig(inputs=(str(samples),)))


def test_main_writes_every_reproduction_artifact(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    _write_lr_bundle(samples / "T4_memorising_L2.npz")
    _write_bau_bundle(samples / "run__source_data_memorising.npz")

    out_dir = tmp_path / "out"
    ps.main(
        ps.PathStabilityConfig(
            inputs=(f"rope={samples}",),
            output_dir=str(out_dir),
            stride=2,
        )
    )
    aggregate = pd.read_csv(out_dir / "path_stability.csv")
    assert set(aggregate["family"]) == {"lr", "bau"}
    assert aggregate["setup"].str.startswith("rope/").all()
    assert (out_dir / "path_stability_per_prompt.csv").is_file()
    assert (out_dir / "path_stability.png").is_file()
    assert (out_dir / "config.json").is_file()
    with np.load(out_dir / "theta_reference.npz") as references:
        assert "rope/T4_memorising_L2::theta_n" in references.files
        assert "rope/T4_memorising_L2::grid" in references.files

    # one trajectory per setup, from a single realisation of z_{1:n}
    per_prompt_dir = tmp_path / "out_per_prompt"
    ps.main(
        ps.PathStabilityConfig(
            inputs=(str(samples),),
            output_dir=str(per_prompt_dir),
            realisations=1,
        )
    )
    assert (per_prompt_dir / "path_stability.png").is_file()
    assert (per_prompt_dir / "path_stability_by_family.png").is_file()
