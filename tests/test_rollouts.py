from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.rollouts import (
    compact_token_array,
    is_rollout_bundle,
    load_rollout_bundle,
    rollout_bundle_path,
    sample_bundles,
    save_rollout_bundle,
    save_samples_and_rollouts,
    split_rollout_arrays,
)


def test_rollout_bundle_path_is_a_sibling_of_the_sample_bundle() -> None:
    assert rollout_bundle_path("outputs/samples/T8_prior.npz") == Path(
        "outputs/samples/T8_prior_rollouts.npz"
    )


def test_sample_bundles_never_returns_the_rollout_sidecars(tmp_path: Path) -> None:
    """Readers that expect sample keys must not be handed a rollout bundle.

    The sidecars share the directory and the .npz extension, so a bare glob
    picks them up and every marginal plotter fails on the missing schema.
    """
    for name in (
        "T8_gaussian_L32.npz",
        "T8_gaussian_L32_rollouts.npz",
        "T64_prior.npz",
        "T64_prior_rollouts.npz",
    ):
        (tmp_path / name).write_bytes(b"")

    assert [path.name for path in sample_bundles(tmp_path)] == [
        "T64_prior.npz",
        "T8_gaussian_L32.npz",
    ]
    assert is_rollout_bundle("a/T8_prior_rollouts.npz")
    assert not is_rollout_bundle("a/T8_prior.npz")
    # a bundle whose stem merely mentions rollouts is still a sample bundle
    assert not is_rollout_bundle("a/rollouts_summary.npz")


def test_compact_token_array_picks_the_smallest_lossless_dtype() -> None:
    assert compact_token_array(np.empty((0, 3), dtype=np.int64)).dtype == np.int16
    assert compact_token_array(np.array([[0, 9]], dtype=np.int64)).dtype == np.int16
    assert compact_token_array(np.array([[0, 40_000]])).dtype == np.int32
    assert compact_token_array(np.array([[0, 2**40]])).dtype == np.int64
    np.testing.assert_array_equal(
        compact_token_array(np.array([[3, 1, 2]])), np.array([[3, 1, 2]])
    )


def test_split_rollout_arrays_separates_traces_from_estimates() -> None:
    estimates, traces = split_rollout_arrays(
        {
            "pt": np.zeros(2),
            "theta_pool": np.ones(2),
            "rollout_x": np.zeros(3),
            "rollout_y": np.zeros(3),
        }
    )
    assert set(estimates) == {"pt", "theta_pool"}
    assert set(traces) == {"rollout_x", "rollout_y"}


def test_save_rollout_bundle_skips_empty_arrays(tmp_path: Path) -> None:
    path = tmp_path / "nothing_rollouts.npz"
    assert save_rollout_bundle(path, {}, step=np.int64(3)) is None
    assert not path.exists()


def test_save_samples_and_rollouts_writes_both_archives(tmp_path: Path) -> None:
    samples = {
        "pt": np.arange(4.0).reshape(2, 2),
        "rollout_y": np.arange(6.0).reshape(2, 3),
    }
    samples_path = tmp_path / "nested/T4_prior.npz"
    written = save_samples_and_rollouts(samples_path, samples)

    assert written == tmp_path / "nested/T4_prior_rollouts.npz"
    with np.load(samples_path) as archive:
        assert set(archive.files) == {"pt"}
    traces = load_rollout_bundle(written)
    np.testing.assert_array_equal(traces["rollout_y"], samples["rollout_y"])


def test_save_samples_and_rollouts_can_drop_the_traces(tmp_path: Path) -> None:
    samples_path = tmp_path / "T4_prior.npz"
    written = save_samples_and_rollouts(
        samples_path,
        {"pt": np.zeros(2), "rollout_y": np.zeros(3)},
        save_rollouts=False,
    )

    assert written is None
    assert samples_path.is_file()
    assert not rollout_bundle_path(samples_path).exists()
