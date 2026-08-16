from __future__ import annotations

import pytest
import torch

from qwen_j_lens.nla.capacity import (
    bit_budget_sweep,
    capacity_report,
    code_bits,
    count_code_usage,
    entropy_weight_sweep,
    linear_reference,
    row_count_sweep,
    trivial_baselines,
)
from qwen_j_lens.nla.train import train_autoencoder


def _clustered(rows: int, width: int, clusters: int, seed: int = 0) -> torch.Tensor:
    """Rows drawn from a small number of centers, so a tiny code can fit them."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(clusters, width, generator=generator)
    picks = torch.randint(clusters, (rows,), generator=generator)
    noise = 0.01 * torch.randn(rows, width, generator=generator)
    return centers[picks] + noise


_FIT = dict(
    code_length=2,
    code_vocab=8,
    width=32,
    embedding_dim=8,
    temperature=1.0,
    steps=120,
    lr=0.01,
    batch_size=32,
    seed=0,
)


def test_zero_predictor_baseline_is_exactly_one() -> None:
    baselines = trivial_baselines(torch.randn(40, 12))

    assert baselines.zero_predictor == 1.0


def test_constant_offset_drives_the_mean_predictor_baseline() -> None:
    width = 16
    offset = 50.0 * torch.ones(1, width)
    spread = torch.randn(64, width)

    centered = trivial_baselines(spread)
    shifted = trivial_baselines(spread + offset)

    assert centered.mean_predictor == pytest.approx(1.0, abs=0.05)
    assert shifted.mean_predictor < 0.05
    assert shifted.constant_energy_fraction > 0.95


def test_baselines_reject_degenerate_input() -> None:
    with pytest.raises(ValueError):
        trivial_baselines(torch.randn(4, 3, 2))
    with pytest.raises(ValueError):
        trivial_baselines(torch.zeros(4, 3))


def test_code_usage_never_exceeds_the_available_code_space() -> None:
    activations = _clustered(48, 32, clusters=3)
    result = train_autoencoder(activations, entropy_weight=0.01, **_FIT)

    usage = count_code_usage(result.model, activations, code_length=2, code_vocab=8)

    assert usage.available_codes == 8**2
    assert 1 <= usage.distinct_codes <= usage.available_codes
    assert 0.0 <= usage.occupancy <= 1.0


def test_redundant_rows_stay_reconstructable_as_the_row_count_grows() -> None:
    """A genuinely low-rank sample keeps a low error at every row count.

    This is the positive control for the diagnostic. When the error rises only
    because rows were added, the low value at the small count was redundancy.
    """
    activations = _clustered(96, 32, clusters=3)

    points = row_count_sweep(
        activations, [24, 96], entropy_weight=0.01, **_FIT
    )

    assert [point.rows for point in points] == [24, 96]
    assert all(point.relative_error < 0.1 for point in points)


def test_row_sweep_skips_counts_beyond_the_sample() -> None:
    activations = _clustered(32, 32, clusters=2)

    points = row_count_sweep(
        activations, [16, 32, 4096], entropy_weight=0.01, **_FIT
    )

    assert [point.rows for point in points] == [16, 32]


def test_row_sweep_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError):
        row_count_sweep(_clustered(8, 4, 2), [0], entropy_weight=0.01, **_FIT)


def test_entropy_sweep_records_one_point_per_weight() -> None:
    activations = _clustered(48, 32, clusters=3)

    points = entropy_weight_sweep(activations, [0.01, 0.5], **_FIT)

    assert [point.entropy_weight for point in points] == [0.01, 0.5]
    assert all(point.rows == 48 for point in points)
    assert all(point.distinct_codes >= 1 for point in points)


def test_entropy_sweep_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        entropy_weight_sweep(_clustered(8, 4, 2), [-1.0], **_FIT)


def test_report_flags_a_code_that_beats_the_mean_predictor() -> None:
    activations = _clustered(96, 32, clusters=3)

    report = capacity_report(
        activations,
        row_counts=[24, 96],
        entropy_weights=[0.01, 0.5],
        entropy_weight=0.01,
        **_FIT,
    )

    assert report["activation_rows"] == 96
    assert report["hidden_size"] == 32
    assert report["available_codes"] == 64
    assert len(report["row_count_sweep"]) == 2
    assert len(report["entropy_weight_sweep"]) == 2
    assert report["beats_mean_predictor"] is True


def test_report_flags_a_code_that_carries_no_information() -> None:
    """Full-rank rows with a constant offset defeat a few discrete symbols.

    This mirrors the real residual-stream case. A modest constant offset makes
    the mean predictor a genuinely strong baseline, so a collapsed code that
    reconstructs nothing per row cannot clear it.
    """
    generator = torch.Generator().manual_seed(3)
    activations = torch.randn(512, 256, generator=generator) + 0.23

    report = capacity_report(
        activations,
        row_counts=[512],
        entropy_weights=[0.01],
        entropy_weight=0.01,
        **_FIT,
    )

    assert report["baselines"]["mean_predictor"] < 1.0
    assert report["beats_mean_predictor"] is False


def test_code_bits_counts_slots_times_log_vocab() -> None:
    assert code_bits(4, 64) == 24.0
    assert code_bits(16, 64) == 96.0
    assert code_bits(12, 256) == 96.0


def test_code_bits_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        code_bits(0, 64)
    with pytest.raises(ValueError):
        code_bits(4, 0)


def test_linear_reference_rank_zero_equals_the_mean_predictor() -> None:
    """Rank zero keeps only the mean, so the two baselines must agree exactly."""
    activations = _clustered(200, 48, clusters=6, seed=5)

    baselines = trivial_baselines(activations)
    points = linear_reference(activations, [0])

    assert points[0].rank == 0
    assert points[0].relative_error == pytest.approx(baselines.mean_predictor, abs=1e-6)


def test_linear_reference_recovers_a_known_low_rank_structure() -> None:
    generator = torch.Generator().manual_seed(11)
    basis = torch.randn(5, 40, generator=generator)
    coefficients = torch.randn(300, 5, generator=generator)
    activations = coefficients @ basis

    points = linear_reference(activations, [1, 5, 20])
    errors = {point.rank: point.relative_error for point in points}

    assert errors[5] < 1e-8
    assert errors[1] > errors[5]
    assert errors[20] < 1e-8


def test_linear_reference_error_never_increases_with_rank() -> None:
    activations = _clustered(160, 32, clusters=8, seed=7)

    points = linear_reference(activations, [0, 2, 4, 8, 16])
    errors = [point.relative_error for point in points]

    assert errors == sorted(errors, reverse=True)


def test_linear_reference_rejects_negative_ranks() -> None:
    with pytest.raises(ValueError):
        linear_reference(torch.randn(10, 4), [-1])


def test_bit_budget_sweep_reports_bits_for_every_spec() -> None:
    activations = _clustered(120, 32, clusters=4, seed=9)

    points = bit_budget_sweep(
        activations,
        [(2, 8, 32), (4, 8, 32)],
        embedding_dim=8,
        temperature=1.0,
        steps=80,
        lr=0.01,
        batch_size=32,
        entropy_weight=0.1,
        seed=0,
    )

    assert [point.bits for point in points] == [6.0, 12.0]
    assert [point.code_length for point in points] == [2, 4]
    assert all(point.width == 32 for point in points)
    assert all(point.distinct_codes >= 1 for point in points)


def test_bit_budget_sweep_carries_width_as_its_own_axis() -> None:
    """Width must vary independently so it can be shown not to bind the frontier."""
    activations = _clustered(120, 32, clusters=4, seed=9)

    points = bit_budget_sweep(
        activations,
        [(4, 8, 16), (4, 8, 64)],
        embedding_dim=8,
        temperature=1.0,
        steps=80,
        lr=0.01,
        batch_size=32,
        entropy_weight=0.1,
        seed=0,
    )

    assert [point.width for point in points] == [16, 64]
    assert all(point.bits == 12.0 for point in points)
