from __future__ import annotations

import torch

from qwen_j_lens.nla.faithfulness import (
    FaithfulnessReport,
    code_usage,
    gist_versus_specific,
    independent_reader_gap,
)


def test_memorized_codes_produce_a_large_independent_gap() -> None:
    generator = torch.Generator().manual_seed(1)
    count = 40
    activations = torch.randn((count, 6), generator=generator)
    codes = torch.stack(
        (torch.arange(count), torch.zeros(count, dtype=torch.long)), dim=1
    )
    train_index = torch.arange(0, count // 2)
    test_index = torch.arange(count // 2, count)

    result = independent_reader_gap(
        codes,
        activations,
        train_index=train_index,
        test_index=test_index,
        code_vocab=count,
        steps=300,
        seed=0,
    )

    assert result["independent_test_error"] > result["co_adapted_test_error"]
    assert result["gap"] > 0.3


def test_generalizable_codes_produce_a_small_independent_gap() -> None:
    generator = torch.Generator().manual_seed(2)
    clusters = 4
    per_cluster = 20
    count = clusters * per_cluster
    centers = torch.randn((clusters, 6), generator=generator) * 3.0
    labels = torch.arange(count) % clusters
    activations = centers[labels] + torch.randn((count, 6), generator=generator) * 0.05
    codes = labels.unsqueeze(1)
    train_index = torch.arange(0, count // 2)
    test_index = torch.arange(count // 2, count)

    result = independent_reader_gap(
        codes,
        activations,
        train_index=train_index,
        test_index=test_index,
        code_vocab=clusters,
        steps=300,
        seed=0,
    )

    assert result["gap"] < 0.15
    assert result["independent_test_error"] < 0.3


def test_gist_only_reconstruction_drops_specific_probe() -> None:
    generator = torch.Generator().manual_seed(4)
    count = 80
    specific = torch.randn((count, 1), generator=generator)
    gist = torch.randn((count, 5), generator=generator)
    activations = torch.cat((specific, gist), dim=1)
    destroyed = activations.clone()
    destroyed[:, 0] = 0.0
    train_index = torch.arange(0, count // 2)
    test_index = torch.arange(count // 2, count)

    result = gist_versus_specific(
        activations,
        destroyed,
        specific.squeeze(1),
        train_index=train_index,
        test_index=test_index,
    )

    assert result["original_specific_r2"] > 0.9
    assert result["reconstructed_specific_r2"] < 0.3


def test_faithful_reconstruction_retains_specific_probe() -> None:
    generator = torch.Generator().manual_seed(6)
    count = 80
    specific = torch.randn((count, 1), generator=generator)
    gist = torch.randn((count, 5), generator=generator)
    activations = torch.cat((specific, gist), dim=1)
    train_index = torch.arange(0, count // 2)
    test_index = torch.arange(count // 2, count)

    result = gist_versus_specific(
        activations,
        activations.clone(),
        specific.squeeze(1),
        train_index=train_index,
        test_index=test_index,
    )

    assert result["specific_retained_fraction"] > 0.8


def test_code_usage_distinguishes_memorized_from_clustered() -> None:
    count = 40
    memorized = torch.stack(
        (torch.arange(count), torch.zeros(count, dtype=torch.long)), dim=1
    )
    clustered = (torch.arange(count) % 4).unsqueeze(1)

    memorized_usage = code_usage(memorized, code_vocab=count)
    clustered_usage = code_usage(clustered, code_vocab=4)

    assert memorized_usage["distinct_code_fraction"] == 1.0
    assert clustered_usage["distinct_code_fraction"] < 0.2


def test_report_flags_and_serialization() -> None:
    report = FaithfulnessReport(
        reconstruction_error=0.1,
        independent_reader={"gap": 0.6, "co_adapted_test_error": 0.05, "independent_test_error": 0.65},
        gist_versus_specific={"specific_retained_fraction": 0.2, "original_specific_r2": 0.9, "reconstructed_specific_r2": 0.18},
        code_usage={"distinct_code_fraction": 0.95},
        gap_threshold=0.2,
        retained_threshold=0.5,
    )

    assert report.private_code_suspected
    assert report.gist_only_suspected
    assert not report.faithful
    mapping = report.to_mapping()
    assert mapping["flags"]["faithful"] is False
