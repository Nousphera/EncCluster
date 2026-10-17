import json
from enccluster.train import main, partition


def fixture_args(tmp_path):
    from enccluster.dataset_io import prepare_dataset
    from test_preparation import fixture_images

    prepare_dataset(
        "imagefolder", tmp_path / "data", source=fixture_images(tmp_path / "source")
    )
    return [
        "--dataset",
        "imagefolder",
        "--data-root",
        str(tmp_path / "data"),
        "--clients",
        "2",
        "--batch-size",
        "8",
        "--alpha",
        "10",
        "--rounds",
        "2",
    ]


def test_two_round_training_writes_real_metrics(tmp_path):
    output = tmp_path / "run.json"
    result = main(
        fixture_args(tmp_path) + ["--output", str(output), "--method", "fedavg"]
    )
    assert len(result["history"]) == 2
    assert all(0 <= r["accuracy"] <= 1 for r in result["history"])
    assert json.loads(output.read_text())["configuration"]["seed"] == 42
    assert result["benchmark_reproduction"] is False


def test_dirichlet_partitions_preserve_disjoint_sample_identity():
    subsets = partition([0] * 50 + [1] * 50, clients=3, alpha=10, seed=7)
    flat = [i for subset in subsets for i in subset]
    assert sorted(flat) == list(range(100)) and len(set(flat)) == 100


def test_output_refuses_overwrite(tmp_path):
    import pytest

    output = tmp_path / "old.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert output.read_text() == "preserve"


def test_checkpoint_is_loadable_and_matches_report(tmp_path):
    import torch

    output = tmp_path / "new.json"
    result = main(
        fixture_args(tmp_path)
        + ["--output", str(output), "--method", "fedavg", "--rounds", "1"]
    )
    checkpoint = torch.load(
        output.with_suffix(".pt"), weights_only=True, map_location="cpu"
    )
    assert checkpoint["configuration"] == result["configuration"]
    assert result["configuration"]["model"] == "resnet20"
    assert checkpoint["model"]["body.backbone.conv1.weight"].shape == (16, 3, 3, 3)
    assert result["environment"]["torch"]


def test_mobilenet_rejects_singleton_batch_before_training():
    from enccluster.train import build_parser, validate_args
    import pytest

    args = build_parser().parse_args(["--model", "mobilenet_v2", "--batch-size", "1"])
    with pytest.raises(ValueError, match="MobileNet.*two"):
        validate_args(args)


def test_mobilenet_rejects_singleton_partition_before_model(monkeypatch):
    from enccluster import experiment
    from enccluster.train import build_parser
    import pytest

    args = build_parser().parse_args(["--model", "mobilenet_v2", "--clients", "2"])
    monkeypatch.setattr(
        experiment, "load_data", lambda _: (None, None, [[0], [1, 2]], [0, 1, 1], 2)
    )
    monkeypatch.setattr(
        experiment,
        "ImageModel",
        lambda *_: pytest.fail("Allocated model before validating partitions"),
    )
    with pytest.raises(ValueError, match="MobileNet.*two"):
        experiment.Experiment(args)


def test_singleton_batch_sampler_length_matches_iteration():
    from enccluster.train import CompleteBatchSampler

    sampler = CompleteBatchSampler(3, 1, False)
    assert len(sampler) == len(list(sampler)) == 3
