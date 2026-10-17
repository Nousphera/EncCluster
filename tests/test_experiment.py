import torch
import pytest
from argparse import Namespace


def arguments(tmp_path):
    from test_preparation import fixture_images
    from enccluster.dataset_io import prepare_dataset
    from enccluster.train import build_parser, validate_args

    prepare_dataset(
        "imagefolder", tmp_path / "data", source=fixture_images(tmp_path / "source")
    )
    args = build_parser().parse_args(
        [
            "--dataset",
            "imagefolder",
            "--data-root",
            str(tmp_path / "data"),
            "--clients",
            "2",
            "--rounds",
            "1",
            "--method",
            "fedavg",
            "--alpha",
            "10",
            "--batch-size",
            "8",
        ]
    )
    validate_args(args)
    return args


def test_native_client_api_updates_full_model_and_evaluates_aggregate(tmp_path):
    from enccluster.experiment import Experiment

    experiment = Experiment(arguments(tmp_path))
    initial = {
        key: value.detach().cpu().clone()
        for key, value in experiment.model.state_dict().items()
    }
    ids = experiment.participants(1)
    results = [experiment.client_update(cid, initial, 1, {}) for cid in ids]
    states, counts, _ = zip(*results)
    combined = experiment.aggregate_plaintext(states, counts, ids, initial, 1)
    assert sum(counts) == 16 and len(combined) == len(initial)
    assert any(
        not torch.equal(initial[k], torch.as_tensor(combined[k])) for k in initial
    )
    metrics = experiment.evaluate(combined)
    assert 0 <= metrics["accuracy"] <= 1 and metrics["samples"] == 6
    report = experiment.result([metrics])
    assert (
        len(report["data_manifest_sha256"]) == 64
        and len(report["partition_sha256"]) == 64
    )


def test_partition_is_independent_of_training_seed(tmp_path):
    from enccluster.experiment import Experiment

    args = arguments(tmp_path)
    first = Experiment(args).partitions
    args.seed += 1
    assert Experiment(args).partitions == first


def test_public_parser_rejects_synthetic_data():
    from enccluster.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset", "synthetic"])


def test_all_floating_bn_state_is_clustered():
    from enccluster.clustering import flatten_state, aggregate_buffers

    state = {
        "weight": torch.tensor([1.0]),
        "running_mean": torch.tensor([2.0]),
        "running_var": torch.tensor([3.0]),
        "num_batches_tracked": torch.tensor(4),
    }
    values, keys, _ = flatten_state(state)
    assert list(values) == [1.0, 2.0, 3.0] and keys == [
        "weight",
        "running_mean",
        "running_var",
    ]
    assert set(aggregate_buffers([state, state], [1, 1])) == {"num_batches_tracked"}


def test_training_loader_merges_singleton_tail_without_dropping_samples():
    from enccluster.train import loader
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.arange(9), torch.arange(9))
    batches = list(
        loader(dataset, list(range(9)), Namespace(batch_size=4), shuffle=False)
    )
    assert [len(x) for x, _ in batches] == [4, 5]
    assert torch.cat([x for x, _ in batches]).tolist() == list(range(9))
