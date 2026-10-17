"""Release preparation contracts: real data, idempotence and immutable contents."""

import pytest


def test_prepare_defaults_to_all_paper_datasets(tmp_path, monkeypatch, capsys):
    from enccluster import dataset_io

    prepared = []

    def acquire(name, root, **options):
        prepared.append(name)
        return {"splits": {"train": {"samples": 10}, "test": {"samples": 2}}}

    monkeypatch.setattr(dataset_io, "prepare_dataset", acquire)
    dataset_io.main(["--root", str(tmp_path)])
    assert prepared == ["cifar10", "cifar100"]
    prepared.clear()
    dataset_io.main(["--dataset", "cifar10", "--root", str(tmp_path)])
    assert prepared == ["cifar10"]


def test_repeat_preparation_validates_existing_content(tmp_path):
    from enccluster.dataset_io import prepare_dataset, load_prepared
    from test_preparation import fixture_images

    source = fixture_images(tmp_path / "source")
    root = tmp_path / "data"
    original = prepare_dataset("imagefolder", root, source=source)
    assert prepare_dataset("imagefolder", root, source=source) == original
    (root / "imagefolder/train/a/0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="files differ"):
        prepare_dataset("imagefolder", root, source=source)
    with pytest.raises(ValueError, match="files differ"):
        load_prepared("imagefolder", root)


def test_train_and_test_transform_are_independent(tmp_path):
    from enccluster.dataset_io import prepare_dataset, load_prepared
    from test_preparation import fixture_images

    prepare_dataset(
        "imagefolder", tmp_path / "data", source=fixture_images(tmp_path / "source")
    )
    train, test, classes = load_prepared(
        "imagefolder",
        tmp_path / "data",
        lambda x: "train",
        test_transform=lambda x: "test",
    )
    assert train[0][0] == "train" and test[0][0] == "test" and classes == 2
