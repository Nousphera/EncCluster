# Modified 2026-10-15: standalone training integration; see LICENSE.
"""EncCluster local experiment runner.

Native Flower transport is exposed separately by scripts/run_experiment.py.
Real CIFAR data, full paper backbones and independent data/training seeds share
these training primitives with the federated client application.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT = "EncCluster"


class ImageModel(torch.nn.Module):
    def __init__(self, name, classes):
        super().__init__()
        from enccluster.network import Model

        self.body = Model(name, classes)
        self.input_mean = (0.4914, 0.4822, 0.4465)
        self.input_std = (0.247, 0.2435, 0.2616)
        self.label2class = {i: i for i in range(classes)}

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        mean = x.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = x.new_tensor(self.input_std).view(1, 3, 1, 1)
        return self.body((x - mean) / std)


def partition(targets, clients, alpha, seed):
    """Class-wise Dirichlet partitions; reject empty clients explicitly."""
    if clients < 2 or alpha <= 0:
        raise ValueError("Use at least two clients and positive Dirichlet alpha")
    rng = np.random.default_rng(seed)
    result = [[] for _ in range(clients)]
    labels = np.asarray(targets)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.repeat(alpha, clients))
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for subset, indices in zip(result, np.split(indices, cuts)):
            subset.extend(indices.tolist())
    if any((not x for x in result)):
        raise ValueError(
            "Empty Dirichlet client; increase data/alpha or choose another seed"
        )
    return result


def load_data(args):
    from torchvision import transforms
    from .dataset_io import load_prepared, prepare_dataset, labels

    size = 32
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    if args.download:
        prepare_dataset(args.dataset, args.data_root, download=True)
    train, test, classes = load_prepared(
        args.dataset, args.data_root, transform, test_transform=test_transform
    )
    targets = labels(train)
    subsets = partition(targets, args.clients, args.alpha, args.partition_seed)
    return train, test, subsets, targets, classes


class CompleteBatchSampler:
    """Keep every sample while merging a singleton tail for BatchNorm models."""

    def __init__(self, size, batch_size, shuffle):
        self.size, self.batch_size, self.shuffle = size, batch_size, shuffle

    def __iter__(self):
        order = (
            torch.randperm(self.size).tolist()
            if self.shuffle
            else list(range(self.size))
        )
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, self.size, self.batch_size)
        ]
        if self.batch_size > 1 and len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        count = math.ceil(self.size / self.batch_size)
        return count - 1 if count > 1 and self.size % self.batch_size == 1 else count


def loader(dataset, indices, args, shuffle=True):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_sampler=CompleteBatchSampler(len(subset), args.batch_size, shuffle),
        num_workers=0,
    )


def fit(model, data, epochs, lr):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        for x, y in data:
            x, y = (x.to(model.device), y.to(model.device))
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, data):
    model.eval()
    correct, total, loss = (0, 0, 0.0)
    with torch.no_grad():
        for x, y in data:
            x, y = (x.to(model.device), y.to(model.device))
            logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
            loss += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
    return (
        {"accuracy": correct / total, "loss": loss / total, "samples": total}
        if total
        else {"accuracy": None, "loss": None, "samples": 0}
    )


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "imagefolder"], default="cifar10"
    )
    p.add_argument(
        "--model",
        choices=["resnet20", "convnext", "convmixer256_8", "mobilenet_v2"],
        default="resnet20",
    )
    p.add_argument("--clients", type=int, default=30)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Release optimizer setting; paper does not specify learning rate",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dirichlet concentration: 0.1 non-IID; 10 paper IID",
    )
    p.add_argument("--participation", type=float, default=1.0)
    p.add_argument("--kappa", type=int, default=128)
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--max-abs", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition-seed", type=int, default=2025)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--download",
        action="store_true",
        help="Prepare missing public data before running",
    )
    p.add_argument("--output", default=None)
    p.add_argument(
        "--method",
        choices=["enccluster", "bfuse", "cluster", "fedavg"],
        default="enccluster",
    )
    p.add_argument("--bpe", type=int, choices=[8, 16], default=8)
    return p


def validate_args(args):
    if args.model == "mobilenet_v2" and args.batch_size < 2:
        raise ValueError("MobileNet training requires at least two samples per batch")
    if (
        args.clients < 2
        or min(args.rounds, args.batch_size, args.local_epochs, args.kappa) < 1
    ):
        raise ValueError("Positive sizes/rounds and at least two clients are required")
    if (
        not math.isfinite(args.alpha)
        or args.alpha <= 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError(
            "Learning rate and Dirichlet concentration must be positive and finite"
        )
    if (
        not 0 < args.participation <= 1
        or math.floor(args.clients * args.participation) < 2
    ):
        raise ValueError("Participation must select at least two clients")
    if not 0 <= args.seed < 2**32 or not 0 <= args.partition_seed < 2**32:
        raise ValueError("Seeds must be unsigned 32-bit integers")
    if args.precision < 0 or not math.isfinite(args.max_abs) or args.max_abs <= 0:
        raise ValueError("Precision must be nonnegative and max-abs positive/finite")
    if args.output:
        destination = Path(args.output)
        if destination.suffix != ".json":
            raise ValueError("--output must end in .json")
        if destination.exists() or destination.with_suffix(".pt").exists():
            raise FileExistsError(
                f"Output already exists: {destination}; choose a new name"
            )
    return args


def main(argv=None):
    from .experiment import Experiment

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    experiment = Experiment(args)
    state = {
        key: value.detach().cpu().clone()
        for key, value in experiment.model.state_dict().items()
    }
    history = []
    caches = {cid: {} for cid in range(args.clients)}
    for round_id in range(1, args.rounds + 1):
        selected = experiment.participants(round_id)
        results = [
            experiment.client_update(cid, state, round_id, caches[cid])
            for cid in selected
        ]
        states, counts, _ = zip(*results)
        if args.method == "enccluster":
            from .methods.enccluster import aggregate

            state = aggregate(
                states,
                counts,
                kappa=args.kappa,
                bpe=args.bpe,
                seed=args.seed,
                precision=args.precision,
                max_abs=args.max_abs,
            )
        else:
            state = experiment.aggregate_plaintext(
                states, counts, selected, state, round_id
            )
        metrics = {
            "round": round_id,
            "participants": selected,
            "sample_counts": list(counts),
            **experiment.evaluate(state),
        }
        history.append(metrics)
        print(json.dumps(metrics))
    result = experiment.result(history)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        torch.save(
            {
                "model": {
                    key: value.detach().cpu()
                    for key, value in experiment.model.state_dict().items()
                },
                "configuration": vars(args),
                "classes": experiment.classes,
            },
            path.with_suffix(".pt"),
        )
    return result


def cli():
    main()


if __name__ == "__main__":
    cli()
