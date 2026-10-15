# Modified 2026-10-15: standalone training integration; see THIRD_PARTY_LICENSES.
"""Local federated training with decentralized coordinate-labelled FE.

Each client encrypts its expanded model vector, sending one ciphertext per
coordinate. See README.md for the protocol and experiment settings.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

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
    if args.dataset == "synthetic":
        generator = torch.Generator().manual_seed(args.seed)
        x = torch.rand(args.clients * args.samples_per_client, 3, 32, 32, generator=generator)
        y = x.mean(dim=(2, 3)).argmax(1).long()
        train = TensorDataset(x, y)
        test_x = torch.rand(4, 3, 32, 32, generator=generator)
        test = TensorDataset(test_x, test_x.mean(dim=(2, 3)).argmax(1))
        indices = [
            list(range(i * args.samples_per_client, (i + 1) * args.samples_per_client))
            for i in range(args.clients)
        ]
        return (train, test, indices, y.numpy(), 3)
    from torchvision import transforms
    from .dataset_io import load_prepared, prepare_dataset, labels

    size = 32
    transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    if args.download:
        prepare_dataset(args.dataset, args.data_root, download=True)
    train, test, classes = load_prepared(args.dataset, args.data_root, transform)
    targets = labels(train)
    if getattr(args, "partition", "dirichlet") == "writer":
        if args.dataset != "femnist":
            raise ValueError("Writer partitioning is only defined for FEMNIST")
        if args.clients != len(train.users):
            raise ValueError(
                f"Writer partitioning needs --clients {len(train.users)} (one per training writer)"
            )
        subsets = train.user_indices
    else:
        subsets = partition(targets, args.clients, args.alpha, args.seed)
    return train, test, subsets, targets, classes


def loader(dataset, indices, args, shuffle=True):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
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
        {"accuracy": correct / total, "loss": loss / total}
        if total
        else {"accuracy": None, "loss": None}
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=["synthetic", "cifar10", "cifar100", "imagefolder"],
        default="synthetic",
    )
    p.add_argument("--model", choices=['resnet20', 'convnext', 'convmixer256_8', 'mobilenet_v2'], default="resnet20")
    p.add_argument("--clients", type=int, default=2)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--samples-per-client", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--participation", type=float, default=1.0)
    p.add_argument("--kappa", type=int, default=4)
    p.add_argument("--precision", type=int, default=2)
    p.add_argument("--max-abs", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument(
        "--output",
        default=None,
        help="Write configuration, partition indices and measured metrics as JSON",
    )
    p.add_argument(
        "--method",
        choices=["enccluster", "bfuse", "cluster", "fedavg"],
        default="enccluster",
    )
    p.add_argument("--bpe", type=int, choices=[8, 16], default=8)
    args = p.parse_args(argv)
    if args.output:
        destination = Path(args.output)
        if destination.suffix != ".json":
            p.error(
                "--output must end in .json; a matching .pt checkpoint is saved beside it"
            )
        if destination.exists() or destination.with_suffix(".pt").exists():
            raise FileExistsError(
                f"Output already exists: {destination}; choose a new name"
            )
    if (
        args.clients < 2
        or args.rounds < 1
        or min(args.batch_size, args.local_epochs, args.samples_per_client, args.kappa)
        < 1
    ):
        p.error("Positive sizes/rounds and at least two clients are required")
    if (
        not 0 < args.participation <= 1
        or math.floor(args.clients * args.participation) < 2
    ):
        p.error("Participation must select at least two clients")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    train, test, partitions, targets, classes = load_data(args)
    model = ImageModel(args.model, classes).to(args.device)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    history = []
    rng = np.random.default_rng(args.seed)
    for round_id in range(1, args.rounds + 1):
        active = list(range(args.clients))
        count = math.floor(len(active) * args.participation)
        if count < 2:
            raise ValueError("At least two active participants must remain")
        selected = sorted(rng.choice(active, size=count, replace=False).tolist())
        states = []
        sizes = []
        for cid in selected:
            local = copy.deepcopy(model)
            indices = partitions[cid]
            local = fit(local, loader(train, indices, args), args.local_epochs, args.lr)
            sizes.append(len(indices))
            states.append(
                {k: v.detach().cpu().clone() for k, v in local.state_dict().items()}
            )
        from enccluster.methods import METHODS

        updated = METHODS[args.method].aggregate(
            states,
            sizes,
            kappa=args.kappa,
            bpe=args.bpe,
            seed=args.seed,
            precision=args.precision,
            max_abs=args.max_abs,
        )
        full = model.state_dict()
        full.update(
            {
                k: torch.as_tensor(v, dtype=full[k].dtype, device=args.device)
                for k, v in updated.items()
            }
        )
        model.load_state_dict(full, strict=True)
        metrics = {
            "round": round_id,
            "participants": selected,
            "sample_counts": sizes,
            **evaluate(model, test_loader),
        }
        history.append(metrics)
        print(json.dumps(metrics))
    result = {
        "project": "EncCluster",
        "configuration": vars(args),
        "partitions": partitions,
        "history": history,
        "protocol": (
            "decentralized-coordinate-fe"
            if args.method == "enccluster"
            else "plaintext-baseline"
        ),
        "benchmark_reproduction": False,
    }
    from .dataset_io import manifest_digest

    result["data_manifest_sha256"] = manifest_digest(args.dataset, args.data_root)
    result["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        torch.save(
            {
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "configuration": vars(args),
                "classes": classes,
            },
            path.with_suffix(".pt"),
        )
    return result


def cli():
    main()


if __name__ == "__main__":
    cli()
