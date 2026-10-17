"""Reusable experiment semantics for local and native Flower transports."""

import copy
import hashlib
import json
import platform
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .train import ImageModel, load_data, loader, fit, evaluate, validate_args
from .dataset_io import manifest_digest


class Experiment:
    def __init__(self, args):
        self.args = validate_args(args)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.train, self.test, self.partitions, self.targets, self.classes = load_data(
            args
        )
        if args.model == "mobilenet_v2" and any(
            len(part) < 2 for part in self.partitions
        ):
            raise ValueError(
                "MobileNet requires at least two samples per client; use more data, fewer clients or another partition seed"
            )
        self.model = ImageModel(args.model, self.classes).to(args.device)
        self.test_loader = DataLoader(
            self.test, batch_size=args.batch_size, num_workers=0
        )
        self.data_hash = manifest_digest(args.dataset, args.data_root)
        self.partition_hash = hashlib.sha256(
            json.dumps(self.partitions, separators=(",", ":")).encode()
        ).hexdigest()

    def participants(self, round_id):
        rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, round_id]))
        count = int(np.floor(self.args.clients * self.args.participation))
        return sorted(rng.choice(self.args.clients, size=count, replace=False).tolist())

    def _load_state(self, model, state):
        reference = model.state_dict()
        if set(state) != set(reference):
            raise ValueError("Global state keys differ from the model")
        converted = {
            key: torch.as_tensor(state[key], dtype=value.dtype, device=value.device)
            for key, value in reference.items()
        }
        if any(not torch.isfinite(value).all() for value in converted.values()):
            raise ValueError("Global state contains nonfinite values")
        model.load_state_dict(converted, strict=True)

    def client_update(self, cid, global_state, round_id, cache):
        if cid not in range(self.args.clients):
            raise ValueError("Unknown client")
        seed = int(
            np.random.SeedSequence([self.args.seed, round_id, cid]).generate_state(1)[0]
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        local = copy.deepcopy(self.model)
        self._load_state(local, global_state)
        indices = self.partitions[cid]
        fit(
            local,
            loader(self.train, indices, self.args),
            self.args.local_epochs,
            self.args.lr,
        )
        state = {
            key: value.detach().cpu().clone()
            for key, value in local.state_dict().items()
        }
        return state, len(indices), {"local_epochs": self.args.local_epochs}

    def aggregate_plaintext(self, states, counts, client_ids, global_state, round_id):
        if len(client_ids) != len(states) or len(set(client_ids)) != len(client_ids):
            raise ValueError("Plaintext cohort identities differ from client states")
        from .methods import METHODS

        if self.args.method == "enccluster":
            raise ValueError(
                "Encrypted EncCluster requires its secure aggregation transport"
            )
        return METHODS[self.args.method].aggregate(
            states,
            counts,
            kappa=self.args.kappa,
            bpe=self.args.bpe,
            seed=self.args.seed,
        )

    def evaluate(self, state):
        self._load_state(self.model, state)
        return evaluate(self.model, self.test_loader)

    def result(self, history):
        return {
            "project": "EncCluster",
            "configuration": vars(self.args),
            "partitions": self.partitions,
            "partition_sha256": self.partition_hash,
            "data_manifest_sha256": self.data_hash,
            "history": history,
            "protocol": "encrypted"
            if self.args.method == "enccluster"
            else "plaintext-baseline",
            "benchmark_reproduction": False,
            "model": {
                "architecture": self.args.model,
                "pretrained": False,
                "parameters": sum(p.numel() for p in self.model.parameters()),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
            },
        }
