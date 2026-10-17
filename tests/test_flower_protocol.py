import argparse
import json
import numpy as np
import pytest
import torch
from flwr.app import Context, RecordDict
from enccluster import flower
from flwr.supercore.task_identity import TaskIdentity


@pytest.fixture(autouse=True)
def task_identity(monkeypatch):
    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)


class Experiment:
    def __init__(self, args):
        self.args = args
        self.model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()

    def participants(self, rnd):
        return [0, 1]

    def client_update(self, cid, state, rnd, cache, **kw):
        cache["seen"] = cache.get("seen", 0) + 1
        assert cache["seen"] == rnd
        return (
            {k: v + (cid + 1) * 0.1 for k, v in state.items()},
            cid + 1,
            {"loss": 1.0},
        )

    def evaluate(self, state):
        self.model.load_state_dict(state)
        return {"accuracy": 0.5, "samples": 4}

    def aggregate_plaintext(self, states, counts, *args, **kw):
        return {
            k: sum(s[k] * n / sum(counts) for s, n in zip(states, counts))
            for k in states[0]
        }

    def result(self, history):
        return {"history": history}


class Grid:
    def __init__(self, args):
        self.args = args
        self.contexts = {
            n: Context(
                1,
                n,
                {"partition-id": n - 10},
                RecordDict(),
                {"arguments": json.dumps(vars(args))},
            )
            for n in [10, 11]
        }
        self.wire = []

    def get_node_ids(self):
        return [10, 11]

    def send_and_receive(self, messages, timeout=None):
        replies = []
        for message in messages:
            context = self.contexts[message.metadata.dst_node_id]
            TaskIdentity._node_id = context.node_id
            try:
                reply = flower.handle(
                    message, context, experiment=Experiment(self.args)
                )
            finally:
                TaskIdentity._node_id = 0
            self.wire.append((message, reply))
            replies.append(reply)
        return replies


def arguments(method):
    return argparse.Namespace(
        method=method,
        clients=2,
        rounds=2,
        kappa=2,
        bpe=16,
        precision=3,
        max_abs=8.0,
        seed=42,
        reply_timeout=30.0,
        registration_timeout=1.0,
        torch_threads=1,
    )


@pytest.mark.parametrize("method", ["enccluster", "fedavg"])
def test_all_phases_and_private_client_cache(method):
    args = arguments(method)
    grid = Grid(args)
    exp = Experiment(args)
    result = flower.run_server(grid, args, experiment=exp)
    assert len(result["history"]) == 2
    np.testing.assert_allclose(
        exp.model.weight.detach().numpy(),
        np.full((2, 2), (267 + 2 * 367) / 3000 if method == "enccluster" else 1 / 3),
        atol=1e-7,
    )
    if method == "enccluster":
        for message, reply in grid.wire:
            assert not reply.content.array_records, (
                "Encrypted clients must not return plaintext arrays"
            )
            assert "private" not in reply.content
        assert result["history"][0]["communication"]["ciphertext_points"] == 4
    assert len(grid.contexts[10].state) > 0


def test_missing_reply_rejected():
    args = arguments("fedavg")
    grid = Grid(args)
    original = grid.send_and_receive
    grid.send_and_receive = lambda *a, **k: original(*a, **k)[:1]
    with pytest.raises(RuntimeError, match="reply|replies|cohort"):
        flower.run_server(grid, args, experiment=Experiment(args))


def test_chunked_authorization_keeps_state_until_complete(monkeypatch):
    original = flower.Aggregator.request_chunks
    monkeypatch.setattr(
        flower.Aggregator,
        "request_chunks",
        lambda self, messages, maps: original(self, messages, maps, chunk_size=2),
    )
    args = arguments("enccluster")
    exp = Experiment(args)
    grid = Grid(args)
    flower.run_server(grid, args, experiment=exp)
    phases = [m.content["config"]["phase"] for m, _ in grid.wire]
    assert phases.count("authorize") > args.clients * args.rounds
    assert phases.count("commit") == args.clients * args.rounds
    for context in grid.contexts.values():
        private = flower._load(context)
        assert private["phase"] == "committed"
        assert private["last_round"] == args.rounds
        assert "participant" not in private
    assert all(not reply.content.array_records for _, reply in grid.wire)


def test_bad_integer_counters_do_not_commit_clients():
    args = arguments("enccluster")
    grid = Grid(args)
    original = grid.send_and_receive

    def corrupt(messages, timeout=None):
        messages = list(messages)
        replies = original(messages, timeout)
        if messages[0].content["config"]["phase"] == "encrypt":
            replies[0].content["meta"]["counters"] = "bad json"
        return replies

    grid.send_and_receive = corrupt
    with pytest.raises(ValueError):
        flower.run_server(grid, args, experiment=Experiment(args))
    for context in grid.contexts.values():
        private = flower._load(context)
        assert private["phase"] == "authorizing"
        assert private["last_round"] == 0
        assert "participant" in private
