"""Centroid encryption and filter reconstruction for local numerical runs."""

import hashlib
import json
from enccluster.clustering import (
    weight_clustering,
    unflatten_state,
    aggregate_buffers,
    validate_states,
)
from enccluster.filters import (
    encode_mapping,
    reconstruct_mapping,
    serialize_filter,
    deserialize_filter,
)
from enccluster.centroid_fe import Config, Participant, Aggregator


def aggregate(
    states, sizes, kappa=128, bpe=8, seed=42, precision=3, max_abs=8.0, round_id=1, **_
):
    validate_states(states, sizes)
    clustered = [weight_clustering(state, kappa=kappa, seed=seed) for state in states]
    keys, shapes = clustered[0][2:]
    dimension = len(clustered[0][1])
    layout = hashlib.sha256(
        json.dumps([keys, shapes], separators=(",", ":")).encode()
    ).hexdigest()
    cfg = Config.create(
        sizes,
        dimension=dimension,
        kappa=kappa,
        precision=precision,
        max_abs=max_abs,
        round_id=round_id,
        layout_digest=layout,
    )
    clients = [Participant(cfg, i) for i in range(len(states))]
    peers = {i: client.public_key() for i, client in enumerate(clients)}
    messages = []
    mappings = []
    for client, (centroids, mapping, _, _) in zip(clients, clustered):
        client.set_peers(peers)
        wire = serialize_filter(encode_mapping(mapping, bpe), bpe)
        recovered = reconstruct_mapping(
            deserialize_filter(wire, dimension), dimension, len(centroids)
        )
        mappings.append(recovered)
        messages.append(client.encrypt(centroids, recovered))
    server = Aggregator(cfg, peers)
    chunks = (
        (request, [client.authorize(request) for client in clients])
        for request in server.request_chunks(messages, mappings)
    )
    flat = server.aggregate_chunks(messages, chunks)
    out = unflatten_state(flat, keys, shapes)
    out.update(aggregate_buffers(states, sizes))
    return out
