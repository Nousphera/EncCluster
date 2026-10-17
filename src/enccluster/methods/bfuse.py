"""Plaintext clustering plus Binary Fuse mapping ablation."""

from enccluster.clustering import (
    weight_clustering,
    unflatten_state,
    aggregate_buffers,
    validate_states,
)
from enccluster.filters import encode_mapping, reconstruct_mapping


def aggregate(states, sizes, kappa=128, bpe=8, seed=42, **_):
    total = validate_states(states, sizes)
    expanded = []
    for state in states:
        centroids, mapping, keys, shapes = weight_clustering(
            state, kappa=kappa, seed=seed
        )
        reconstructed = reconstruct_mapping(
            encode_mapping(mapping, bpe=bpe), len(mapping), len(centroids)
        )
        expanded.append(centroids[reconstructed])
    result = unflatten_state(
        sum(v * (n / total) for v, n in zip(expanded, sizes)), keys, shapes
    )
    result.update(aggregate_buffers(states, sizes))
    return result
