"""FedAvg with full local model weight clustering; plaintext ablation."""

from enccluster.clustering import (
    weight_clustering,
    unflatten_state,
    aggregate_buffers,
    validate_states,
)


def aggregate(states, sizes, kappa=128, seed=42, **_):
    total = validate_states(states, sizes)
    expanded = []
    for state in states:
        centroids, mapping, keys, shapes = weight_clustering(
            state, kappa=kappa, seed=seed
        )
        expanded.append(centroids[mapping])
    result = unflatten_state(
        sum(v * (n / total) for v, n in zip(expanded, sizes)), keys, shapes
    )
    result.update(aggregate_buffers(states, sizes))
    return result
