"""Deterministic full-model clustering and sample-weighted state aggregation."""

import numbers
import numpy as np
import torch
from sklearn.cluster import KMeans
from enccluster import config


def validate_states(states, sizes):
    if not states or len(states) != len(sizes):
        raise ValueError("Every nonempty client state needs exactly one sample count")
    if any(
        isinstance(n, bool) or not isinstance(n, numbers.Integral) or n <= 0
        for n in sizes
    ):
        raise ValueError("Sample counts must be positive integers")
    layout = [(k, tuple(v.shape), v.dtype) for k, v in states[0].items()]
    if not layout:
        raise ValueError("Client states cannot be empty")
    for state in states:
        if [(k, tuple(v.shape), v.dtype) for k, v in state.items()] != layout:
            raise ValueError("Clients must use identical state keys, shapes and dtypes")
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Client states must contain finite tensors")
    return sum(sizes)


def flatten_state(state_dict):
    keys, shapes, chunks = [], [], []
    for key, value in state_dict.items():
        if value.is_floating_point():
            keys.append(key)
            shapes.append(tuple(value.shape))
            chunks.append(value.detach().cpu().numpy().ravel())
    if not chunks:
        raise ValueError("No model parameters to cluster")
    vector = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Cannot cluster nonfinite model parameters")
    return vector, keys, shapes


def unflatten_state(vector, keys, shapes):
    out, offset = {}, 0
    if len(keys) != len(shapes) or len(set(keys)) != len(keys):
        raise ValueError("Invalid model parameter layout")
    for key, shape in zip(keys, shapes):
        count = int(np.prod(shape)) if shape else 1
        out[key] = np.asarray(vector[offset : offset + count]).reshape(shape)
        offset += count
    if offset != len(vector):
        raise ValueError("Vector length differs from model parameter layout")
    return out


def weight_clustering(state_dict, kappa=config.KAPPA, seed=config.SEED):
    vector, keys, shapes = flatten_state(state_dict)
    if isinstance(kappa, bool) or not isinstance(kappa, numbers.Integral) or kappa < 1:
        raise ValueError("kappa must be a positive integer")
    clusters = min(kappa, len(np.unique(vector)))
    km = KMeans(n_clusters=clusters, n_init=4, random_state=seed)
    mapping = km.fit_predict(vector.reshape(-1, 1))
    return (
        km.cluster_centers_.ravel().astype(np.float64),
        mapping.astype(np.int64),
        keys,
        shapes,
    )


def expand(centroids, mapping, keys, shapes):
    return unflatten_state(np.asarray(centroids)[np.asarray(mapping)], keys, shapes)


def aggregate_buffers(states, sizes):
    total = validate_states(states, sizes)
    out = {}
    for key in states[0]:
        if not states[0][key].is_floating_point():
            values = [state[key].detach().cpu().numpy() for state in states]
            if key.endswith("num_batches_tracked"):
                out[key] = np.maximum.reduce(values)
            else:
                out[key] = sum(
                    value.astype(np.float64) * (n / total)
                    for value, n in zip(values, sizes)
                )
    return out
