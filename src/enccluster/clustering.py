# Modified 2026-10-15: namespaced release integration; see PROVENANCE.md.
import numpy as np
from sklearn.cluster import KMeans

from enccluster import config
_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def _is_weight(name):
    return not any(b in name for b in _BUFFERS)


def flatten_state(state_dict):
    keys, shapes, chunks = [], [], []
    for k, v in state_dict.items():
        if _is_weight(k):
            keys.append(k); shapes.append(tuple(v.shape))
            chunks.append(v.detach().cpu().numpy().ravel())
    return np.concatenate(chunks).astype(np.float64), keys, shapes


def unflatten_state(vec, keys, shapes):
    out, i = {}, 0
    for k, shape in zip(keys, shapes):
        n = int(np.prod(shape)) if shape else 1
        out[k] = vec[i:i + n].reshape(shape); i += n
    return out


def weight_clustering(state_dict, kappa=config.KAPPA, seed=config.SEED):
    """Return (centroids Z[kappa], mapping P[d], weight keys, shapes)."""
    vec, keys, shapes = flatten_state(state_dict)
    km = KMeans(n_clusters=kappa, n_init=4, random_state=seed)
    P = km.fit_predict(vec.reshape(-1, 1))
    return km.cluster_centers_.ravel().astype(np.float64), P.astype(np.int64), keys, shapes


def expand(Z, P, keys, shapes):
    return unflatten_state(np.asarray(Z)[np.asarray(P)], keys, shapes)


def aggregate_buffers(states, sizes):
    """Plaintext weighted average of the non-weight buffers (BN running stats)."""
    total = sum(sizes)
    buf_keys = [k for k in states[0] if not _is_weight(k)]
    out = {}
    for k in buf_keys:
        if "num_batches_tracked" in k:
            out[k] = states[0][k]                      # integer counter: just keep one
        else:
            out[k] = sum(s[k].cpu().numpy().astype(np.float64) * (w / total)
                         for s, w in zip(states, sizes))
    return out
