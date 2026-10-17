import numpy as np
import torch
from enccluster.methods.enccluster import aggregate
from enccluster.clustering import weight_clustering
from enccluster.filters import encode_mapping, reconstruct_mapping


def test_unequal_weight_encrypted_reconstruction_including_bn():
    states = [
        {
            "weight": torch.tensor([-0.3, 0.2, 0.8, 0.8]),
            "running_mean": torch.tensor([0.1, 0.2]),
            "num_batches_tracked": torch.tensor(2),
        },
        {
            "weight": torch.tensor([0.1, 0.4, 0.9, 0.9]),
            "running_mean": torch.tensor([0.2, 0.3]),
            "num_batches_tracked": torch.tensor(4),
        },
    ]
    sizes = [2, 5]
    expected = []
    for state in states:
        z, p, keys, shapes = weight_clustering(state, kappa=3, seed=9)
        p = reconstruct_mapping(encode_mapping(p, 16), len(p), len(z))
        expected.append(np.rint(z * 1000)[p] / 1000)
    actual = aggregate(states, sizes, kappa=3, bpe=16, seed=9, precision=3)
    vector = np.concatenate([np.asarray(actual[k]).reshape(-1) for k in keys])
    np.testing.assert_allclose(
        vector, np.average(expected, axis=0, weights=sizes), atol=1e-7
    )
    assert actual["num_batches_tracked"] == 4
