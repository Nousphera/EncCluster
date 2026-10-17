import numpy as np
import pytest
import torch


def test_weighted_fedavg_preserves_bn_statistics_and_max_counter():
    from enccluster.methods.fedavg import aggregate

    states = [
        dict(
            weight=torch.tensor([2.0, 4.0]),
            running_mean=torch.tensor([4.0]),
            num_batches_tracked=torch.tensor(2),
        ),
        dict(
            weight=torch.tensor([6.0, 8.0]),
            running_mean=torch.tensor([8.0]),
            num_batches_tracked=torch.tensor(5),
        ),
    ]
    result = aggregate(states, [1, 3])
    np.testing.assert_allclose(result["weight"], [5.0, 7.0])
    np.testing.assert_allclose(result["running_mean"], [7.0])
    assert result["num_batches_tracked"] == 5


@pytest.mark.parametrize("sizes", [[0, 1], [1], [1, float("nan")], [1, -1]])
def test_plaintext_aggregation_rejects_invalid_sample_counts(sizes):
    from enccluster.methods.fedavg import aggregate

    states = [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([3.0])}]
    with pytest.raises(ValueError):
        aggregate(states, sizes)


def test_aggregation_rejects_mismatched_state_layout():
    from enccluster.methods.fedavg import aggregate

    with pytest.raises(ValueError):
        aggregate([{"weight": torch.ones(2)}, {"weight": torch.ones(3)}], [1, 1])
