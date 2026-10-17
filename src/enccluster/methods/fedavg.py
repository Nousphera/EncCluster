"""Standard sample-weighted FedAvg, including floating BatchNorm buffers."""

import numpy as np
from enccluster.clustering import validate_states


def aggregate(states, sizes, **_):
    total = validate_states(states, sizes)
    result = {}
    for key in states[0]:
        values = [state[key].detach().cpu().numpy() for state in states]
        if key.endswith("num_batches_tracked"):
            result[key] = np.maximum.reduce(values)
        else:
            result[key] = sum(
                value.astype(np.float64) * (n / total)
                for value, n in zip(values, sizes)
            )
    return result
