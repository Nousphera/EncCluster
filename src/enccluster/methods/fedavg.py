import numpy as np


def aggregate(states, sizes, **_):
    total = sum(sizes)
    keys = [k for k in states[0] if "num_batches_tracked" not in k]
    return {k: sum(s[k].cpu().numpy().astype(np.float64) * (w / total)
                   for s, w in zip(states, sizes)) for k in keys}
