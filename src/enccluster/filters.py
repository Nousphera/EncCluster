# Modified 2026-10-15: validate filter width and successful population.
import numpy as np
from pyfilters.pyfilters import Fuse8, Fuse16

from enccluster import config
def encode_mapping(P, bpe=config.BPE):
    """Insert {(i, P[i])} into a BF filter (its fingerprints are transmitted)."""
    if bpe not in (8,16):
        raise ValueError("Binary Fuse fingerprint width must be 8 or 16")
    keys = [(int(i), int(c)) for i, c in enumerate(P)]
    f = (Fuse16 if bpe >= 16 else Fuse8)(len(keys))
    if not keys or not f.populate(keys):
        raise ValueError("Binary Fuse filter construction failed or received an empty mapping")
    return f


def reconstruct_mapping(bf, d, kappa):
    """Recover P'[i] = c such that Member((i, c)) for every weight (Eq. 3)."""
    P = np.zeros(d, dtype=np.int64)
    for i in range(d):
        for c in range(kappa):
            if bf.contains((i, c)):
                P[i] = c; break
    return P
