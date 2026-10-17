import numpy as np
import pytest
from enccluster.filters import (
    encode_mapping,
    serialize_filter,
    deserialize_filter,
    reconstruct_mapping,
)


@pytest.mark.parametrize("bits", [8, 16])
def test_portable_filter_matches_native(bits):
    mapping = np.random.default_rng(8).integers(0, 12, 8192)
    native = encode_mapping(mapping, bits)
    wire = serialize_filter(native, bits)
    restored = deserialize_filter(wire, dimension=len(mapping))
    for i in range(8192):
        for c in (0, int(mapping[i]), 11):
            assert restored.contains((i, c)) == native.contains((i, c))
    np.testing.assert_array_equal(
        reconstruct_mapping(native, len(mapping), 12),
        reconstruct_mapping(restored, len(mapping), 12),
    )
    for bad in (b"", wire[:-1], wire + b"0", wire[:4] + b"\xff" * 24 + wire[28:]):
        with pytest.raises(ValueError):
            deserialize_filter(bad, dimension=len(mapping))


def test_no_match_fails_instead_of_silent_zero():
    class Missing:
        def contains(self, key):
            return False

    with pytest.raises(ValueError, match="match"):
        reconstruct_mapping(Missing(), 5, 2)
