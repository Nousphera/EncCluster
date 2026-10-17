"""Binary Fuse mappings with validated, portable little-endian transport."""

from dataclasses import dataclass
import hashlib
import struct
import numpy as np
from pyfilters.pyfilters import Fuse8, Fuse16
from enccluster import config

_HEADER = struct.Struct("<4sBQIIIII")
_MASK = (1 << 64) - 1


def encode_mapping(mapping, bpe=config.BPE):
    mapping = np.asarray(mapping)
    if bpe not in (8, 16):
        raise ValueError("Binary Fuse fingerprint width must be 8 or 16")
    if (
        mapping.ndim != 1
        or not len(mapping)
        or not np.issubdtype(mapping.dtype, np.integer)
        or np.any(mapping < 0)
    ):
        raise ValueError("Mapping must contain nonnegative integer cluster IDs")
    filt = (Fuse16 if bpe == 16 else Fuse8)(max(2, len(mapping)))
    if not filt.populate([(i, int(c)) for i, c in enumerate(mapping)]):
        raise ValueError("Binary Fuse filter construction failed")
    return filt


def reconstruct_mapping(filt, dimension, kappa):
    if type(dimension) != int or dimension < 1 or type(kappa) != int or kappa < 1:
        raise ValueError("Positive dimension and kappa required")
    mapping = np.empty(dimension, dtype=np.int64)
    for i in range(dimension):
        for c in range(kappa):
            if filt.contains((i, c)):
                mapping[i] = c
                break
        else:
            raise ValueError(f"No matching centroid for coordinate {i}")
    return mapping


@dataclass(frozen=True)
class PortableFilter:
    bits: int
    seed: int
    segment: int
    mask: int
    count: int
    count_length: int
    fingerprints: np.ndarray

    def contains(self, item):
        i, c = item
        key = int.from_bytes(
            hashlib.blake2b(
                f"[{int(i)},{int(c)}]".encode(), digest_size=8, person=b"enccluster-bf"
            ).digest(),
            "little",
        )
        h = (key + self.seed) & _MASK
        h = ((h ^ (h >> 33)) * 0xFF51AFD7ED558CCD) & _MASK
        h = ((h ^ (h >> 33)) * 0xC4CEB9FE1A85EC53) & _MASK
        h ^= h >> 33
        a = (h * self.count_length) >> 64
        b = (a + self.segment) ^ ((h >> 18) & self.mask)
        c = (a + 2 * self.segment) ^ (h & self.mask)
        fingerprint = (h ^ (h >> 32)) & ((1 << self.bits) - 1)
        return fingerprint == int(
            self.fingerprints[a] ^ self.fingerprints[b] ^ self.fingerprints[c]
        )


def serialize_filter(filt, bpe):
    if bpe not in (8, 16):
        raise ValueError("Invalid fingerprint width")
    raw = bytes(filt.serialize())
    seed, segment, mask, count, count_length, length = struct.unpack(
        "=QIIIII", raw[:28]
    )
    # Only three queried segments belong on the wire. Older allocators reserve
    # one unused segment, which must never expose uninitialized native memory.
    used = count_length + 2 * segment
    values = (seed, segment, mask, count, count_length, used)
    dtype = np.dtype(f"u{bpe // 8}").newbyteorder("=")
    fp = (
        np.frombuffer(raw[28 : 28 + used * (bpe // 8)], dtype=dtype)
        .astype(f"<u{bpe // 8}", copy=False)
        .tobytes()
    )
    return _HEADER.pack(b"BF01", bpe, *values) + fp


def deserialize_filter(wire, dimension):
    if (
        not isinstance(wire, bytes)
        or len(wire) < _HEADER.size
        or type(dimension) != int
        or dimension < 1
    ):
        raise ValueError("Invalid Binary Fuse payload")
    magic, bits, seed, segment, mask, count, count_length, length = _HEADER.unpack_from(
        wire
    )
    if (
        magic != b"BF01"
        or bits not in (8, 16)
        or not 1 <= segment <= 262144
        or segment & (segment - 1)
        or mask != segment - 1
        or count < 1
        or count_length != segment * count
        or length != count_length + 2 * segment
        or length > max(1024, dimension * 8)
        or len(wire) != _HEADER.size + length * (bits // 8)
    ):
        raise ValueError("Invalid Binary Fuse header or payload size")
    fingerprints = np.frombuffer(wire, dtype=f"<u{bits // 8}", offset=_HEADER.size)
    return PortableFilter(bits, seed, segment, mask, count, count_length, fingerprints)
