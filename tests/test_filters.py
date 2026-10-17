import subprocess
import sys
import numpy as np
from enccluster.filters import encode_mapping, reconstruct_mapping


def test_membership_and_mapping_roundtrip():
    mapping = np.arange(80) % 4
    for bits in (8, 16):
        bf = encode_mapping(mapping, bpe=bits)
        assert all(bf.contains((i, int(c))) for i, c in enumerate(mapping))
        estimated = reconstruct_mapping(bf, len(mapping), 4)
        assert len(estimated) == len(mapping)
        assert np.all((estimated >= 0) & (estimated < 4))


def test_string_hashing_is_process_independent(tmp_path):
    target = tmp_path / "filter.bin"
    create = 'from pyfilters.pyfilters import Fuse16; f=Fuse16(20); f.populate([f"x_{i}" for i in range(20)]); open(__import__("sys").argv[1],"wb").write(f.serialize())'
    check = 'from pyfilters.pyfilters import Fuse16; f=Fuse16(20); f.deserialize(bytearray(open(__import__("sys").argv[1],"rb").read())); assert all(f.contains(f"x_{i}") for i in range(20))'
    subprocess.run(
        [sys.executable, "-c", create, str(target)],
        check=True,
        env={"PYTHONHASHSEED": "1"},
    )
    subprocess.run(
        [sys.executable, "-c", check, str(target)],
        check=True,
        env={"PYTHONHASHSEED": "2"},
    )
