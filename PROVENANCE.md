# Attribution and licenses

## Source attribution

- EncCluster implementation: [SamaneMohammadi/EncCluster](https://github.com/SamaneMohammadi/EncCluster/tree/9a1b4e00503813a69ecbe84bea564a087cf33ca3), commit `9a1b4e00503813a69ecbe84bea564a087cf33ca3`.
- Decentralized FE primitive: [MechFroG88/PyMIFE](https://github.com/MechFroG88/PyMIFE/tree/69938282ed17fd2f1ddb11102c1480ad6dc5c04b), commit `69938282ed17fd2f1ddb11102c1480ad6dc5c04b`, package 0.0.14. Source and MIT license are included in `modules/mife`.
- Binary Fuse and XOR headers: [FastFilter/xor_singleheader](https://github.com/FastFilter/xor_singleheader), Apache-2.0. The license is included beside the headers and in `licenses/XOR-SINGLEHEADER-LICENSE`.
- Repository history includes [FederatedML/EncCluster](https://github.com/FederatedML/EncCluster), commit `f07235b81184d3e75d620662be2d0b5673a3354f`, identified by the `federatedml-baseline` tag.

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) identifies upstream file paths and hashes. Imported files retain attribution and modification notices. The `enccluster` package integrates dataset preparation, local training, stable filter hashing and coordinate-labelled FE; see [implementation details](REPRODUCIBILITY.md).

## License records

The repository root [LICENSE](LICENSE) is Apache-2.0. Imported EncCluster source retains its upstream MIT notice, copyright 2025 the EncCluster authors, in [licenses/UPSTREAM-LICENSE](licenses/UPSTREAM-LICENSE) and the Python package. Canonical PyMIFE is MIT, copyright 2023 Tan Kel Zin. An earlier MIFE snapshot's Apache text is retained in `licenses/MIFE-SNAPSHOT-LICENSE` as a source record.

The pyfilters wrapper metadata names Vasileios Tsouvalas and declares Apache 2.0; no separate wrapper license file accompanies it. The native headers have their own Apache-2.0 license. The wrapper package version is 1.0.0 (`setup.py`); its internal `VERSION` is 0.2.0.

Original native-header SHA256 values:

| File | SHA256 |
|---|---|
| `binaryfusefilter.h` | `024c950be375a6f0164928e3a601bc41612ae9d84f760da40f29f9f034dbb968` |
| `xorfilter.h` | `2fcb816b53d1813347c75c3bd3b3b0e691b8f69f20876803fc4b290492d26ffc` |

[NOTICE](NOTICE) collects component attribution. Dataset access terms are separate from software licenses.
