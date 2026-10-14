# Provenance and licensing

## Recovery sources

- Authors' public implementation: [SamaneMohammadi/EncCluster](https://github.com/SamaneMohammadi/EncCluster/tree/9a1b4e00503813a69ecbe84bea564a087cf33ca3), commit `9a1b4e00503813a69ecbe84bea564a087cf33ca3`.
- Maintained decentralized primitive: [MechFroG88/PyMIFE](https://github.com/MechFroG88/PyMIFE/tree/69938282ed17fd2f1ddb11102c1480ad6dc5c04b), commit `69938282ed17fd2f1ddb11102c1480ad6dc5c04b`, package version 0.0.14. Its complete source snapshot and original MIT license are in `modules/mife`.
- The existing Nousphera history, `docs/` project website and paper citation are preserved. Full recovery snapshots remain separately archived during preparation; this release contains the selected executable source, not generated build artifacts, binaries, checkpoints or data.

`SOURCE_MANIFEST.json` records upstream paths, original hashes and released hashes.
Source moved into the `enccluster` Python namespace. Modified imported files carry
notices; new harness/protocol/test/documentation files are maintenance additions.

## Changes from the recovered implementation

- Replaced central-master-key / repeated centroid-label wrappers with the existing decentralized PyMIFE primitive and unique coordinate labels.
- Added an explicit numeric/shape/participant contract, sample-weighted verification and a no-download seeded training path.
- Added Python package metadata, constraints, tests, provenance and reproducibility documentation.
- Removed nonfunctional launchers and machine-specific data paths from the released tree; source commits above preserve their history.
- Fixed process-independent filter prehashing and native build metadata; removed copied Linux binary modules.

## License records and review item

The existing repository root license is Apache-2.0. Imported EncCluster source
is covered by its upstream MIT text (`licenses/UPSTREAM-LICENSE`, copyright 2025
The EncCluster authors); that notice also accompanies the Python package.
The older bundled mife snapshot supplied Apache text while declaring MIT
metadata. It is replaced by canonical MIT PyMIFE (copyright 2023 Tan Kel Zin),
and its old text remains in `licenses/MIFE-SNAPSHOT-LICENSE` for provenance.

The pyfilters wrapper declares **Apache 2.0** and Vasileios Tsouvalas authorship in
its original `setup.py`. Neither the recovered public snapshot nor the matching
Drive source contains a standalone wrapper license file. This is a remaining
license-provenance review item; no new grant is inferred here. The original
native headers derive from [FastFilter/xor_singleheader](https://github.com/FastFilter/xor_singleheader)
and its Apache-2.0 license is preserved in `licenses/XOR-SINGLEHEADER-LICENSE`
and beside the headers. The Drive and public header SHA-256 hashes match exactly:

- `binaryfusefilter.h`: `024c950be375a6f0164928e3a601bc41612ae9d84f760da40f29f9f034dbb968`
- `xorfilter.h`: `2fcb816b53d1813347c75c3bd3b3b0e691b8f69f20876803fc4b290492d26ffc`

The wrapper's internal VERSION (0.2.0) differs from setup.py's package version
(1.0.0); installation follows setup.py. Drive identifiers/private source paths
are not included in this public package.

## Earlier repository provenance

This repository is an independent Nousphera copy of [EncCluster](https://github.com/FederatedML/EncCluster). The original FederatedML repository remains in place.

- Source commit: [`f07235b81184d3e75d620662be2d0b5673a3354f`](https://github.com/FederatedML/EncCluster/commit/f07235b81184d3e75d620662be2d0b5673a3354f)
- Source branch: `main`
- Baseline tag in this repository: `federatedml-baseline`
- Original Git history and author attribution are retained.

Maintenance changes after this baseline improve documentation, setup, and validation. They do not constitute a new reproduction of the published experiments. For the original code and configuration, inspect the baseline tag.

GitHub issues, pull requests, stars, releases, and Pages configuration are not part of Git history and have not been transferred. Existing discussions remain at the source repository.

## Dataset/setup follow-up (2026-10-15)

The maintained release adds `scripts/setup.sh`, `scripts/prepare_data.py`, namespaced dataset preparation/validation, manifests and generated-source integration tests. These modules are new integration code rather than recovered historical experiment scripts. Official dataset loading remains delegated to torchvision; fixed-split imports remain user-supplied. See `DATASETS.md` for split conventions and `REPRODUCIBILITY.md` for method/benchmark limits.
