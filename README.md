# EncCluster

**Scalable Functional Encryption in Federated Learning through Weight Clustering and Probabilistic Filters**

[Paper](https://doi.org/10.1016/j.pmcj.2025.102021) · [Project](https://nousphera.github.io/EncCluster/) · [Citation](CITATION.cff) · [Provenance](PROVENANCE.md)

Federated training with K-means weight clustering, native Binary Fuse filters and encrypted weighted aggregation.

This repository recovers the authors' research code and provides a tested local
simulation. The default encryption path uses decentralized, coordinate-labelled
FE to avoid defects in the recovered compressed wrapper. **It does not reproduce
the paper's kappa-ciphertext cost or establish its security claims.** See
[code-to-paper mapping and limits](REPRODUCIBILITY.md) before using results.

## Install and run

Use Python 3.11–3.13 in a dedicated environment (tested: 3.12).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements.txt
enccluster-train --output runs/smoke.json
python -m pytest -q
```

The Binary Fuse bindings compile from source and require a C compiler. On macOS
with standalone Command Line Tools, prefix installation with
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`. No prebuilt Linux binaries
are redistributed.

The default is a small, two-round CPU experiment with generated data. It trains
real client models and performs real P-256 FE operations. The output JSON records
configuration, partition indices and measured metrics. No dataset is downloaded.

For options and a CIFAR experiment with the compact model:

```bash
enccluster-train --help
enccluster-train --dataset cifar10 --download --model tiny --clients 2 \
  --rounds 2 --alpha 10 --precision 1 --output runs/cifar10.json
```

Use `--method fedavg|cluster|bfuse|enccluster` to compare the building blocks.
`--bpe 8` (default) or `--bpe 16` selects the Binary Fuse fingerprint width. Models
`resnet20`, `mlpmixer` and `convnext` retain the recovered architectures; the
compact `tiny` model makes the default validation inexpensive. Full models under
coordinate FE are expensive and have not been benchmarked here.

## Attribution

The existing project website and citation files are preserved. Original research
code, vendored cryptographic primitives and native filters keep their source
attribution and license texts. [PROVENANCE.md](PROVENANCE.md) records exact source
commits, changes and remaining licensing/reproduction review items.
