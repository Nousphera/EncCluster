# EncCluster

**EncCluster: Scalable Functional Encryption in Federated Learning through Weight Clustering and Probabilistic Filters**

[Paper](https://www.sciencedirect.com/science/article/pii/S1574119225000100) · [Preprint](https://arxiv.org/abs/2406.09152) · [Project](https://nousphera.github.io/EncCluster/) · [Citation](CITATION.cff)

## Abstract

Federated Learning (FL) enables model training across decentralized devices by communicating solely local model updates to an aggregation server. Although such limited data sharing makes FL more secure than centralized approached, FL remains vulnerable to inference attacks during model update transmissions. Existing secure aggregation approaches rely on differential privacy or cryptographic schemes like Functional Encryption (FE) to safeguard individual client data. However, such strategies can reduce performance or introduce unacceptable computational and communication overheads on clients running on edge devices with limited resources. In this work, we present EncCluster, a novel method that integrates model compression through weight clustering with recent decentralized FE and privacy-enhancing data encoding using probabilistic filters to deliver strong privacy guarantees in FL without affecting model performance or adding unnecessary burdens to clients. We performed a comprehensive evaluation, spanning various datasets and architectures, to demonstrate EncCluster's scalability across encryption levels. Our findings reveal that EncCluster significantly reduces communication costs - below even conventional FedAvg - and accelerates encryption by more than four times over all baselines; at the same time, it maintains high model accuracy and enhanced privacy assurances.

*The abstract describes the paper. This maintained research implementation uses decentralized coordinate-labelled FE. It does not reproduce the paper's compressed-ciphertext costs or establish its security claims.* See [implementation scope](REPRODUCIBILITY.md).

## Setup

Python 3.12 is tested; package metadata permits 3.11–3.13. Use a dedicated virtual environment. Linux needs a C/C++ toolchain, Python development headers, GMP and libffi development libraries (for example, `build-essential python3-dev libgmp-dev libffi-dev` on Debian/Ubuntu). macOS needs Python 3.12 and the standalone Command Line Tools; install GMP/libffi if native cryptographic wheels are unavailable. On Windows, use WSL2 with the Linux prerequisites; native Windows is untested.

```bash
git clone https://github.com/Nousphera/EncCluster.git
cd EncCluster
bash scripts/setup.sh .venv
source .venv/bin/activate
python -m pytest -q
```

The setup script creates the venv, installs `requirements.txt` with the tested `constraints.txt`, and runs `pip check`. It selects installed macOS Command Line Tools without accepting an Xcode license. Equivalent manual setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# macOS with standalone Command Line Tools:
# export DEVELOPER_DIR=/Library/Developer/CommandLineTools
python -m pip install -c constraints.txt -r requirements.txt
```

Requirements install the vendored PyMIFE source and native cryptographic dependencies, and build the bundled Binary Fuse C bindings. Installation and tests do not download datasets or model weights. See [source attribution](PROVENANCE.md).

## Datasets

Prepare once, validate, and pass the same parent root to training:

```bash
python scripts/prepare_data.py --dataset cifar10 --root data --download
python scripts/prepare_data.py --dataset cifar10 --root data --validate
```

This fetches the official [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) Python data through torchvision's checksum-verified loader, preserving its 50,000/10,000 train/test split. Use `cifar100` for CIFAR-100. Output lives under `data/cifar10/`, including `manifest.json` with file SHA256 hashes, class mapping and split sizes. To import an existing torchvision CIFAR root, use `--source /path/to/extracted-cifar-root` instead of `--download`. Existing prepared roots are never overwritten.

For an explicit image dataset split, use `--dataset imagefolder --source /path/to/splits`, where the source contains `train/CLASS/*` and `test/CLASS/*`. Validation checks class agreement, readable images, exact duplicate image content across splits and file changes. [DATASETS.md](DATASETS.md) gives the complete layout and limits; `python scripts/prepare_data.py --help` lists all options.

## Run

A two-round offline check trains real client models and executes real P-256 functional encryption:

```bash
enccluster-train --output runs/smoke.json
```

Run the prepared CIFAR data through the same primary trainer:

```bash
enccluster-train --dataset cifar10 --data-root data --model tiny \
  --clients 2 --rounds 2 --alpha 10 --precision 1 --output runs/cifar10.json
enccluster-train --help
```

`--method fedavg|cluster|bfuse|enccluster` selects the plaintext baseline and method components. `--bpe 8|16` sets Binary Fuse fingerprint width. Recovered full models are `resnet20`, `mlpmixer` and `convnext`; `tiny` is a diagnostic image model. Full-model coordinate FE can be expensive.

The maintained runner is a local federated simulation, with all parties in one process. Its input and aggregation paths are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Outputs and reproducibility

Each named run saves JSON containing configuration, data-manifest hash, exact client partitions, participant/sample counts, metrics and environment versions. A sibling `.pt` stores the final CPU state dictionary and configuration, loadable with `torch.load(..., weights_only=True)`. Existing JSON/checkpoint paths are protected; choose a new output name to rerun. Save the input manifest and run `--validate` before a research experiment; training checks label/split identity, while full file hashing is an explicit preparation check.

**The release encrypts expanded coordinates (d ciphertexts), not kappa centroids.** It retains the local learning, clustering and native filter operations while avoiding mask reuse in the recovered compressed wrapper. It does not establish paper benchmark results, cryptographic security, network isolation or dropout recovery. Full-model encrypted runs are expensive and unbenchmarked. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) separates paper settings, implemented behavior and remaining gaps.

## Citation

```bibtex
@article{tsouvalas2025enccluster,
  title = {{EncCluster}: Scalable functional encryption in federated learning through weight clustering and probabilistic filters},
  author = {Tsouvalas, Vasileios and Mohammadi, Samaneh and Balador, Ali and Ozcelebi, Tanir and Flammini, Francesco and Meratnia, Nirvana},
  journal = {Pervasive and Mobile Computing},
  volume = {108},
  pages = {102021},
  year = {2025},
  publisher = {Elsevier},
  doi = {10.1016/j.pmcj.2025.102021},
  url = {https://doi.org/10.1016/j.pmcj.2025.102021}
}
```

## License and attribution

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [PROVENANCE.md](PROVENANCE.md) for the first-party and third-party terms and exact source commits. Imported code retains its original license notices; these may differ from the repository's root license. The recovered pyfilters wrapper declares Apache-2.0 in package metadata but lacks a standalone license grant; that provenance limitation remains recorded.
