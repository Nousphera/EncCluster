# EncCluster

**EncCluster: Scalable Functional Encryption in Federated Learning through Weight Clustering and Probabilistic Filters**

[<img src="docs/assets/icons/paper.svg" width="16" height="16" alt=""> Paper](https://www.sciencedirect.com/science/article/pii/S1574119225000100) · [<img src="docs/assets/icons/project.svg" width="16" height="16" alt=""> Project](https://nousphera.github.io/projects/enccluster/) · [<img src="docs/assets/icons/code.svg" width="16" height="16" alt=""> Code](https://github.com/Nousphera/EncCluster) · [<img src="docs/assets/icons/paper.svg" width="16" height="16" alt=""> Preprint](https://arxiv.org/abs/2406.09152) · [<img src="docs/assets/icons/paper.svg" width="16" height="16" alt=""> Poster](https://nousphera.github.io/assets/posters/enccluster-a0-landscape.pdf) · [Citation](CITATION.cff)

Vasileios Tsouvalas · Samaneh Mohammadi · Ali Balador · Tanir Ozcelebi · Francesco Flammini · Nirvana Meratnia

Pervasive and Mobile Computing 108, 102021 (2025)

<p align="center">
  <a href="https://nousphera.github.io/projects/enccluster/#findings">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/results-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/assets/readme/results-light.svg">
      <img src="docs/assets/readme/results-light.svg" width="920" alt="Paper-reported upstream volume relative to FedAvg on CIFAR-10: weight clustering 0.034, DeTrust-FL 3.743, SEFL 2.581, and EncCluster 0.284.">
    </picture>
  </a>
</p>

*Paper results: upstream volume on CIFAR-10, ResNet-20; 30 clients, full participation, 128 clusters, 256-bit keys. [Table 3](https://arxiv.org/pdf/2406.09152v1#page=12), using the paper’s encrypted-centroid transport. Explore the [interactive findings](https://nousphera.github.io/projects/enccluster/#findings).*

## Abstract

Federated Learning (FL) enables model training across decentralized devices by communicating solely local model updates to an aggregation server. Although such limited data sharing makes FL more secure than centralized approached, FL remains vulnerable to inference attacks during model update transmissions. Existing secure aggregation approaches rely on differential privacy or cryptographic schemes like Functional Encryption (FE) to safeguard individual client data. However, such strategies can reduce performance or introduce unacceptable computational and communication overheads on clients running on edge devices with limited resources. In this work, we present EncCluster, a novel method that integrates model compression through weight clustering with recent decentralized FE and privacy-enhancing data encoding using probabilistic filters to deliver strong privacy guarantees in FL without affecting model performance or adding unnecessary burdens to clients. We performed a comprehensive evaluation, spanning various datasets and architectures, to demonstrate EncCluster's scalability across encryption levels. Our findings reveal that EncCluster significantly reduces communication costs - below even conventional FedAvg - and accelerates encryption by more than four times over all baselines; at the same time, it maintains high model accuracy and enhanced privacy assurances.

## Installation

Python 3.12 is tested; package metadata permits 3.11–3.13. Use a dedicated virtual environment. Linux needs a C/C++ toolchain, Python development headers, GMP and libffi development libraries (for example, `build-essential python3-dev libgmp-dev libffi-dev` on Debian/Ubuntu). macOS needs Python 3.12 and the standalone Command Line Tools; install GMP/libffi if native cryptographic wheels are unavailable. On Windows, use WSL2 with the Linux prerequisites; native Windows is untested.

```bash
git clone https://github.com/Nousphera/EncCluster.git
cd EncCluster
bash scripts/setup.sh .venv
source .venv/bin/activate
python -m pytest -q
```

The setup script creates the venv, installs `requirements.txt`, and runs `pip check`. On macOS it selects the installed standalone Command Line Tools. Equivalent manual setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# macOS with standalone Command Line Tools:
# export DEVELOPER_DIR=/Library/Developer/CommandLineTools
python -m pip install -r requirements.txt
```

Requirements install the vendored PyMIFE source and native cryptographic dependencies, and build the bundled Binary Fuse C bindings. Installation and tests do not download datasets or model weights. See [third-party licenses](THIRD_PARTY_LICENSES).

## Data

Prepare once, validate, and pass the same parent root to training:

```bash
python scripts/prepare_data.py --dataset cifar10 --root data --download
python scripts/prepare_data.py --dataset cifar10 --root data --validate
```

This fetches the official [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) Python data through torchvision's checksum-verified loader, preserving its 50,000/10,000 train/test split. Use `cifar100` for CIFAR-100. Output lives under `data/cifar10/`, including `manifest.json` with file SHA256 hashes, class mapping and split sizes. To import an existing torchvision CIFAR root, use `--source /path/to/extracted-cifar-root` instead of `--download`. Existing prepared roots are never overwritten.

For an explicit image dataset split, use `--dataset imagefolder --source /path/to/splits`, where the source contains `train/CLASS/*` and `test/CLASS/*`. Validation checks class agreement, readable images, exact duplicate image content across splits and file changes. [DATASETS.md](DATASETS.md) gives the complete layout and limits; `python scripts/prepare_data.py --help` lists all options.

## Training

A bounded offline check trains the full paper architecture on generated images with plaintext FedAvg:

```bash
enccluster-train --method fedavg --samples-per-client 2 --rounds 1 --output runs/check.json
```

Run the prepared CIFAR data through the same primary trainer:

```bash
enccluster-train --dataset cifar10 --data-root data --model resnet20 \
  --clients 30 --rounds 100 --alpha 10 --local-epochs 1 --kappa 128 --precision 3 --output runs/cifar10.json
enccluster-train --help
```

`--method fedavg|cluster|bfuse|enccluster` selects the plaintext baseline and method components. `--bpe 8|16` sets Binary Fuse fingerprint width. All models train from scratch on 32×32 RGB images, as in paper §5.1. Supported models are `resnet20` (3/3/3 residual blocks, author GroupNorm configuration), `convnext` (the authors’ CIFAR configuration, widths 32/64/128 and depths 2/4/2), `convmixer256_8` (official CIFAR ConvMixer, width 256, depth 8, patch 2, kernel 5), and `mobilenet_v2` (torchvision, width 1.0). The paper calls its 0.16M-parameter CIFAR model “ConvNeXt-Tiny”; this is distinct from torchvision’s 28M ImageNet ConvNeXt-Tiny. MLP-Mixer is not one of the evaluated paper architectures. Full-model coordinate FE can be expensive.

The runner simulates clients and aggregation in one process. Synthetic images check execution only; their metrics are not benchmark results. Encrypted runs process every model coordinate and can be expensive.

## Evaluation

The trainer evaluates the prepared test split after each aggregation round. JSON `history` stores `accuracy` as a fraction in `[0, 1]`, mean cross-entropy `loss`, participants and sample counts. Compare `--method fedavg`, `cluster`, `bfuse` and `enccluster` with the same prepared data, seed and training settings.

Inspect the final evaluation from a run above:

```bash
python - <<'PYTHON'
import json
from pathlib import Path
report = json.loads(Path("runs/cifar10.json").read_text())
print(json.dumps(report["history"][-1], indent=2))
PYTHON
```

### Saved outputs

Each named run saves JSON containing configuration, data-manifest hash, exact client partitions, participant/sample counts, metrics and environment versions. A sibling `.pt` stores the final CPU state dictionary and configuration, loadable with `torch.load(..., weights_only=True)`. Existing JSON/checkpoint paths are protected; choose a new output name to rerun. Save the input manifest and run `--validate` before a research experiment; training checks label/split identity, while full file hashing is an explicit preparation check.

## Results

[Table 3 of the paper](https://arxiv.org/pdf/2406.09152v1#page=12) reports the following CIFAR-10 comparison with ResNet-20, 30 clients, full participation, 128 clusters and 256-bit keys. IID and non-IID use Dirichlet concentrations 10 and 0.1. Accuracy changes are percentage points relative to FedAvg; upstream volume is normalized to FedAvg.

| Method | IID accuracy Δ (pp) | Non-IID accuracy Δ (pp) | Upstream / FedAvg ↓ | Client encryption (s) ↓ |
|---|---:|---:|---:|---:|
| FedAvg + weight clustering | −0.12 | −0.71 | 0.034 | — |
| DeTrust-FL | −0.09 | −0.10 | 3.743 | 329.53 |
| SEFL | −0.98 | −1.74 | 2.581 | 7.64 |
| **EncCluster** | **−0.32** | **−0.79** | **0.284** | **2.04** |

The paper's encrypted-centroid protocol sends 28.4% of FedAvg's upstream volume in this setting, with less than one percentage point of accuracy loss. The weight-clustering row is an unencrypted compression reference. These communication and encryption measurements describe the paper's κ-centroid transport; the executable protocol below encrypts each coordinate. See [interactive findings](https://nousphera.github.io/projects/enccluster/#findings) for additional architectures and settings.

## Protocol

The implementation uses decentralized, coordinate-labelled functional encryption on P-256. Clients cluster weights, expand their local vector, and encrypt each coordinate with a unique session/round/position label. Each client sends **d ciphertexts for d model coordinates**; the paper's compressed transport uses κ encrypted centroids. These are different communication and encryption costs. Client secrets stay in client objects; the aggregation object holds public parameters and functional shares. BatchNorm buffers are aggregated in plaintext.

All selected clients must contribute to a round. This local simulator uses in-process message delivery; network authentication, process isolation and dropout recovery require a separate deployment layer.


Public setup fixes the active cohort, positive integer sample counts, parameter layout, precision and absolute bound. Clients generate independent secrets, exchange pairwise public keys and generate functional shares; the server checks participant IDs, function weights, dimensions and labels. Each selected client must reply. Cryptographic keys use OS entropy independently of the training seed; a changed cohort receives fresh setup.

Weights are rounded after multiplication by `10**precision`. Decryption divides the integer weighted sum by `sum(sample_counts) * 10**precision`; its rounding error is at most `0.5 / 10**precision` per coordinate relative to the reconstructed clustered mean. Nonfinite and out-of-bound weights fail. The public decryption interval is fixed at setup to `±ceil(max_abs * 10**precision + 0.5) * sum(sample_counts)`, and is reused only for that cohort/configuration. Larger bounds, precision and dimensions increase discrete-log work; clustering adds approximation error and coarse precision can erase updates.

K-means clustering uses the run seed and four initializations. Native Fuse8/Fuse16 filters use stable client/server prehashing and a three-location construction. When several clusters match a position, reconstruction chooses the smallest match; false positives introduce approximation error before encryption. BatchNorm integer counters retain the first client value.

The bounded test suite exercises full paper architectures, image preparation, forward/backward passes, plaintext training and checkpoints. Separate small-array tests execute real P-256 encryption, weighted aggregation and malformed-message rejection. It does not encrypt an entire production backbone in CI and does not reproduce the paper benchmarks. Run `python -m pytest -q` and `python -m pip check` after installation.

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

[LICENSE](LICENSE) covers the repository software. [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) preserves component attribution and complete third-party terms. Independently buildable vendored packages retain their own license files; dataset and pretrained-model access terms remain separate.
