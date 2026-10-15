# Implementation and experiment settings

EncCluster combines local training, K-means weight clustering, Binary Fuse membership filters and encrypted aggregation. See [the paper](https://doi.org/10.1016/j.pmcj.2025.102021) and [preprint methods](https://arxiv.org/abs/2406.09152).

| Component | Code | Behavior |
|---|---|---|
| Local training | `src/enccluster/train.py`, `network.py` | Seeded client training with tiny, ResNet20, MLP-Mixer or ConvNeXt models |
| Weight clustering | `clustering.py` | K-means centroids and a coordinate-to-centroid map |
| Probabilistic map | `filters.py`, `modules/pyfilters` | Native Fuse8/Fuse16 membership with stable client/server prehashing |
| Encrypted aggregation | `coordinate_fe.py`, `methods/enccluster.py` | Client-side map reconstruction followed by coordinate-labelled P-256 encryption |
| Controls | `methods/` | Plain FedAvg, clustering-only and Binary-Fuse reconstruction |

## Clustering and filters

`--method fedavg|cluster|bfuse|enccluster` selects the full pipeline or a component control. `--kappa` sets the number of centroids and `--bpe 8|16` selects the Binary Fuse fingerprint width. Membership queries recover a centroid assignment for each model position; if more than one cluster matches, the smallest matching cluster is used. The client reconstructs this approximate clustered vector before encryption.

The bundled native Binary Fuse implementation uses its own three-location construction. Filter false positives can change assignments; clustering and filter reconstruction are lossy operations. Filter error is an approximation effect. Full FE runs use the reconstructed coordinate vector, as specified below.

## Experiment configuration

The CLI exposes client count, participation, local epochs, batch size, learning rate, seed, Dirichlet concentration, model and quantization. CIFAR-10/100 use their official train/test splits. Images are resized to 32×32 and converted to tensors; image models apply CIFAR normalization internally. The default synthetic case is a complete offline execution check.

The paper's principal settings include CIFAR-10/100, ResNet-20, 30 clients, one local epoch, 128 centroids, 256-bit keys, and Dirichlet concentration 10 or 0.1. Configure the run length and participation for the experiment being studied; the CLI defaults are intentionally small. The seeded Dirichlet partitioner uses each training sample once and rejects empty clients. At least two clients must be active in encrypted rounds.

## Coordinate-labelled functional encryption

Aggregation uses PyMIFE's `FeDDHMultiClientDec` primitive over P-256: public parameters, independent client secrets, pairwise key exchange, client-generated functional shares and decryption of the sample-weighted sum. Each expanded coordinate has a unique session/round/coordinate label. The aggregator holds public parameters and shares; it checks participant IDs, function weights, labels and dimensions. Keys use operating-system entropy independently of training seeds.

Each client sends **d ciphertexts for d model coordinates**. The paper's compressed-centroid transport uses κ ciphertexts; the implemented coordinate transport has a different cost. Client-side expansion gives every coordinate its own encryption label even when several coordinates use the same centroid. BatchNorm running statistics are averaged in plaintext and integer counters retain the first client value.

The executable is a local simulator: all parties run in one process and exchange messages directly. Each selected client must reply. Authenticated transport, client-process isolation, persistent key provisioning and dropout handling belong to a deployment layer. PyMIFE is a research cryptography library; application tests verify arithmetic and message invariants.

## Numerical representation

Clients agree on the active cohort, positive integer sample counts, model layout, fixed-point precision and public absolute weight bound. A weight is rounded after multiplication by `10**precision`; decryption returns the sample-weighted mean of those integers, scaled back to real values. The fixed-point rounding error is at most `0.5 / 10**precision` per coordinate relative to the reconstructed clustered mean; clustering and filter approximation add their own error. Nonfinite and out-of-bound values fail validation.

Larger precision, sample counts and model dimensions increase bounded discrete-log work. Coarse precision can erase small updates. Use the tiny model to check the complete path before allocating a full-model experiment, and record `kappa`, `precision` and `max_abs` with results. A changed active cohort receives a fresh decentralized setup.

## Data, randomness and saved results

[DATASETS.md](DATASETS.md) specifies preparation and consumed paths. `scripts/prepare_data.py` creates `ROOT/DATASET/manifest.json`; the trainer's `--data-root` is the same parent root. Run `--validate` before an experiment to hash every file. Training checks split sizes, class maps and label order, and records the manifest SHA256.

Training, partitioning and K-means use explicit seeds; cryptographic randomness stays fresh. The default synthetic configuration uses two clients, eight training vectors per client, three classes, a linear model, four centroids, precision 2, CPU and one local epoch. Imports, installation and tests require no dataset or pretrained-weight download.

JSON records configuration, exact client partitions, participants, sample counts, metrics and Python/platform/Torch/NumPy versions. The sibling `.pt` contains CPU model tensors, configuration and class information:

```python
import torch
checkpoint = torch.load("runs/experiment.pt", weights_only=True, map_location="cpu")
```

Existing output or checkpoint paths are rejected before training. Keep the prepared manifest, dependency constraints and seed with each result. Client keys and plaintext input records are excluded from reports.

## Validation

```bash
python -m pip check
python -m pytest -q
```

The suite covers seeded partitions, the primary trainer, prepared image inputs, signed fixed-point encrypted aggregation, sample weighting, participant/message validation and checkpoint output. `constraints.txt` records the dependency versions tested with Python 3.12 on macOS arm64.

Native-filter tests cover membership roundtrips, stable hashing across processes, reconstruction, signed clustered aggregation and both fingerprint widths.
