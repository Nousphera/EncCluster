# Code-to-paper mapping and reproduction limits

Reference audited: arXiv:2406.09152v1, Sections 2–5, Algorithms 1–2 and
Appendix A. The 2025 PMC publication and NeurIPS 2024 workshop are separate
versions; no cross-version result equivalence is assumed.

| Paper component | Implementation | Status |
|---|---|---|
| Local training + K-means weight clustering, Eq. 5 | `train.py`, `clustering.py` | Real optimizer steps; recovered model architectures and K-means |
| Binary Fuse membership and reconstruction, Eq. 6 | `filters.py`, native `modules/pyfilters` | Genuine Fuse8/Fuse16, stable client/server prehash; first positive match chooses cluster |
| Client-generated secret keys and functional shares | `coordinate_fe.py`, vendored PyMIFE decentralized DDH | Executed with P-256; server has public parameters only |
| Encrypted centroid substitution, Algorithm 2 | Client reconstructs its filter then encrypts every coordinate | Numerically equivalent reconstruction, different encrypted transport and cost |
| Sample-weighted FedAvg | `methods/enccluster.py` | Exact integer sample counts; signed fixed-point roundtrip checked |
| FedAvg / cluster / BF ablations | `methods/` | Executable plaintext controls |
| Paper communication/privacy/architecture tables | No reproduced result artifacts | Not inferred from smoke checks |

The paper's principal setup uses CIFAR-10/100, ResNet-20, 30 clients, one local
epoch, 128 clusters, 256-bit keys, Dir(10) or Dir(0.1), and 100/full or
300–500/20%-participation rounds. Source defaults differ. The maintained harness
exposes cohort, participation, seed and clustering controls, but its coordinate
FE mode changes the transmitted ciphertext count. It does not implement the
paper's reported communication protocol. The current CIFAR path uses resize and
ToTensor; the paper's AutoAugment pipeline is not reproduced.

Native filters have false positives; the server's map was intended to be
recoverable in the paper. The maintained coordinate mode reconstructs the map on
the client before encryption. A first-match rule chooses the smallest matching
cluster. This policy is explicit and can differ from historical implementations
when multiple clusters match. The C header source uses its existing Binary Fuse
construction; the paper's stated 4-wise characterization was not independently
verified. Filter error is not a differential-privacy guarantee.

The old Flower launchers had server-side encryption, plaintext incoming models,
incorrect sample-count handling and inconsistent import/configuration paths. They
remain in the external recovery snapshot for audit, not in the public execution
path. The new harness is an actual local federated loop; no superficial Flower
adapter is advertised as a distributed deployment.

## Protocol boundary

The recovered compressed wrappers encrypted every centroid with the same scalar
key and round label. In that construction, subtracting two ciphertexts cancels
the shared mask and exposes a bounded discrete-log encoding of the centroid
difference. The wrappers also held a central master key; the original EncCluster
Flower path sent plaintext centroids and mappings to the server before encryption.
Those paths are **not shipped as runnable options**.

The maintained path uses PyMIFE's existing `FeDDHMultiClientDec` primitive:
public P-256 parameters, independent client secrets, pairwise key exchange,
client-generated functional shares, and server decryption with all shares.
Each expanded model coordinate has a unique session/round/coordinate label.
The server object retains no client private keys. It checks the participant set,
function weights, labels and dimensions. Cryptographic keys use OS entropy;
training seeds do not seed keys. The local simulator holds all parties in one
process and supplies their messages directly; authenticated transport, process
isolation, persistent keys, dropout recovery and distributed key provisioning
are not implemented. Application input checks are not a cryptographic proof.

This conservative mode sends **d ciphertexts per client**, not kappa encrypted
centroids. It therefore cannot substantiate the paper's encryption-time or
communication claims. Reintroducing centroid reuse requires a compatible
multi-message/label construction and a security argument covering substitution;
changing only the tag to a centroid index is insufficient when different clients
select different centroid indices at the same model position.

PyMIFE is a research library whose authors explicitly state that it is not
peer-reviewed. Tests establish arithmetic and application invariants, not a
security certification. The current release is a research implementation with
these stated differences, not a complete reproduction of the paper protocol.

## Numeric contract

Clients agree on the active cohort, positive integer sample counts, model layout,
fixed-point precision and public absolute weight bound before setup. Each scalar
is rounded to the nearest integer after multiplication by `10**precision`.
Decryption returns the sample-weighted mean of those fixed-point values.
Quantization alone introduces at most `0.5 / 10**precision` absolute error per
coordinate relative to the reconstructed clustered mean. K-means/filter error
is additional. Oversized/nonfinite values are rejected, not silently clipped.
The bounded discrete-log search can be expensive for large sample counts,
precision or models. This execution mode is intended for bounded experiments.
BatchNorm running statistics are aggregated in plaintext; integer counters
retain the source convention. These buffers are outside the encrypted guarantee.

## Reproducibility

Python 3.12 on Apple Silicon was used for release checks. Supported metadata is
Python 3.11–3.13; other Python/platform combinations have not been executed here.
`constraints.txt` records the tested dependency closure. Model/training,
partition and clustering randomness is seeded; crypto randomness remains fresh.
The JSON output records CLI settings, exact partition indices, forget indices
when applicable, participant IDs, sample counts, losses and accuracies. It does
not contain input records, weights, private keys or claims of benchmark parity.

The default two-round synthetic test uses two clients, eight training samples per
client, three classes, a 4-to-3 linear model, four K-means centroids, precision 2,
CPU and one local epoch. No data/model download occurs on import, installation
or test. `--download` is required to fetch CIFAR. `--pretrained` explicitly allows
weights to be fetched for supported EFU image models. No paper checkpoints or
paper-result logs are included. No GPU run or full paper benchmark was executed.

The shared CLI supports Dirichlet label partitioning for CIFAR, explicit
participation, local epochs, learning rate, seed, model, precision, public bound
and output path. It rejects empty clients and cohorts with fewer than two active
participants. A changed cohort gets a fresh decentralized setup. Setup cost is
included in the current simulator and is not comparable to the paper timings.

## Release artifact protection

The maintained CLI rejects an existing output JSON or sibling checkpoint before training. Final `.pt` files contain CPU state dictionaries, configuration and class count, loadable with `torch.load(path, weights_only=True, map_location="cpu")`. The JSON includes Python, platform, torch and NumPy versions. Save the dependency constraints and input-data version alongside the run. Encryption key generation remains random; seeded training does not imply deterministic ciphertext bytes.

## Dataset setup follow-up

[DATASETS.md](DATASETS.md) defines the maintained preparation contract. Training now consumes `ROOT/DATASET` roots created by `scripts/prepare_data.py`; older direct CIFAR roots can be imported with `--source`. The preparation manifest identifies class maps, split sizes, label order and file hashes. Reports include its SHA256. Generated source-format fixtures exercise import, validation and actual training; their metrics are correctness diagnostics. No newly implemented ingestion path implies recovered historical splits or reproduced benchmark results.
