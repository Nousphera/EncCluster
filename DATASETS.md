# Dataset preparation contract

Run `python scripts/prepare_data.py --help` after installation. The script writes `ROOT/DATASET/manifest.json` together with the files actually opened by training. `--root` and the trainer's `--data-root` always refer to the same **parent** root. Datasets are kept out of Git.

Preparation takes exactly one of `--download`, `--source` and `--validate`. A source is an already extracted directory; user archives are not extracted. Symlinks, existing output directories, class mismatches and malformed inputs are rejected. Imports copy only the selected dataset's consumed directories/files, excluding unrelated data from the source root. They are staged and published only after validation succeeds, leaving source files unchanged. Public downloads use torchvision's official dataset implementations and checksums, without falling back to arbitrary mirrors.

The manifest records the source, preparation convention, split sizes, label-sequence hashes, class-to-index map and SHA256 of every input file. Validation rehashes all files and reopens the splits. Training verifies split/label/class identity and saves the manifest hash; it does not rehash every image in each training round. Run `--validate` immediately before experiments and keep the manifest with results. Exact byte-identical images across ImageFolder train/test splits are rejected; perceptually identical recompressed images are not detected.

## CIFAR-10 and CIFAR-100

Acquire from the [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html), or let preparation download it:

```bash
python scripts/prepare_data.py --dataset cifar10 --root data --download
python scripts/prepare_data.py --dataset cifar10 --root data --validate
```

The expected output is `data/cifar10/cifar-10-batches-py/{data_batch_1,...,data_batch_5,test_batch,batches.meta}` plus the manifest. CIFAR-100 uses `data/cifar100/cifar-100-python/{train,test,meta}`. Official train/test splits are preserved. For an existing extracted download, `--source` points to the directory **containing** `cifar-10-batches-py` or `cifar-100-python`, not to that batch directory itself. The torchvision loader checks official file hashes before reading its Python batch format.

## Explicit image splits

Import existing class folders with:

```bash
python scripts/prepare_data.py --dataset imagefolder --source /path/to/splits --root data
python scripts/prepare_data.py --dataset imagefolder --root data --validate
```

The source layout is:

```text
splits/
  train/class-a/image-001.jpg
  train/class-b/image-002.jpg
  test/class-a/image-101.jpg
  test/class-b/image-102.jpg
```

Every class must have at least one image in both splits. The sorted class directory names define label indices. This adapter supports supplied fixed splits; it does not claim to recover any particular benchmark split or create labels. Sources and labels must be acquired under their applicable terms.
