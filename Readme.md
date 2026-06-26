# NiTi-TPMS-CNN-Surrogate

This repository contains the code used to (1) voxelize NiTi TPMS lattice geometries from STL files and pair them with FEA stress–strain data, and (2) train a 3D convolutional neural network (CNN) surrogate model, via 5-fold cross-validation, to predict point-wise pseudoelastic stress from voxelized geometry, strain, and loading state.

This code accompanies the manuscript *A Deep Learning Framework for Predicting Nonlinear Pseudoelasticity in NiTi
Triply Periodic Minimal Surface Lattice Structures, Mehran Bahramyan et al.*.

## Repository structure

The training script expects the voxelization script and its outputs to live in a `Voxelization/` subfolder:

```
repo-root/
├── README.md
├── 3DCNN_90voxels_Crossval.py
├── Test_Data/                       # held-out generalisation-fold lattices (see below)
└── Voxelization/
    ├── Voxelization_code.py
    ├── STL_Data/                    # input: lattice mesh files (.stl)
    ├── Stress_Strain_FEA/           # input: FEA stress-strain CSVs
    └── Processed_NPY/               # output: voxel arrays + metadata.csv (generated)
```

## Pipeline overview

```
STL mesh files  ─┐
                  ├─► Voxelization_code.py ─► voxel arrays (.npy) + metadata.csv
FEA stress-strain ┘         (run from inside Voxelization/)
   CSV files                                          │
                                                       ▼
                                  3DCNN_90voxels_Crossval.py
                                       (run from repo root)
                                                       │
                                                       ▼
                  5 trained models (.keras) + predictions + training history
```

Step 1 converts each lattice's surface mesh into a binary solid/void voxel grid and builds a single metadata table linking every (lattice, strain, loading state) combination to its simulated stress. Step 2 trains a 3D CNN on that data using 5-fold cross-validation grouped by lattice (`GroupKFold` on `lattice_name`), so that no single lattice contributes points to both the training and validation set within a fold.

## Repository contents

| File | Purpose |
|---|---|
| `Voxelization/Voxelization_code.py` | Converts STL lattice geometries into binary voxel grids and builds the training metadata table from FEA stress-strain CSVs. |
| `3DCNN_90voxels_Crossval.py` | Defines and trains the 3D CNN surrogate model using 5-fold group cross-validation, with hyperparameters fixed to the values selected via Bayesian optimisation in the manuscript (Table 3). |

## Requirements

Tested with Python 3.10+. Install dependencies with:

```bash
pip install numpy pandas trimesh tensorflow scikit-learn
```

- `trimesh` is required only for Step 1 (voxelization).
- `tensorflow` (with Keras) and `scikit-learn` are required for Step 2 (model training).

## Data

This repository contains code only. The data is hosted separately:

- **Raw data** (STL geometries, FEA stress-strain CSVs) and **processed data** (voxel arrays, metadata table) are available at **[Zenodo DOI — 10.5281/zenodo.20935966]**.
- The processed voxel arrays (`Processed_NPY/`) are provided pre-computed as a shortcut — you do not need to re-run the voxelization step to train the model. Re-run it only if you want to regenerate the voxel grids from the raw STL files yourself (e.g. after modifying the voxelization parameters).

Folder descriptions:

| Folder | Contents |
|---|---|
| `STL_Data/` | STL mesh files for all lattice geometries (training pool and generalisation-fold lattices). |
| `Stress_Strain_FEA/` | FEA-derived global stress-strain CSVs, one per lattice, matched by filename to `STL_Data/` (`GlobalStressStrain_{lattice_name}.csv`). Used as input to the voxelization script to build the training metadata table. |
| `Processed_NPY/` | Voxel arrays (`{lattice_name}_90voxel.npy`) and the combined `90voxel_metadata.csv`, produced by running `Voxelization_code.py`. |
| `Test_Data/` | Held-out lattices excluded entirely from training, used to evaluate generalisation to unseen topologies. Currently contains Generalisation Fold 1 (SplitP, wall thicknesses 0.5/1.0/1.5 mm); folds for SchwarzD and Lidinoid follow the same format. |

> **Note:** this repository does not currently include a separate script for evaluating the trained models on `Test_Data` (i.e. predicting the held-out generalisation-fold lattices and extracting EM/UCS/HA). If that evaluation code should be included, let us know and we'll add it alongside the two scripts here.

## Usage

### Step 1 — Voxelization

```bash
cd Voxelization
python Voxelization_code.py
```

For each STL file in `./STL_Data/`, this script:
- Voxelizes the mesh at a fixed grid spacing (`pitch = 0.166`, chosen to yield a 90-voxel resolution across the lattice unit cell domain) and crops the Y-axis to a fixed physical range (-7.5 mm to 7.5 mm) so all lattices share a common grid size regardless of their original mesh bounds.
- Saves the resulting binary solid/void voxel array as `{lattice_name}_90voxel.npy`.
- Reads the matching `GlobalStressStrain_{lattice_name}.csv` file from `./Stress_Strain_FEA/` and appends one row per (strain, loading step, stress) data point to a combined metadata table.

**Outputs** (written to `./Processed_NPY/`, i.e. `Voxelization/Processed_NPY/` relative to the repo root):
- `{lattice_name}_90voxel.npy` — one voxel array per lattice
- `90voxel_metadata.csv` — combined table of `lattice_name, strain, step, stress` across all lattices

### Step 2 — Model training (5-fold cross-validation)

```bash
cd ..   # back to repo root
python 3DCNN_90voxels_Crossval.py
```

This script:
- Loads `./Voxelization/Processed_NPY/90voxel_metadata.csv` and the corresponding voxel arrays.
- Builds a 3-channel input volume per sample: channel 1 is the binary geometry voxel grid, channels 2 and 3 are constant volumes filled with the strain value and loading-state flag for that sample, respectively.
- Splits the data into 5 folds using `GroupKFold` grouped by `lattice_name`, ensuring every point from a given lattice falls entirely within either the training or validation set for each fold (no within-lattice leakage).
- Trains a 3D CNN (Conv3D + MaxPooling3D blocks, followed by dense layers and dropout, with a single linear output for predicted stress) for each fold, with early stopping on validation loss (patience = 20 epochs).

**Outputs** (written to the working directory, i.e. the repo root):
- `model_FOLD{n}_..._90.keras` — trained model weights for each fold
- `train_predictions_fold{n}.txt`, `test_predictions_fold{n}.txt` — true vs. predicted stress values
- `TrainingHistory_FOLD{n}_..._90.csv` — per-epoch training/validation loss and MAE

### Hyperparameters

Hyperparameters are fixed to the values selected via Bayesian optimisation, as reported in the manuscript (Table 3):

| Parameter | Selected value |
|---|---|
| Learning rate | 0.0001 |
| Dropout rate | 0.1 |
| Conv3D filters | [16, 32, 64] |
| Dense layer sizes | [128, 64, 32] |
| Batch size | 4 |
| Kernel size (Conv3D) | 3 |
| Activation function | Leaky-ReLU |
| Optimiser | Adam |
| Epochs | 300 |
| Loss function | Huber |

To explore alternative hyperparameters, edit the corresponding variables near the top of `main()` in `3DCNN_90voxels_Crossval.py`.

## Reproducibility notes

- Cross-validation is grouped by `lattice_name` (`GroupKFold`), so results reflect genuine generalization to unseen lattice configurations rather than interpolation within a single curve.
- Voxel arrays are loaded as `float16` in the data generator to reduce memory footprint during training; this is a precision choice, not a correctness requirement.
- The voxelization pitch (`0.166`) and Y-axis crop range (`±7.5 mm`) are fixed to the unit cell dimensions used in this study (5 mm unit cells, 3×3×3 array) and should be adjusted if applied to lattices of a different size.
- **No random seed is set.** Model weight initialization, dropout masks, and the resulting training/validation loss values will differ between runs, even with identical data and hyperparameters. The fold assignments themselves (`GroupKFold`) and batch order are deterministic given the same metadata file. To obtain fully reproducible training runs, set seeds for `numpy`, Python's `random`, and `tf.random` near the top of `main()`; note that bit-for-bit reproducibility on GPU additionally requires `tf.config.experimental.enable_op_determinism()`.

## Citation

If you use this code, please cite:

```
[Author list]. [Year]. [Paper title]. [Journal]. [DOI]
```

## License

[Specify license, e.g. MIT — add before publishing]

## Contact

For questions, please contact [corresponding author email] or open an issue in this repository.