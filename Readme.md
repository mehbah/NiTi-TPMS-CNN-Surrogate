# NiTi TPMS Lattice Voxelization and 3D CNN Surrogate Model

Code to (1) voxelize NiTi TPMS lattice geometries from STL files and pair them with FEA stress–strain data, and (2) train a 3D CNN surrogate model via 5-fold cross-validation to predict point-wise pseudoelastic stress from voxelized geometry, strain, and loading state. Accompanies the manuscript *[paper title, journal, DOI — add on acceptance]*.

## Structure

```
repo-root/
├── README.md
├── 3DCNN_90voxels_Crossval.py
├── Test_Data/                       # held-out generalisation-fold lattices
└── Voxelization/
    ├── Voxelization_code.py
    ├── STL_Data/                    # input: lattice mesh files (.stl)
    ├── Stress_Strain_FEA/           # input: FEA stress-strain CSVs
    └── Processed_NPY/               # output: voxel arrays + metadata.csv
```

Step 1 (`Voxelization_code.py`) converts each lattice's STL mesh into a binary voxel grid and builds a metadata table linking each (lattice, strain, loading state) to its FEA stress. Step 2 (`3DCNN_90voxels_Crossval.py`) trains a 3D CNN on that data via `GroupKFold` (grouped by `lattice_name`, so no lattice contributes to both train and validation within a fold).

## Requirements

```bash
pip install numpy pandas trimesh tensorflow scikit-learn
```
(`trimesh` is only needed for Step 1; `tensorflow` and `scikit-learn` only for Step 2.)

## Data

Code and data are both hosted in this repository. Data is provided as zip archives attached to the **[Releases](../../releases)** page (too large to track as regular repo files):

| Archive | Contents |
|---|---|
| `STL_Data.zip` | Lattice mesh files for all configurations |
| `Stress_Strain_FEA.zip` | FEA stress-strain CSVs, one per lattice (`GlobalStressStrain_{lattice_name}.csv`) |
| `Processed_NPY.zip` | Pre-computed voxel arrays + `90voxel_metadata.csv` (skip Step 1 if using this) |
| `Test_Data.zip` | Held-out generalisation-fold lattices (currently Fold 1: SplitP, 0.5/1.0/1.5 mm) |

Download and extract these into the folder structure above before running the scripts. *(Optional: this repository is archived with a permanent DOI via Zenodo — [DOI link, if applicable].)*

> Not yet included: a script for evaluating trained models on `Test_Data` (predicting held-out lattices and extracting EM/UCS/HA). Let us know if this should be added.

## Usage

**Step 1 — Voxelization**
```bash
cd Voxelization
python Voxelization_code.py
```
Voxelizes each STL at `pitch = 0.166` (90-voxel resolution), crops the Y-axis to ±7.5 mm, and saves `{lattice_name}_90voxel.npy` plus `90voxel_metadata.csv` to `./Processed_NPY/`.

**Step 2 — Model training**
```bash
cd ..
python 3DCNN_90voxels_Crossval.py
```
Loads the metadata and voxel arrays, builds a 3-channel input (geometry, strain, loading state), and trains one 3D CNN per `GroupKFold` fold (early stopping, patience 20). Outputs per fold: `model_FOLD{n}_..._90.keras`, `train_predictions_fold{n}.txt`, `test_predictions_fold{n}.txt`, `TrainingHistory_FOLD{n}_..._90.csv`.

**Hyperparameters** (fixed to the manuscript's Table 3 Bayesian-optimisation result; edit near the top of `main()` in `3DCNN_90voxels_Crossval.py` to change):

| Parameter | Value | | Parameter | Value |
|---|---|---|---|---|
| Learning rate | 0.0001 | | Activation | Leaky-ReLU |
| Dropout rate | 0.1 | | Optimiser | Adam |
| Conv3D filters | [16, 32, 64] | | Epochs | 300 |
| Dense layers | [128, 64, 32] | | Loss | Huber |
| Batch size | 4 | | Kernel size | 3 |

## Reproducibility notes

- `GroupKFold` by `lattice_name` prevents within-lattice leakage; fold assignment and batch order are deterministic.
- Voxel arrays load as `float16` (memory efficiency, not a correctness requirement).
- Voxelization pitch/crop are fixed to this study's 5 mm, 3×3×3 unit cells — adjust for other geometries.
- **No random seed is set**, so weight initialization and dropout (and thus training/validation loss) differ between runs. Add seeds for `numpy`, `random`, and `tf.random` in `main()` for reproducible runs; GPU bit-for-bit reproducibility additionally needs `tf.config.experimental.enable_op_determinism()`.

## Citation

```
[Author list]. [Year]. [Paper title]. [Journal]. [DOI]
```

## License

[Specify license, e.g. MIT]

## Contact

[Corresponding author email] or open an issue in this repository.
