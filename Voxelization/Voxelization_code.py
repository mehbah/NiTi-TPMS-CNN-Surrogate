import os
import glob
import numpy as np
import pandas as pd
import trimesh

# === Configuration ===
pitch = 0.166 # voxel size
input_stl_dir = "./STL_Data"
input_csv_dir = "./Stress_Strain_FEA"
output_npy_dir = "./Processed_NPY/"
os.makedirs(output_npy_dir, exist_ok=True)

metadata_rows = []  # to store (lattice_name, strain, step, stress)

stl_files = glob.glob(os.path.join(input_stl_dir, "*.stl"))
print(f"🔍 Found {len(stl_files)} STL files to process.")

for stl_path in stl_files:
    lattice_name = os.path.splitext(os.path.basename(stl_path))[0]
    print(f"\n🔹 Processing lattice: {lattice_name}")

    try:
        mesh = trimesh.load(stl_path)
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"⚠️ Skipping non-Trimesh object: {lattice_name}")
            continue

        filled = mesh.voxelized(pitch).fill()
        voxels = filled.matrix.astype(np.uint8).transpose(2, 1, 0)

        # === CROP Y-DIRECTION ===
        y0 = filled.bounds[0][1]
        y_min_physical = -7.5
        y_max_physical = 7.5
        y_voxel_min = int(np.floor((y_min_physical - y0) / pitch))
        y_voxel_max = int(np.ceil((y_max_physical - y0) / pitch))
        y_voxel_min = max(y_voxel_min, 0)
        y_voxel_max = min(y_voxel_max, voxels.shape[1])
        voxels = voxels[:, y_voxel_min:y_voxel_max, :]  # crop Y
        solid = voxels[..., np.newaxis].astype(np.uint8)
        print(f"✅ Cropped voxel shape: {solid.shape}")

        # === Save voxel once ===
        np.save(os.path.join(output_npy_dir, f"{lattice_name}_90voxel.npy"), solid)

        # === Load Global Stress-Strain CSV ===
        csv_path = os.path.join(input_csv_dir, f"GlobalStressStrain_{lattice_name}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)

        # === Record metadata ===
        for _, row in df.iterrows():
            global_stress = float(row.iloc[1])
            strain_value = float(row.iloc[2])
            load_step = float(row.iloc[3])
            metadata_rows.append([lattice_name, strain_value, load_step, global_stress])

    except Exception as e:
        print(f"❌ Error processing {lattice_name}: {e}")

# === Save metadata CSV ===
metadata_df = pd.DataFrame(metadata_rows, columns=["lattice_name", "strain", "step", "stress"])
metadata_df.to_csv(os.path.join(output_npy_dir, "90voxel_metadata.csv"), index=False)
print(f"\n📄 Metadata saved with {len(metadata_df)} samples.")