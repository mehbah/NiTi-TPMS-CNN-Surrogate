import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Dropout, Input, MaxPooling3D, LeakyReLU
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import GroupKFold
import pandas as pd
import os


def get_optimizer(name, lr):
    if name == "adam":
        return Adam(learning_rate=lr)
    elif name == "sgd":
        return SGD(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_loss(name):
    if name == "mse":
        return "mse"
    elif name == "mae":
        return "mae"
    elif name == "huber":
        return Huber()
    else:
        raise ValueError(f"Unsupported loss: {name}")


def get_activation_layer(name):
    if name == "relu":
        return tf.keras.layers.Activation("relu")
    elif name == "leaky_relu":
        return LeakyReLU()
    elif name == "elu":
        return tf.keras.layers.Activation("elu")
    else:
        raise ValueError(f"Unsupported activation: {name}")


def build_model(input_shape, filters, dense_units, activation, dropout_rate, optimizer, learning_rate, loss_fn):
    input_layer = Input(shape=input_shape)
    x = input_layer

    for f in filters:
        x = Conv3D(f, kernel_size=3, padding="same")(x)
        x = get_activation_layer(activation)(x)
        x = MaxPooling3D(pool_size=2)(x)

    x = Flatten()(x)
    for units in dense_units:
        x = Dense(units)(x)
        x = get_activation_layer(activation)(x)

    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=get_optimizer(optimizer, learning_rate),
        loss=get_loss(loss_fn),
        metrics=["mae"]
    )
    return model


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, metadata_df, voxel_dir, batch_size, input_shape):
        self.metadata = metadata_df
        self.voxel_dir = voxel_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.indices = np.arange(len(metadata_df))

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.metadata.iloc[batch_idx]

        X_batch = np.zeros((len(batch_data),) + self.input_shape, dtype=np.float16)
        Y_batch = np.zeros(len(batch_data), dtype=np.float16)

        for i, (_, row) in enumerate(batch_data.iterrows()):
            lattice = row['lattice_name']
            strain = row['strain']
            step = row['step']
            stress = row['stress']

            voxel_path = os.path.join(self.voxel_dir, f"{lattice}_90voxel.npy")
            solid = np.load(voxel_path).astype(np.float16)

            strain_channel = np.full_like(solid, strain, dtype=np.float16)
            step_channel = np.full_like(solid, step, dtype=np.float16)
            X_batch[i] = np.concatenate([solid, strain_channel, step_channel], axis=-1)
            Y_batch[i] = stress

        return X_batch, Y_batch


def main():

    # === Load metadata ===
    metadata = pd.read_csv("./Voxelization/Processed_NPY/90voxel_metadata.csv")
    voxel_dir = "./Voxelization/Processed_NPY/"
    print(f"✅ Metadata loaded: {metadata.shape[0]} samples")

    input_shape = tuple(np.load(os.path.join(voxel_dir, f"{metadata.iloc[0]['lattice_name']}_90voxel.npy")).shape[:3]) + (3,)

    # === Best hyperparameters (Table 3 of the manuscript: Bayesian optimisation
    # search space and final selected values) ===
    learning_rate = 0.0001
    dropout_rate = 0.1
    batch_size = 4
    epochs = 300
    filters = [16, 32, 64]
    dense_units = [128, 64, 32]
    optimizer = "adam"
    activation = "leaky_relu"
    loss_fn = "huber"

    gkf = GroupKFold(n_splits=5)
    fold = 1

    for train_idx, val_idx in gkf.split(metadata, groups=metadata["lattice_name"]):
        print(f"\n🔹 Fold {fold} — Train: {len(train_idx)} | Val: {len(val_idx)}")

        metadata_train = metadata.iloc[train_idx].reset_index(drop=True)
        metadata_val = metadata.iloc[val_idx].reset_index(drop=True)

        train_gen = DataGenerator(metadata_train, voxel_dir, batch_size, input_shape)
        val_gen = DataGenerator(metadata_val, voxel_dir, batch_size, input_shape)

        fold_run_name = f"FOLD{fold}_drop{dropout_rate}_lr{learning_rate}_bs{batch_size}_act{activation}_f163264_den1286432_loss{loss_fn}_90"

        model = build_model(input_shape, filters, dense_units, activation, dropout_rate,
                            optimizer, learning_rate, loss_fn)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )

        model.save(f"model_{fold_run_name}.keras")
        print(f"✅ Saved model: model_{fold_run_name}.keras")

        # === Predict on training and validation generators ===
        Y_train_pred = model.predict(train_gen)
        Y_val_pred = model.predict(val_gen)
        # === Get ground truth from metadata ===
        Y_train_true = metadata_train["stress"].values
        Y_val_true = metadata_val["stress"].values

        # === Save predictions ===
        np.savetxt(f"train_predictions_fold{fold}.txt", np.column_stack((Y_train_true, Y_train_pred.flatten())), fmt="%.6f", header="Y_true Y_pred", comments='')
        np.savetxt(f"test_predictions_fold{fold}.txt", np.column_stack((Y_val_true, Y_val_pred.flatten())), fmt="%.6f", header="Y_true Y_pred", comments='')

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f"TrainingHistory_{fold_run_name}.csv", index=False)
        print(f"✅ Saved training history: TrainingHistory_{fold_run_name}.csv")

        fold += 1


if __name__ == "__main__":
    main()