import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """Convert a Pandas DataFrame into a TensorFlow Dataset."""
    df = dataframe.copy()
    labels = df.pop('target')
    # Convert each feature column to (batch, 1) shape
    df = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def get_normalization_layer(name, dataset):
    """Create and adapt a normalization layer for a numerical feature."""
    normalizer = layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])  # Extract a single feature
    normalizer.adapt(feature_ds)
    return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    """Create and adapt a category encoding layer for a categorical feature."""
    # Create either a StringLookup or IntegerLookup based on dtype
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)

    # Adapt the index layer to the dataset
    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)

    # Create a one-hot (multi-hot) encoding layer
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    return lambda feature: encoder(index(feature))

def load_datasets(csv_file, batch_size=32):
    """
    Load the CSV file and split it into train, validation, and test sets.
    Returns tf.data.Datasets for each split.
    """
    dataframe = pd.read_csv(csv_file)

    # Rename 'diabetes' column to 'target'
    dataframe['target'] = dataframe.pop('diabetes')

    # Randomly shuffle and split data into train/val/test
    train, val, test = np.split(
        dataframe.sample(frac=1),
        [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))]
    )

    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds

def build_feature_layers(train_ds):
    """
    Build input layers and corresponding preprocessing/encoding layers for each feature.
    Returns (dict_of_inputs, list_of_encoded_features).
    """
    all_inputs = {}
    encoded_features = []

    # Numerical features
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    for header in numeric_features:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs[header] = numeric_col
        encoded_features.append(encoded_numeric_col)

    # Integer categorical features
    int_categorical_cols = ['hypertension', 'heart_disease']
    for header in int_categorical_cols:
        int_categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int32')
        encoding_layer = get_category_encoding_layer(
            name=header, dataset=train_ds, dtype='int', max_tokens=5
        )
        encoded_int_categorical_col = encoding_layer(int_categorical_col)
        all_inputs[header] = int_categorical_col
        encoded_features.append(encoded_int_categorical_col)

    # String categorical features
    categorical_cols = ['gender', 'smoking_history']
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(
            name=header, dataset=train_ds, dtype='string', max_tokens=5
        )
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs[header] = categorical_col
        encoded_features.append(encoded_categorical_col)

    return all_inputs, encoded_features

def build_model(all_inputs, encoded_features, dropout_rate=0.5):
    """
    Build and compile the TensorFlow model.
    """
    # Concatenate all encoded features
    all_features = layers.concatenate(encoded_features)
    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=all_inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def train_evaluate(hparams):
    """
    Main training procedure: loads datasets, builds a model, trains, evaluates,
    and exports the model in SavedModel format.

    hparams(dict) is expected to have the following keys:
      - 'data-file': path to the CSV file
      - 'batch-size': batch size
      - 'epochs': number of epochs
      - 'dropout': dropout rate
      - 'model-dir': path where the model will be exported
    """
    # 1. Load the data
    csv_file = hparams['data-file']
    train_ds, val_ds, test_ds = load_datasets(csv_file, batch_size=hparams['batch-size'])

    # 2. Build feature layers and the model
    all_inputs, encoded_features = build_feature_layers(train_ds)
    model = build_model(all_inputs, encoded_features, dropout_rate=hparams['dropout'])

    # 3. Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=hparams['epochs']
    )

    # 4. Evaluate on the test set
    test_result = model.evaluate(test_ds, return_dict=True)
    print("Test evaluation:", test_result)

    # 5. Export the model
    export_dir = hparams['model-dir']
    model.save(export_dir)
    print(f"Model exported to: {export_dir}")

    return history
