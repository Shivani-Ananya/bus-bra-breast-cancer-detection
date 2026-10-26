"""Patient-safe split creation and tf.data input pipelines."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .preprocess import decode_resize, build_image_augmentation


def load_metadata(path, cfg):
    df = pd.read_csv(path)
    cols = [cfg["patient_col"], cfg["image_col"], cfg["label_col"]]
    cols += cfg.get("continuous_features", []) + cfg.get("categorical_features", [])
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"Metadata is missing configured columns: {missing}")
    mapping = cfg.get("label_mapping", {"benign": 0, "malignant": 1})
    if not pd.api.types.is_numeric_dtype(df[cfg["label_col"]]):
        df[cfg["label_col"]] = df[cfg["label_col"]].astype(str).str.lower().map(mapping)
    if df[cfg["label_col"]].isna().any():
        raise ValueError("Unmapped/missing labels found")
    return df


def create_patient_splits(df, patient_col, label_col, out_dir, seed=42,
                          train_fraction=.70, val_fraction=.15):
    """Stratify patients by their max lesion label and save JSON ID lists."""
    patients = df.groupby(patient_col)[label_col].max().reset_index()
    train, remainder = train_test_split(
        patients, train_size=train_fraction, random_state=seed,
        stratify=patients[label_col])
    rel_val = val_fraction / (1.0 - train_fraction)
    val, test = train_test_split(
        remainder, train_size=rel_val, random_state=seed,
        stratify=remainder[label_col])
    result = {"train": train[patient_col].tolist(),
              "val": val[patient_col].tolist(), "test": test[patient_col].tolist()}
    assert_no_patient_leakage(result)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, ids in result.items():
        (out / f"{name}.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    return result


def load_splits(split_dir):
    base = Path(split_dir)
    splits = {name: json.loads((base / f"{name}.json").read_text(encoding="utf-8"))
              for name in ("train", "val", "test")}
    assert_no_patient_leakage(splits)
    return splits


def assert_no_patient_leakage(splits):
    sets = {k: set(map(str, v)) for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"Patient leakage between {a}/{b}: {sorted(overlap)[:5]}"


def make_dataset(df, ids, cfg, training=False):
    patient_col, image_col, label_col = (cfg[k] for k in
                                          ("patient_col", "image_col", "label_col"))
    subset = df[df[patient_col].astype(str).isin(set(map(str, ids)))].copy()
    continuous = cfg.get("continuous_features", [])
    categorical = cfg.get("categorical_features", [])
    image_root = Path(cfg["image_root"])
    paths = np.asarray([str(image_root / str(p)) for p in subset[image_col]])
    clinical = {c: subset[c].fillna(subset[c].median()).astype("float32").to_numpy()[:, None]
                for c in continuous}
    clinical.update({c: subset[c].fillna("unknown").astype(str).to_numpy()[:, None]
                     for c in categorical})
    labels = subset[label_col].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(((paths, clinical), labels))
    if training:
        ds = ds.shuffle(len(subset), seed=cfg.get("seed", 42), reshuffle_each_iteration=True)
    size = tuple(cfg.get("image_size", [224, 224]))
    augmenter = build_image_augmentation() if training else None
    def load(inputs, label):
        path, features = inputs
        image = decode_resize(path, size)
        if augmenter is not None:
            image = augmenter(image, training=True)
        return (image, features), label
    # Keep memory bounded so multiple source-specific experiments can run on
    # ordinary research workstations without TensorFlow autotune over-allocation.
    ds = ds.map(load, num_parallel_calls=cfg.get("data_parallel_calls", 2))
    return ds.batch(cfg.get("batch_size", 16)).prefetch(cfg.get("prefetch_batches", 1))
