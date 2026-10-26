"""Config-driven custom TensorFlow training loop."""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml
from src.data.dataset import load_metadata, create_patient_splits, load_splits, make_dataset
from src.data.preprocess import build_clinical_preprocessor
from src.models.image_encoder import build_image_encoder
from src.models.clinical_encoder import build_clinical_encoder
from src.models.model import BreastMultimodalModel
from src.losses.evidential_loss import evidential_loss


def load_config(path):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)


def build_all(cfg):
    data = cfg["data"]
    df = load_metadata(data["metadata_csv"], data)
    split_dir = data["split_dir"]
    if not all((Path(split_dir) / f"{x}.json").exists() for x in ("train", "val", "test")):
        splits = create_patient_splits(df, data["patient_col"], data["label_col"], split_dir,
                                       seed=data.get("seed", 42))
    else: splits = load_splits(split_dir)
    train_df = df[df[data["patient_col"]].astype(str).isin(set(map(str, splits["train"])))]
    prep = build_clinical_preprocessor(train_df, data.get("continuous_features", []),
                                       data.get("categorical_features", []))
    image_encoder = build_image_encoder(tuple(data.get("image_size", [224, 224])),
                                        cfg["model"].get("backbone", "EfficientNetB0"),
                                        cfg["model"].get("pretrained", True),
                                        cfg["model"].get("backbone_trainable", False))
    clinical_encoder = build_clinical_encoder(prep, tuple(cfg["model"].get("clinical_hidden", [64, 32])))
    model = BreastMultimodalModel(image_encoder, clinical_encoder, cfg["model"]["mode"])
    datasets = {k: make_dataset(df, ids, data, training=(k == "train")) for k, ids in splits.items()}
    return model, datasets


def run_epoch(model, ds, optimizer, annealing, train):
    loss_m, acc_m, unc_m = [tf.keras.metrics.Mean() for _ in range(3)]
    for inputs, labels in ds:
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=train)
            loss = evidential_loss(labels, outputs["fused"]["alpha"], annealing)
            # Deep supervision keeps both views evidentially meaningful.
            for key in ("image", "clinical"):
                if key in outputs:
                    loss += .25 * evidential_loss(labels, outputs[key]["alpha"], annealing)
            if model.losses: loss += tf.add_n(model.losses)
        if train:
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        probs = outputs["fused"]["alpha"] / tf.reduce_sum(outputs["fused"]["alpha"], -1, keepdims=True)
        loss_m.update_state(loss); acc_m.update_state(tf.reduce_mean(tf.cast(tf.argmax(probs, -1, output_type=tf.int32) == labels, tf.float32)))
        unc_m.update_state(tf.reduce_mean(outputs["fused"]["uncertainty"]))
    return {"loss": float(loss_m.result()), "accuracy": float(acc_m.result()), "uncertainty": float(unc_m.result())}


def main(config_path):
    cfg = load_config(config_path); tf.keras.utils.set_random_seed(cfg.get("seed", 42))
    model, datasets = build_all(cfg)
    # Create every variable before checkpoint restore. Keras 3 otherwise may
    # attach restored values to a partially built subclassed model.
    model(next(iter(datasets["train"]))[0], training=False)
    optimizer = tf.keras.optimizers.Adam(cfg["training"].get("learning_rate", 1e-4))
    out = Path(cfg["training"].get("output_dir", "runs/default")); out.mkdir(parents=True, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, str(out / "checkpoints"), max_to_keep=3)
    history_path = out / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    start_epoch = 0
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        start_epoch = int(manager.latest_checkpoint.rsplit("-", 1)[-1])
    best = min((row["val"]["loss"] for row in history), default=np.inf)
    epochs, anneal_epochs = cfg["training"]["epochs"], cfg["training"].get("kl_anneal_epochs", 10)
    for epoch in range(start_epoch, epochs):
        annealing = min(1.0, (epoch + 1) / max(1, anneal_epochs))
        train_log = run_epoch(model, datasets["train"], optimizer, annealing, True)
        val_log = run_epoch(model, datasets["val"], optimizer, 1.0, False)
        row = {"epoch": epoch + 1, "annealing": annealing, "train": train_log, "val": val_log}
        history.append(row); print(json.dumps(row))
        manager.save(checkpoint_number=epoch + 1)
        if val_log["loss"] < best:
            best = val_log["loss"]; model.save_weights(str(out / "best.weights.h5"))
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    main(p.parse_args().config)

