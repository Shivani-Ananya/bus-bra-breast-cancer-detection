import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.train import load_config, build_all
from src.data.dataset import load_metadata, load_splits


def expected_calibration_error(y, p, bins=10):
    confidence, prediction = p.max(1), p.argmax(1); edges = np.linspace(0, 1, bins + 1); ece = 0.
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            acc, conf = (prediction[mask] == y[mask]).mean(), confidence[mask].mean()
            ece += mask.mean() * abs(acc - conf)
            rows.append((float(conf), float(acc), int(mask.sum())))
    return float(ece), rows


def evaluate_arrays(y, p, uncertainty):
    pred = p.argmax(1); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ece, reliability = expected_calibration_error(y, p)
    coverages = np.linspace(1., .1, 10); order = np.argsort(uncertainty)
    selective = [{"coverage": float(c), "accuracy": float(accuracy_score(y[order[:max(1, int(len(y)*c))]], pred[order[:max(1, int(len(y)*c))]]))} for c in coverages]
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(y, pred, zero_division=0)), "specificity": float(tn / max(1, tn + fp)),
            "f1": float(f1_score(y, pred, zero_division=0)), "roc_auc": float(roc_auc_score(y, p[:, 1])),
            "ece": ece, "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "mean_uncertainty": float(np.mean(uncertainty)),
            "uncertainty_quantiles": {str(q): float(np.quantile(uncertainty, q)) for q in (.1, .25, .5, .75, .9)},
            "reliability": reliability,
            "selective_prediction": selective}


def subgroup_report(frame, y, p, u, columns):
    report = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        groups = (pd.qcut(values, q=min(4, values.nunique()), duplicates="drop").astype(str)
                  if np.issubdtype(values.dtype, np.number)
                  else values.fillna("unknown").astype(str))
        report[column] = {}
        for group in sorted(groups.unique()):
            mask = groups.to_numpy() == group
            if mask.sum():
                report[column][group] = {
                    "n": int(mask.sum()),
                    "accuracy": float(accuracy_score(y[mask], p[mask].argmax(1))),
                    "mean_uncertainty": float(u[mask].mean())}
    return report


def source_report(frame, y, p, u, column="source"):
    """Calculate complete metrics independently for every dataset source."""
    if column not in frame.columns:
        return {}
    report = {}
    for source in sorted(frame[column].fillna("unknown").astype(str).unique()):
        mask = frame[column].fillna("unknown").astype(str).to_numpy() == source
        labels = y[mask]
        probabilities = p[mask]
        uncertainty = u[mask]
        if len(np.unique(labels)) < 2:
            report[source] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(labels, probabilities.argmax(1))),
                "roc_auc": None,
                "mean_uncertainty": float(uncertainty.mean()),
            }
        else:
            report[source] = {"n": int(mask.sum()), **evaluate_arrays(labels, probabilities, uncertainty)}
    return report


def main(config_path):
    cfg = load_config(config_path); model, ds = build_all(cfg)
    model(next(iter(ds["train"]))[0])
    model.load_weights(str(Path(cfg["training"]["output_dir"]) / "best.weights.h5"))
    ys, ps, us = [], [], []
    for inputs, labels in ds["test"]:
        out = model(inputs, training=False)["fused"]
        alpha = out["alpha"].numpy(); probs = alpha / alpha.sum(-1, keepdims=True)
        ys.append(labels.numpy()); ps.append(probs); us.append(out["uncertainty"].numpy().ravel())
    y, p, u = np.concatenate(ys), np.concatenate(ps), np.concatenate(us)
    metrics = evaluate_arrays(y, p, u); out_dir = Path(cfg["training"]["output_dir"])
    data_cfg = cfg["data"]; frame = load_metadata(data_cfg["metadata_csv"], data_cfg)
    test_ids = set(map(str, load_splits(data_cfg["split_dir"])["test"]))
    frame = frame[frame[data_cfg["patient_col"]].astype(str).isin(test_ids)].copy().reset_index(drop=True)
    if len(frame) == len(y):
        metrics["subgroups"] = subgroup_report(frame, y, p, u, data_cfg.get("subgroup_features", ["age"]))
        metrics["by_source"] = source_report(frame, y, p, u, data_cfg.get("source_col", "source"))
        frame["true_label"], frame["predicted_label"] = y, p.argmax(1)
        frame["prob_malignant"], frame["uncertainty"] = p[:, 1], u
        frame.to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rel = metrics["reliability"]; plt.figure(); plt.plot([0,1],[0,1],"--")
    if rel: plt.plot([x[0] for x in rel], [x[1] for x in rel], marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.tight_layout(); plt.savefig(out_dir / "reliability.png", dpi=160); plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", required=True); main(p.parse_args().config)
