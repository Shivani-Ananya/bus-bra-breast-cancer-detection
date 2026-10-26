"""Create a concise comparison table from completed experiment metrics."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "BUS-BRA image-only": "busbra_image",
    "BUSI image-only": "busi_image",
    "Combined image-only": "combined_image",
    "BUS-BRA multimodal": "busbra_multimodal",
    "Synthetic-augmented multimodal": "synthetic_augmented",
}
KINDS = {
    "BUS-BRA image-only": "Image-only baseline",
    "BUSI image-only": "Image-only baseline",
    "Combined image-only": "Cross-source image model",
    "BUS-BRA multimodal": "Real clinical + image fusion",
    "Synthetic-augmented multimodal": "Synthetic training augmentation",
}


def main():
    rows, public = [], []
    for label, folder in EXPERIMENTS.items():
        path = ROOT / "runs" / "experiments_v3" / folder / "test_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": label,
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["recall_sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "mean_uncertainty": metrics["mean_uncertainty"],
        })
        by_source = metrics.get("by_source", {})
        public.append({
            "name": label,
            "kind": KINDS[label],
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["recall_sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "mean_uncertainty": metrics["mean_uncertainty"],
            "test_n": sum(sum(row) for row in metrics["confusion_matrix"]),
            "epochs": 5,
            "test_source": " + ".join(by_source) if by_source else "Configured test split",
            "sources": [{"name": name, "n": values.get("n"), "roc_auc": values.get("roc_auc")}
                        for name, values in by_source.items()],
        })
    if len(rows) != len(EXPERIMENTS):
        missing = set(EXPERIMENTS) - {row["experiment"] for row in rows}
        raise SystemExit(f"Missing completed evaluations: {sorted(missing)}")
    frame = pd.DataFrame(rows)
    output = ROOT / "runs" / "experiments_v3"; output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "comparison.csv", index=False)
    columns = list(frame.columns)
    markdown = "# BreastSight experiment comparison\n\n"
    markdown += "| " + " | ".join(columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    for row in frame.to_dict(orient="records"):
        markdown += "| " + " | ".join(f"{row[column]:.4f}" if isinstance(row[column], float) else str(row[column]) for column in columns) + " |\n"
    markdown += "\n\nSynthetic clinical attributes were used for training only; all validation and test rows for that experiment are real BUS-BRA cases.\n"
    (output / "comparison.md").write_text(markdown, encoding="utf-8")
    public_payload = {
        "status": "5 experiments completed",
        "disclaimer": "Five-epoch exploratory models at 128 x 128 pixels; not clinically validated and not for diagnosis.",
        "experiments": public,
        "interpretation": "Compare sensitivity and source-specific ROC-AUC, not accuracy alone. Synthetic clinical attributes were used only for training; their validation and test cohorts contain real BUS-BRA cases.",
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(public_payload, indent=2), encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
