"""Prepare leakage-safe BUS-BRA, BUSI, combined, and synthetic experiments."""
from pathlib import Path
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments"
SEED = 42


def split_frame(frame, out_dir, stratify_source=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    strata = frame["diagnosis"]
    if stratify_source:
        strata = frame["source"].astype(str) + "_" + frame["diagnosis"].astype(str)
    train, remainder = train_test_split(frame, train_size=.70, random_state=SEED, stratify=strata)
    remainder_strata = remainder["diagnosis"]
    if stratify_source:
        remainder_strata = remainder["source"].astype(str) + "_" + remainder["diagnosis"].astype(str)
    val, test = train_test_split(remainder, train_size=.50, random_state=SEED, stratify=remainder_strata)
    result = {"train": train, "val": val, "test": test}
    for name, part in result.items():
        (out_dir / f"{name}.json").write_text(json.dumps(part.patient_id.tolist(), indent=2), encoding="utf-8")
    return result


def bus_bra():
    frame = pd.read_csv(ROOT / "dataset" / "clinical data" / "bus_bra_metadata.csv")
    rename = {"ID": "patient_id", "Image_filename": "image_path", "Classification": "diagnosis",
              "Age": "age", "BI-RADS": "birads", "Lesion_size": "lesion_size_mm"}
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    # The project's canonical file already contains normalized headings and the validated binary subset.
    canonical = pd.read_csv(ROOT / "data" / "raw" / "metadata.csv")
    canonical["patient_id"] = "busbra_" + canonical["patient_id"].astype(str)
    canonical["image_path"] = "dataset/images/BUS-BRA/" + canonical["image_path"].astype(str)
    canonical["source"] = "BUS-BRA"
    canonical["tissue_h"] = canonical["Tissue_composition"].fillna("unknown").astype(str).str.lower()
    canonical["shape_h"] = canonical["Shape"].fillna("unknown").astype(str).str.lower().replace({"oval": "oval_round", "round": "oval_round"})
    canonical["margin_h"] = canonical["Margin"].fillna("unknown").astype(str).str.lower().map(lambda value: "circumscribed" if value == "circumscribed" else "not_circumscribed")
    canonical["posterior_h"] = canonical["Posterior_features"].fillna("unknown").astype(str).str.lower().str.replace(" ", "_", regex=False)
    return canonical


def busi_images():
    rows = []
    for diagnosis in ("benign", "malignant"):
        for path in sorted((ROOT / "dataset" / "images" / "BUSI" / diagnosis).glob("*.png")):
            rows.append({
                "patient_id": f"busi_{diagnosis}_{re.sub(r'[^0-9]', '', path.stem)}",
                "image_path": path.relative_to(ROOT).as_posix(),
                "diagnosis": diagnosis,
                "age": 0.0,
                "lesion_size_mm": 0.0,
                "birads": "unknown",
                "source": "BUSI",
            })
    return pd.DataFrame(rows)


def synthetic_busi(sheet="synthetic_fold0"):
    path = ROOT / "dataset" / "clinical data" / "synthetic_clinical_data (1).xlsx"
    frame = pd.read_excel(path, sheet_name=sheet)
    frame = frame[frame.label.isin(["benign", "malignant"])].copy()
    frame["patient_id"] = frame.busi_image.map(lambda value: "busi_" + re.sub(r"[^a-zA-Z0-9]", "_", value).strip("_"))
    frame["image_path"] = "dataset/images/BUSI/" + frame.busi_image.astype(str)
    frame["diagnosis"] = frame.label.str.lower()
    frame["source"] = "BUSI-synthetic-clinical"
    frame["tissue_h"] = frame.tissue_composition.fillna("unknown").astype(str).str.lower()
    frame["shape_h"] = frame["shape"].fillna("unknown").astype(str).str.lower()
    frame["margin_h"] = frame.margin.fillna("unknown").astype(str).str.lower()
    frame["posterior_h"] = frame.posterior.fillna("unknown").astype(str).str.lower()
    return frame


def save(frame, name, stratify_source=False):
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "metadata.csv", index=False)
    split_frame(frame, directory / "splits", stratify_source=stratify_source)


def main():
    bra, busi = bus_bra(), busi_images()
    save(bra, "busbra_image")
    save(busi, "busi_image")
    save(pd.concat([bra, busi], ignore_index=True, sort=False).fillna({"age": 0, "lesion_size_mm": 0, "birads": "unknown"}),
         "combined_image", stratify_source=True)
    save(bra, "busbra_multimodal")

    # Synthetic clinical values augment training only. Validation and testing stay real BUS-BRA.
    real_splits = split_frame(bra, OUT / "synthetic_augmented" / "splits-real")
    synthetic = synthetic_busi("synthetic_fold0")
    train = pd.concat([real_splits["train"], synthetic], ignore_index=True, sort=False)
    complete = pd.concat([train, real_splits["val"], real_splits["test"]], ignore_index=True, sort=False)
    directory = OUT / "synthetic_augmented"; directory.mkdir(parents=True, exist_ok=True)
    complete.to_csv(directory / "metadata.csv", index=False)
    split_dir = directory / "splits"; split_dir.mkdir(exist_ok=True)
    ids = {"train": train.patient_id.tolist(), "val": real_splits["val"].patient_id.tolist(), "test": real_splits["test"].patient_id.tolist()}
    for name, values in ids.items():
        (split_dir / f"{name}.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps({
        "busbra_binary": len(bra), "busi_binary": len(busi), "combined_binary": len(bra) + len(busi),
        "synthetic_training_rows": len(synthetic), "synthetic_validation_rows": 0, "synthetic_test_rows": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
