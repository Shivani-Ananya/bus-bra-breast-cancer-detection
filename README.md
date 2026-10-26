# BUS-BRA multimodal evidential classification

TensorFlow/Keras research scaffold for benign–malignant classification from paired ultrasound and clinical data. Each view emits Dirichlet evidence; the full model combines subjective opinions with differentiable Dempster fusion and reports per-case uncertainty.

## Data contract

Place images below `data/raw/images/` and a CSV at `data/raw/metadata.csv`. The default configs expect one row per image/lesion and these columns:

| column | meaning |
|---|---|
| `patient_id` | stable patient identifier used for leakage-safe splitting |
| `image_path` | path relative to `image_root` |
| `diagnosis` | `benign` or `malignant` |
| `age` | numeric age |
| `lesion_size_mm` | numeric lesion size |
| `birads` | categorical BI-RADS value |

BUS-BRA distributions can use different headings; edit the column names and feature lists in the YAML rather than changing Python. Add any available density or other metadata to `continuous_features` or `categorical_features`. The no-BI-RADS config deliberately removes that field.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
pytest -q
streamlit run streamlit_app.py
python -m src.train --config configs/baseline_clinical.yaml
python -m src.train --config configs/baseline_image.yaml
python -m src.train --config configs/fusion_naive_concat.yaml
python -m src.train --config configs/fusion_evidential.yaml
python -m src.train --config configs/fusion_evidential_no_birads.yaml
python -m src.evaluate --config configs/fusion_evidential.yaml
```

The full TensorFlow workbench is `streamlit_app.py`. The root `app.py` is a
lightweight FastAPI entrypoint for Vercel and exports the required ASGI `app`
variable. Vercel installs `requirements.txt`; local ML development installs
`requirements-local.txt`.

The first run creates persistent patient-level split JSON files and asserts disjointness. All later runs reuse exactly those patients. Each epoch logs loss, accuracy, mean uncertainty, and KL annealing weight; checkpoints are retained and the lowest-validation-loss weights are saved.

## Outputs and analysis

`src.evaluate` writes standard metrics, specificity, ROC-AUC, ECE, reliability bins, uncertainty, the confusion matrix, and selective-accuracy points, plus a reliability plot. Compare the two fusion run JSON files for accuracy and uncertainty shifts caused by BI-RADS. Use the notebook as a starting table, and join saved predictions to metadata for age/density subgroup analysis.

`src.explain.gradcam` provides a GradientTape heatmap primitive. Run the same case through the image-only and fused checkpoints and place the resulting maps side by side; prioritize the largest fused-uncertainty cases. Conv-layer names are backbone/version dependent and should be inspected with `model.image_encoder.summary()`.

## Notes

- Patient labels used for stratification are the maximum lesion label for that patient; splitting never occurs at image level.
- Dempster fusion clips the near-total-conflict denominator and renormalizes subjective masses to prevent NaNs.
- The loss uses expected cross-entropy plus annealed incorrect-class Dirichlet KL, with gradient clipping.
- ImageNet weights require network access on first use. Set `pretrained: false` for an offline smoke test.
- This is research software, not a clinical device. Validate labels, cohort construction, calibration, and external generalization before drawing medical conclusions.
