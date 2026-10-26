reastSight Studio

BreastSight Studio is a multimodal breast-ultrasound research workbench for experimenting with benign–malignant classification from paired ultrasound images and clinical features. The project combines an EfficientNet image encoder with a clinical multilayer perceptron, produces Dirichlet evidence for both modalities, and uses differentiable Dempster–Shafer fusion to estimate a final class probability together with model uncertainty.

> **Research prototype only. This project is not a medical device and must not be used to diagnose, exclude, or treat disease.**

## Live application

- Public project interface: [bus-brabreastcancerdetection.vercel.app](https://bus-brabreastcancerdetection.vercel.app/)
- Full local research workbench: `streamlit_app.py`
- Vercel/FastAPI entrypoint: `app.py`

The Vercel interface is intentionally lightweight. It provides navigation, dataset selection and browser-side validation, image previews, clinical-input validation, model documentation, and responsible-use guidance. TensorFlow training and real model inference run in the local Streamlit workbench because long-running ML workloads and large artifacts are not suitable for the Vercel serverless runtime.

## Project team

- Shivani Ananya KP
- Ajisha Threase M
- Shruthi CS

## Features

### Public web interface

- Responsive white UI with blue and green accents
- Home, Dataset & Training, Ultrasound Analysis, Results, and Guide & About pages
- Drag-and-drop metadata and image selection
- CSV row and column inspection in the browser
- Ultrasound image preview
- Clinical-input form for age, BI-RADS, and lesion size
- Case-completeness validation
- Model architecture and responsible-use information
- Copyable local training and evaluation commands
- No fabricated predictions when a trained model is unavailable

Files selected on the public page remain in browser memory. They are not uploaded or permanently stored by the current Vercel application.

### Local Streamlit workbench

- CSV, XLSX, and XLS metadata ingestion
- Individual image and ZIP-folder upload
- Automatic normalization of common BUS-BRA column names
- Benign/malignant label validation
- Dataset readiness checks
- Leakage-safe patient-level train, validation, and test splitting
- Five configurable experiments
- Background model training with a visible log and stop control
- Saved checkpoints and best validation weights
- Ultrasound and clinical-input inference workspace
- Probability and uncertainty presentation
- Evaluation-artifact viewer
- Model, usage, and project documentation

### Machine-learning pipeline

- TensorFlow and Keras implementation
- EfficientNet-B0 ultrasound-image encoder
- Clinical preprocessing with normalization and categorical encoding
- Clinical multilayer perceptron
- Independent evidential heads for image and clinical modalities
- Differentiable Dempster–Shafer fusion
- Dirichlet uncertainty estimation
- Evidential deep supervision and KL-divergence annealing
- Global gradient clipping
- Patient-split leakage assertions
- Calibration and selective-prediction evaluation
- Grad-CAM utility for image explanations

## Architecture


```text
Ultrasound image ──► EfficientNet-B0 ──► Image evidence ────┐
                                                            ├─► Dempster fusion ─► Probability + uncertainty
Clinical features ─► Clinical MLP ─────► Clinical evidence ─┘
```

The application is delivered through two layers:

```text
Public web application
Vercel + FastAPI + browser-side interactions
                │
                ▼
Local research workbench
Streamlit + TensorFlow/Keras
                │
                ├── Dataset preparation
                ├── Patient-level splitting
                ├── Model training
                ├── Checkpoint storage
                ├── Inference
                └── Evaluation and explainability
```

## Technology stack

| Area | Technologies |
|---|---|
| Language | Python 3.11/3.12 |
| Public backend | FastAPI |
| Public deployment | Vercel |
| Research UI | Streamlit |
| Machine learning | TensorFlow and Keras |
| Image backbone | EfficientNet-B0 |
| Data processing | pandas and NumPy |
| Splitting and metrics | scikit-learn |
| Configuration | YAML |
| Visualization | Matplotlib and Seaborn |
| Testing | pytest |
| Planned persistence | Supabase PostgreSQL and private object storage |

## Repository structure

```text
breastcancer/
├── app.py                         # Lightweight FastAPI/Vercel application
├── streamlit_app.py               # Complete local research workbench
├── configs/                       # Experiment configurations
├── src/
│   ├── data/                      # Dataset and preprocessing pipeline
│   ├── losses/                    # Evidential loss
│   ├── models/                    # Encoders, heads, fusion, and model
│   ├── train.py                   # Training loop
│   ├── evaluate.py                # Evaluation and calibration
│   └── explain.py                 # Grad-CAM helper
├── tests/                         # Fusion, splitting, and web tests
├── notebooks/                     # Results-analysis starter notebook
├── docs/                          # Architecture and feature diagrams
├── requirements.txt               # Lightweight Vercel dependencies
├── requirements-local.txt         # Complete local ML dependencies
├── vercel.json                    # Vercel configuration
└── .env.example                   # Safe environment-variable template
```

These local directories are intentionally excluded from Git:

```text
dataset/       Original research datasets
data/          Prepared metadata, images, and patient splits
runs/          Model weights, checkpoints, and evaluation artifacts
.venv/         Local Python environment
.env.local     Private credentials
```

## Dataset contract

The default configurations expect metadata at `data/raw/metadata.csv` and images below `data/raw/images/`.

| Column | Required | Description |
|---|---:|---|
| `patient_id` | Yes | Stable de-identified identifier used for patient-level splitting |
| `image_path` | Yes | Filename or path relative to the configured image root |
| `diagnosis` | Yes | Binary label: `benign` or `malignant` |
| `age` | Configured | Numeric patient age |
| `lesion_size_mm` | Configured | Lesion size in millimetres |
| `birads` | Configured | Categorical BI-RADS value |

The upload workflow recognizes common aliases such as `CaseID`, `Image_filename`, `Classification`, `Age`, and `BIRADS`, and converts them to the internal contract.

Only de-identified data should be used. Do not include names, phone numbers, addresses, hospital record numbers, or other direct identifiers.

## Installation on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/YOUR_USERNAME/bus-bra-breast-cancer-detection.git
cd bus-bra-breast-cancer-detection
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Python 3.11 or 3.12 is recommended. If PowerShell prevents activation, invoke `.venv\Scripts\python.exe` directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

`requirements.txt` is intentionally small and is used by Vercel. Use `requirements-local.txt` for Streamlit, TensorFlow, testing, training, and evaluation.

### 4. Test the project

```powershell
python -m pytest -q
```

### 5. Start the workbench

```powershell
python -m streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8501` in a browser.

## Preparing the data

The easiest workflow is available in **Dataset & Training**:

1. Select a CSV, XLSX, or XLS metadata file.
2. Save it as the canonical metadata dataset.
3. Select individual ultrasound images or a ZIP containing the image folder.
4. Review the normalized columns and warnings.
5. Confirm that the expected images were found.
6. Allow the first training run to create patient-level splits.

For manual setup:

```text
data/
└── raw/
    ├── metadata.csv
    └── images/
        ├── case001.png
        ├── case002.png
        └── ...
```

## Experiments

| Configuration | Purpose |
|---|---|
| `baseline_clinical.yaml` | Clinical-only evidential baseline |
| `baseline_image.yaml` | Image-only evidential baseline |
| `fusion_naive_concat.yaml` | Conventional feature-concatenation baseline |
| `fusion_evidential.yaml` | Full two-branch evidential fusion model |
| `fusion_evidential_no_birads.yaml` | Ablation model without BI-RADS |

These configurations support comparisons between single-modality baselines, conventional fusion, evidential fusion, and the BI-RADS ablation.

## Training

Train the full evidential model with:

```powershell
python -m src.train --config configs/fusion_evidential.yaml
```

Additional experiments:

```powershell
python -m src.train --config configs/baseline_clinical.yaml
python -m src.train --config configs/baseline_image.yaml
python -m src.train --config configs/fusion_naive_concat.yaml
python -m src.train --config configs/fusion_evidential_no_birads.yaml
```

On the first run, the pipeline:

1. Loads and validates the metadata.
2. Groups samples by patient identifier.
3. Stratifies patients using their maximum lesion label.
4. Creates train, validation, and test ID lists.
5. Asserts that the patient sets are disjoint.
6. Builds clinical preprocessing layers using training patients only.
7. Builds the selected model.
8. Trains with evidential loss and KL annealing.
9. Saves periodic checkpoints.
10. Saves the lowest-validation-loss weights as `best.weights.h5`.

Outputs are written under `runs/`. ImageNet weights are downloaded on first use; set `pretrained: false` for an offline smoke test.

## Evaluation

```powershell
python -m src.evaluate --config configs/fusion_evidential.yaml
```

The evaluation pipeline produces or supports:

- Accuracy
- Sensitivity/recall
- Specificity
- F1 score
- ROC-AUC
- Expected calibration error
- Reliability bins and reliability plot
- Confusion matrix
- Case-level uncertainty
- Selective-accuracy points
- Subgroup-analysis inputs

Always evaluate on the untouched test split. Do not select hyperparameters or decision thresholds using test results.

## Explainability

`src.explain.gradcam` provides a Grad-CAM primitive for image heatmaps. Convolutional-layer names depend on the TensorFlow/Keras backbone version and should be inspected with:

```python
model.image_encoder.summary()
```

Grad-CAM indicates influential image regions; it does not establish causality or prove that a prediction is correct.

## Storage and retention

The project does not currently have a connected cloud database.

| Data | Current storage | Retention |
|---|---|---|
| Public-page file selections | Browser memory | Until refresh or tab closure |
| Original dataset | Local `dataset/` | Until manually deleted |
| Prepared training data | Local `data/` | Until manually deleted |
| Patient splits | Local `data/splits/` | Until manually deleted |
| Checkpoints and weights | Local `runs/` | Until manually deleted |
| Training logs | Local files | Until manually deleted |

The planned persistent design uses Supabase PostgreSQL for users, cases, studies, model versions, predictions, and audit records, plus private Supabase Storage buckets for ultrasound images and reports.

## Environment variables

Copy `.env.example` to `.env.local` after creating the Supabase project:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-publishable-key
SUPABASE_SERVICE_ROLE_KEY=your-secret-service-role-key
```

Never commit `.env.local`, and never expose the service-role key in browser JavaScript.

## Vercel deployment

The root `app.py` exports a top-level FastAPI `app` supported by the Vercel Python runtime.

```powershell
vercel deploy --prod
```

The deployment installs `requirements.txt`, not the TensorFlow dependency set. Dataset folders, model artifacts, training code, tests, and logs are excluded through `.vercelignore`.

When GitHub is connected to Vercel, pushes to the configured production branch can trigger automatic deployments.

## Security and privacy

- Keep the GitHub repository private during research development.
- Never commit datasets, medical images, credentials, reports, or sensitive model artifacts.
- Use de-identified patient IDs.
- Use private cloud-storage buckets.
- Store privileged credentials only in server-side environment variables.
- Add authentication and Row-Level Security before cloud persistence.
- Maintain an audit trail for uploads, predictions, exports, and deletion.
- Define a retention and deletion policy before collecting user data.
- Obtain institutional and ethical approval where required.
- Confirm applicable legal and regulatory obligations before processing medical data.

## Testing

```powershell
python -m pytest -q
```

Tests cover Dempster-fusion behavior, high-conflict stability, multi-sample output shape, patient-split leakage prevention, and FastAPI endpoints. New model or data-pipeline features should include regression tests.

## Known limitations

- The public Vercel interface does not perform TensorFlow training.
- The public deployment does not persist selected files.
- A trained checkpoint is required for genuine inference.
- Results require external validation on an independent cohort.
- Calibration and uncertainty do not equal clinical safety.
- Performance may vary across scanners, institutions, populations, and acquisition protocols.
- Placeholder or synthetic clinical fields must not be interpreted scientifically.

## Responsible use

BreastSight Studio is intended for research, education, reproducibility experiments, and interface development. It must not replace radiologist review, pathology, clinical assessment, or established care pathways.

Any publication or demonstration should disclose dataset origin and license, cohort construction, patient-level split methodology, missing-data handling, training procedure, calibration, subgroup performance, external-validation status, and study limitations.

## Contributing

1. Create a feature branch.
2. Keep datasets and credentials outside Git.
3. Add or update tests.
4. Run `python -m pytest -q`.
5. Document configuration changes.
6. Open a pull request describing the scientific and software impact.

Suggested branch names:

```text
feature/supabase-persistence
feature/model-inference-api
fix/dataset-validation
docs/evaluation-protocol
```

## License

No open-source license has been assigned. Until one is added, treat the repository as private, with all rights reserved by the project team. Dataset licenses remain separate from the application source and must be reviewed before redistribution.

## Acknowledgements

This project uses TensorFlow/Keras, Streamlit, FastAPI, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, pytest, and Vercel. Add dataset attribution and citation information according to the terms of the research dataset used in each experiment.
