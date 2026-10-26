"""BreastSight Studio — Streamlit research workbench."""
from pathlib import Path
import json
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import assert_no_patient_leakage, load_splits  # noqa: E402
from src.train import build_all  # noqa: E402

st.set_page_config(page_title="BreastSight Studio", page_icon="◉", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--navy:#07111f;--panel:#0e1c2f;--line:#213954;--teal:#22c7b8;--cyan:#38bdf8;
--muted:#93a4b8;--white:#edf6ff;--amber:#f59e0b;--green:#22c55e}
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:var(--navy);color:var(--white)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#091625,#0b1d30);border-right:1px solid var(--line)}
[data-testid="stSidebar"] .stRadio label{padding:.55rem .7rem;border-radius:10px;margin:.12rem 0}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:#15364a;border-left:3px solid var(--teal)}
.block-container{padding:2rem 2.4rem 4rem;max-width:1500px}.brand{padding:.4rem .25rem 1.4rem}
.brand-name{font-size:1.3rem;font-weight:800;color:#fff}.brand-dot{color:var(--teal)}
.brand-sub{font-size:.73rem;color:var(--muted);line-height:1.35;margin-top:.2rem}
.page-head{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.35rem}
.page-title{font-size:2rem;font-weight:800;line-height:1.15}.page-sub{color:var(--muted);margin-top:.35rem}
.pill{display:inline-flex;align-items:center;gap:.45rem;padding:.5rem .8rem;border:1px solid #25506a;
border-radius:999px;background:#0e2635;color:#bceff0;font-size:.78rem;font-weight:600}
.dot{height:8px;width:8px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green)}
.hero{padding:2rem;border:1px solid #1d4154;border-radius:22px;background:radial-gradient(circle at 90% 10%,#143f4c 0,#0e1c2f 38%);
box-shadow:0 18px 45px rgba(0,0,0,.22);margin-bottom:1.2rem}.eyebrow{font-size:.73rem;letter-spacing:.13em;
text-transform:uppercase;color:var(--teal);font-weight:700}.hero h2{font-size:2.25rem;margin:.45rem 0 .5rem}.hero p{color:#aec0d2;max-width:760px}
.card{padding:1.25rem;border:1px solid var(--line);border-radius:18px;background:var(--panel);height:100%}
.card-title{font-weight:700;font-size:1rem;margin-bottom:.25rem}.card-copy{font-size:.86rem;color:var(--muted);line-height:1.55}
.step{display:flex;gap:.9rem;padding:.9rem 0;border-bottom:1px solid #1b3047}.step:last-child{border:0}
.step-num{height:29px;min-width:29px;border-radius:9px;display:grid;place-items:center;background:#143f4c;color:#73eee2;font-weight:700}
.status-ok{color:#74e6a2}.status-warn{color:#f9c766}.tiny{font-size:.76rem;color:var(--muted)}
.disclaimer{padding:.85rem 1rem;border:1px solid #654c1a;background:#2b220f;border-radius:12px;color:#f6d890;font-size:.78rem}
.footer-note{margin-top:2rem;color:#60748a;font-size:.72rem;text-align:center}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);padding:1rem;border-radius:16px}
[data-testid="stFileUploader"]{background:var(--panel);border:1px dashed #31617a;border-radius:16px;padding:.5rem}
.stButton>button{border-radius:10px;border:1px solid #2a5369;font-weight:650}.stButton>button[kind="primary"]{background:#18a99e;border-color:#22c7b8}
.stTabs [data-baseweb="tab-list"]{gap:.35rem}.stTabs [data-baseweb="tab"]{border-radius:9px;padding:.35rem .85rem}
*:focus-visible{outline:2px solid var(--cyan)!important;outline-offset:2px}
@media(max-width:1100px){.block-container{padding:1.4rem}.page-title{font-size:1.6rem}.hero h2{font-size:1.75rem}}
/* Light clinical theme: white surfaces with blue and green accents. */
:root{--navy:#f4f8fc;--panel:#ffffff;--line:#d7e5f2;--teal:#16a36a;--cyan:#1683d8;
--muted:#5f7488;--white:#102a43;--amber:#c88712;--green:#16a36a}
.stApp{background:linear-gradient(135deg,#f7fbff,#eef8f4);color:#102a43}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);box-shadow:8px 0 30px rgba(20,74,115,.06)}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:#e7f3ff;border-left-color:#1683d8;color:#0a5792}
.brand-name{color:#123452}.brand-dot{color:#16a36a}.pill{border-color:#b9dccd;background:#eaf8f1;color:#19744f}
.hero{border-color:#c8dfef;background:radial-gradient(circle at 90% 10%,#dff6eb 0,#fff 42%);box-shadow:0 18px 50px rgba(32,91,129,.09)}
.hero p{color:#5f7488}.eyebrow{color:#1683d8}.card,[data-testid="stMetric"]{background:#fff;box-shadow:0 10px 30px rgba(35,83,116,.06)}
.step{border-bottom-color:#e2edf5}.step-num{background:#e5f6ee;color:#13845a}.status-ok{color:#13845a}.status-warn{color:#a66a00}
.disclaimer{border-color:#efd08a;background:#fff8e8;color:#79591b}.footer-note{color:#7890a4}
[data-testid="stFileUploader"]{background:#fff;border-color:#8bbfe2}.stButton>button{border-color:#86b7da;color:#15567f;background:#fff}.stButton>button[kind="primary"]{background:#1683d8;border-color:#1683d8;color:#fff}
</style>
""", unsafe_allow_html=True)


def resolve_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key in ("metadata_csv", "image_root", "split_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    cfg["training"]["output_dir"] = str(ROOT / cfg["training"]["output_dir"])
    return cfg


@st.cache_resource(show_spinner="Loading trained evidential model…")
def load_model(config_path: str, weights_timestamp: float):
    cfg = resolve_config(Path(config_path))
    model, datasets = build_all(cfg)
    model(next(iter(datasets["train"]))[0], training=False)
    model.load_weights(str(Path(cfg["training"]["output_dir"]) / "best.weights.h5"))
    return model, cfg


def page_header(title: str, subtitle: str, status: str = "Research workspace") -> None:
    st.markdown(f"""<div class="page-head"><div><div class="page-title">{title}</div>
    <div class="page-sub">{subtitle}</div></div><div class="pill"><span class="dot"></span>{status}</div></div>""",
                unsafe_allow_html=True)


def friendly_config_name(path: Path) -> str:
    names = {"baseline_clinical": "Clinical-only EDL", "baseline_image": "Image-only EDL",
             "fusion_naive_concat": "Naive multimodal fusion", "fusion_evidential": "Full evidential fusion",
             "fusion_evidential_no_birads": "Fusion without BI-RADS"}
    return names.get(path.stem, path.stem.replace("_", " ").title())


def readiness(cfg: dict[str, Any]) -> dict[str, Any]:
    metadata = Path(cfg["data"]["metadata_csv"])
    image_root = Path(cfg["data"]["image_root"])
    split_root = Path(cfg["data"]["split_dir"])
    weights = Path(cfg["training"]["output_dir"]) / "best.weights.h5"
    images = [p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}] if image_root.exists() else []
    split_ready = all((split_root / f"{name}.json").exists() for name in ("train", "val", "test"))
    return {"metadata": metadata, "image_root": image_root, "images": images,
            "splits": split_root, "split_ready": split_ready, "weights": weights}


def validate_metadata(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, ["Metadata CSV has not been added yet."]
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None, ["The metadata file could not be read as CSV."]
    required = [cfg["data"][key] for key in ("patient_col", "image_col", "label_col")]
    optional = cfg["data"].get("continuous_features", []) + cfg["data"].get("categorical_features", [])
    errors = [f"Required column '{name}' is missing." for name in required if name not in frame]
    errors += [f"Configured feature '{name}' is missing from the CSV." for name in optional if name not in frame]
    return frame, errors


ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_SINGLE_IMAGE_BYTES = 100 * 1024 * 1024


def save_metadata_upload(upload: Any, destination: Path) -> tuple[bool, str]:
    """Validate and save CSV/XLSX metadata as the canonical metadata.csv."""
    try:
        suffix = Path(upload.name).suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(upload)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(upload)
        else:
            return False, "Choose a CSV, XLSX, or XLS metadata file."
        if frame.empty:
            return False, "The uploaded metadata file contains no rows."
        # Normalize common BUS-BRA/BrEaST-Lesions headings to the internal contract.
        aliases = {
            "patient_id": ["patient_id", "patientid", "caseid", "case_id"],
            "image_path": ["image_path", "imagefilename", "image_filename", "filename", "image"],
            "diagnosis": ["classification", "class", "label", "diagnosis"],
            "age": ["age", "patient_age"],
            "lesion_size_mm": ["lesion_size_mm", "lesionsizemm", "lesion_size", "tumor_size_mm", "size_mm"],
            "birads": ["birads", "bi_rads", "bi-rads", "birads_category"],
        }
        simplified = {str(column).lower().replace(" ", "").replace("-", "_"): column for column in frame.columns}
        renamed: dict[str, str] = {}
        for canonical, candidates in aliases.items():
            for candidate in candidates:
                key = candidate.lower().replace(" ", "").replace("-", "_")
                if key in simplified:
                    renamed[simplified[key]] = canonical
                    break
        frame = frame.rename(columns=renamed)
        required = {"patient_id", "image_path", "diagnosis"}
        missing_required = sorted(required - set(frame.columns))
        if missing_required:
            return False, f"Missing required fields after column matching: {', '.join(missing_required)}."
        notes: list[str] = []
        frame["diagnosis"] = frame["diagnosis"].astype(str).str.strip().str.lower()
        valid_labels = {"benign", "malignant"}
        invalid_labels = sorted(set(frame["diagnosis"].dropna()) - valid_labels)
        if invalid_labels:
            return False, "Diagnosis/classification must contain only benign and malignant labels."
        if "age" in frame:
            frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
            median_age = frame["age"].median()
            frame["age"] = frame["age"].fillna(float(median_age) if pd.notna(median_age) else 0.0)
        else:
            frame["age"] = 0.0
            notes.append("Age was absent and was filled with 0; exclude it from scientific analysis if unavailable.")
        if "lesion_size_mm" in frame:
            frame["lesion_size_mm"] = pd.to_numeric(frame["lesion_size_mm"], errors="coerce")
            median_size = frame["lesion_size_mm"].median()
            frame["lesion_size_mm"] = frame["lesion_size_mm"].fillna(float(median_size) if pd.notna(median_size) else 0.0)
        else:
            frame["lesion_size_mm"] = 0.0
            notes.append("Lesion size was absent and was filled with 0 for configuration compatibility.")
        if "birads" not in frame:
            frame["birads"] = "unknown"
            notes.append("BI-RADS was absent and was encoded as unknown.")
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["image_path"] = frame["image_path"].map(lambda value: Path(str(value)).name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        message = f"Saved {len(frame):,} metadata rows as {destination.name}."
        if notes:
            message += " " + " ".join(notes)
        return True, message
    except Exception as exc:
        return False, f"The metadata file could not be processed: {exc}"


def save_image_uploads(uploads: list[Any], destination: Path) -> tuple[int, list[str]]:
    """Save individually selected images using safe basename-only filenames."""
    destination.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, []
    for upload in uploads:
        safe_name = Path(upload.name).name
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in ALLOWED_IMAGE_SUFFIXES:
            rejected.append(upload.name)
            continue
        (destination / safe_name).write_bytes(upload.getvalue())
        saved += 1
    return saved, rejected


def extract_image_archive(upload: Any, destination: Path) -> tuple[int, list[str]]:
    """Extract supported images from a ZIP without trusting member paths."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted, rejected = 0, []
    try:
        with zipfile.ZipFile(upload) as archive:
            image_members = [member for member in archive.infolist()
                             if not member.is_dir() and Path(member.filename).suffix.lower() in ALLOWED_IMAGE_SUFFIXES]
            if sum(member.file_size for member in image_members) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                return 0, ["The extracted archive would exceed the 2 GB safety limit."]
            for member in archive.infolist():
                if member.is_dir():
                    continue
                safe_name = Path(member.filename).name
                if not safe_name or Path(safe_name).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                    rejected.append(member.filename)
                    continue
                if member.file_size > MAX_SINGLE_IMAGE_BYTES:
                    rejected.append(member.filename)
                    continue
                target = destination / safe_name
                with archive.open(member) as source:
                    target.write_bytes(source.read())
                extracted += 1
    except zipfile.BadZipFile:
        return 0, ["The uploaded file is not a valid ZIP archive."]
    return extracted, rejected


CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))
if not CONFIGS:
    st.error("No experiment configurations were found in the configs folder.")
    st.stop()

st.sidebar.markdown("""<div class="brand"><div class="brand-name">BreastSight<span class="brand-dot"> Studio</span></div>
<div class="brand-sub">Multimodal Ultrasound<br/>Research Workbench</div></div>""", unsafe_allow_html=True)
page = st.sidebar.radio("Workspace", ["Home", "Dataset & Training", "Ultrasound Analysis", "Results", "Guide & About"], label_visibility="collapsed")
st.sidebar.markdown("---")
selected = st.sidebar.selectbox("Active experiment", CONFIGS, format_func=friendly_config_name)
cfg = resolve_config(selected)
state = readiness(cfg)
st.sidebar.caption(f"Config · {selected.name}")
st.sidebar.markdown("<div class='disclaimer'>Research prototype only.<br/>Not a medical device.</div>", unsafe_allow_html=True)

if page == "Home":
    page_header("Research overview", "Prepare data, train multimodal models, inspect uncertainty, and analyze results.")
    st.markdown("""<div class="hero"><div class="eyebrow">Multimodal evidential learning</div>
    <h2>From ultrasound data to trustworthy uncertainty</h2><p>BreastSight Studio combines an EfficientNet image branch and a clinical MLP, then uses Dirichlet evidence and Dempster fusion to produce a class prediction with case-level uncertainty.</p></div>""", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Metadata", "Ready" if state["metadata"].exists() else "Not added")
    b.metric("Ultrasound images", len(state["images"]))
    c.metric("Patient split", "Ready" if state["split_ready"] else "Not built")
    d.metric("Model weights", "Ready" if state["weights"].exists() else "Not trained")
    st.write("")
    for col, eyebrow, title, copy in zip(st.columns(3), ["01 · Prepare", "02 · Train", "03 · Analyze"],
        ["Dataset workspace", "Five controlled experiments", "Probability + uncertainty"],
        ["Validate metadata, inspect balance, confirm image paths, and create patient-safe splits.",
         "Compare clinical, image, naive fusion, evidential fusion, and no-BI-RADS ablation.",
         "Review fused and branch uncertainty, calibration, selective prediction, and Grad-CAM."]):
        with col:
            st.markdown(f"<div class='card'><div class='eyebrow'>{eyebrow}</div><div class='card-title'>{title}</div><div class='card-copy'>{copy}</div></div>", unsafe_allow_html=True)

elif page == "Dataset & Training":
    page_header("Dataset & training", "Prepare BUS-BRA inputs and launch a selected experiment.",
                "Ready" if state["metadata"].exists() and state["images"] else "Setup required")
    dataset_tab, train_tab = st.tabs(["Dataset preparation", "Training workspace"])
    with dataset_tab:
        left, right = st.columns([1.6, 1])
        with left:
            st.subheader("Upload training data")
            st.caption("Upload metadata first, then add individual ultrasound images or one ZIP containing the image folder.")

            metadata_upload = st.file_uploader(
                "1 · Metadata dataset",
                type=["csv", "xlsx", "xls"],
                help="The file is validated and saved internally as data/raw/metadata.csv.",
                key="metadata_training_upload",
            )
            if st.button("Save metadata", disabled=metadata_upload is None,
                         use_container_width=True, key="save_metadata_upload"):
                success, message = save_metadata_upload(metadata_upload, state["metadata"])
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

            st.markdown("---")
            image_mode = st.radio(
                "2 · Ultrasound images",
                ["Select individual images", "Upload a ZIP folder"],
                horizontal=True,
            )
            if image_mode == "Select individual images":
                image_uploads = st.file_uploader(
                    "Drop one or more image files",
                    type=["png", "jpg", "jpeg", "bmp"],
                    accept_multiple_files=True,
                    help="You can add images in multiple batches. Existing files with the same name are replaced.",
                    key="individual_training_images",
                )
                if st.button("Save selected images", disabled=not image_uploads,
                             use_container_width=True, key="save_individual_images"):
                    saved, rejected = save_image_uploads(image_uploads, state["image_root"])
                    if saved:
                        st.success(f"Saved {saved:,} ultrasound image files.")
                    if rejected:
                        st.warning(f"Skipped {len(rejected)} unsupported files.")
                    if saved:
                        st.rerun()
            else:
                archive_upload = st.file_uploader(
                    "Drop a ZIP containing the image folder",
                    type=["zip"],
                    help="Nested folders are accepted. Only PNG, JPG, JPEG, and BMP files are extracted.",
                    key="bulk_training_archive",
                )
                if st.button("Extract image folder", disabled=archive_upload is None,
                             use_container_width=True, key="extract_image_archive"):
                    extracted, rejected = extract_image_archive(archive_upload, state["image_root"])
                    if extracted:
                        st.success(f"Extracted {extracted:,} ultrasound images safely.")
                    if rejected:
                        st.warning(f"Skipped {len(rejected)} non-image or invalid archive entries.")
                    if extracted:
                        st.rerun()

            with st.expander("Stored data details"):
                st.code(f"Metadata: {state['metadata']}\nImages: {state['image_root']}", language="text")

            frame, issues = validate_metadata(state["metadata"], cfg)
            for issue in issues:
                st.warning(issue)
            if not issues:
                st.success("Metadata columns are valid for this configuration.")
            if frame is not None:
                st.dataframe(frame.head(25), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Readiness")
            checklist = [(state["metadata"].exists(), "Metadata file found"), (not issues, "Configured columns validated"),
                         (bool(state["images"]), f"{len(state['images'])} ultrasound images found"),
                         (state["split_ready"], "Patient-level split built")]
            for ok, label in checklist:
                st.markdown(f"<div class='step'><div class='step-num'>{'✓' if ok else '!'}</div><div class='{'status-ok' if ok else 'status-warn'}'>{label}</div></div>", unsafe_allow_html=True)
            if frame is not None and cfg["data"]["label_col"] in frame:
                st.subheader("Class balance")
                st.bar_chart(frame[cfg["data"]["label_col"]].astype(str).value_counts())
            if state["split_ready"]:
                try:
                    splits = load_splits(state["splits"]); assert_no_patient_leakage(splits)
                    st.success(f"No patient overlap · {len(splits['train'])} train / {len(splits['val'])} validation / {len(splits['test'])} test")
                except Exception:
                    st.error("Saved split files need to be rebuilt before training.")
            st.caption("Patient splits are created on the first training run and reused by all experiments.")
    with train_tab:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Experiment", friendly_config_name(selected)); t2.metric("Epochs", cfg["training"]["epochs"])
        t3.metric("Batch size", cfg["data"]["batch_size"]); t4.metric("KL annealing", f"{cfg['training'].get('kl_anneal_epochs', 10)} epochs")
        command = [sys.executable, "-m", "src.train", "--config", str(selected.relative_to(ROOT))]
        display_command = f"python -m src.train --config {selected.relative_to(ROOT)}"
        can_train = state["metadata"].exists() and bool(state["images"]) and not validate_metadata(state["metadata"], cfg)[1]
        if "training_process" not in st.session_state:
            st.session_state.training_process = None
        process = st.session_state.training_process
        running = process is not None and process.poll() is None
        start_col, stop_col, command_col = st.columns([1, 1, 3])
        if start_col.button("Start training", type="primary", disabled=not can_train or running, use_container_width=True):
            log_path = ROOT / "training-ui.log"; log_handle = log_path.open("w", encoding="utf-8")
            st.session_state.training_log_handle = log_handle
            st.session_state.training_process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            st.rerun()
        if stop_col.button("Stop", disabled=not running, use_container_width=True):
            process.terminate(); st.session_state.training_process = None
            handle = st.session_state.pop("training_log_handle", None)
            if handle: handle.close()
            st.rerun()
        command_col.code(display_command, language="text")
        if not can_train: st.info("Add valid metadata and ultrasound images before starting training.")
        elif running: st.success("Training is running in the background. Refresh to update the log.")
        elif process is not None and process.returncode == 0: st.success("Training completed. Open Results to inspect artifacts.")
        log_path = ROOT / "training-ui.log"
        st.subheader("Training log")
        st.code(log_path.read_text(encoding="utf-8", errors="replace")[-12000:] if log_path.exists() else "Training output will appear here.", language="text")

elif page == "Ultrasound Analysis":
    page_header("Ultrasound analysis", "Upload one scan and combine it with clinical context.",
                "Model ready" if state["weights"].exists() else "Weights required")
    st.markdown("<div class='disclaimer'>Decision-support research only. Do not use this output to diagnose or delay clinical care.</div>", unsafe_allow_html=True)
    st.write("")
    ready = state["metadata"].exists() and bool(state["images"]) and state["weights"].exists()
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Case input")
        upload = st.file_uploader("Breast ultrasound image", type=["png", "jpg", "jpeg", "bmp"], help="Use a de-identified research image.")
        image = None
        if upload:
            try: image = Image.open(upload).convert("RGB"); st.image(image, caption="Selected ultrasound image", use_container_width=True)
            except Exception: st.error("This file could not be opened as an image.")
        st.subheader("Clinical features")
        frame = pd.read_csv(state["metadata"]) if state["metadata"].exists() else None
        clinical: dict[str, Any] = {}; f1, f2 = st.columns(2)
        for index, name in enumerate(cfg["data"].get("continuous_features", [])):
            median = float(pd.to_numeric(frame[name], errors="coerce").median()) if frame is not None and name in frame else 0.0
            clinical[name] = (f1 if index % 2 == 0 else f2).number_input(name.replace("_", " ").title(), min_value=0.0, value=max(0.0, median))
        for index, name in enumerate(cfg["data"].get("categorical_features", [])):
            options = sorted(frame[name].dropna().astype(str).unique()) if frame is not None and name in frame else ["Unknown"]
            clinical[name] = (f1 if index % 2 == 0 else f2).selectbox(name.replace("_", " ").title(), options)
        analyze = st.button("Run evidential analysis", type="primary", disabled=image is None or not ready, use_container_width=True)
        if not ready: st.info("Train the selected experiment first. Analysis unlocks when compatible weights are available.")
    with right:
        st.subheader("Model output")
        if not analyze:
            st.markdown("<div class='card'><div class='eyebrow'>Awaiting analysis</div><div class='card-title'>Prediction results appear here</div><div class='card-copy'>You will see the predicted class, malignancy probability, fused uncertainty, and modality-level uncertainty.</div></div>", unsafe_allow_html=True)
    if analyze:
        try:
            model, loaded_cfg = load_model(str(selected), state["weights"].stat().st_mtime)
            height, width = loaded_cfg["data"].get("image_size", [224, 224]); array = np.asarray(image.resize((width, height)), np.float32)[None, ...]
            tensors = {name: tf.constant([[clinical[name]]], tf.float32) for name in loaded_cfg["data"].get("continuous_features", [])}
            tensors.update({name: tf.constant([[str(clinical[name])]], tf.string) for name in loaded_cfg["data"].get("categorical_features", [])})
            outputs = model((tf.constant(array), tensors), training=False); alpha = outputs["fused"]["alpha"].numpy()[0]
            probability = alpha / alpha.sum(); uncertainty = float(outputs["fused"]["uncertainty"].numpy()[0, 0])
            with right:
                st.metric("Prediction", "Malignant" if probability[1] >= .5 else "Benign")
                st.metric("Malignancy probability", f"{probability[1]:.1%}"); st.metric("Fused uncertainty", f"{uncertainty:.1%}")
                st.progress(float(probability[1]), text="Malignancy probability")
                if "image" in outputs and "clinical" in outputs:
                    st.progress(float(outputs["image"]["uncertainty"].numpy()[0, 0]), text="Image branch uncertainty")
                    st.progress(float(outputs["clinical"]["uncertainty"].numpy()[0, 0]), text="Clinical branch uncertainty")
                if uncertainty >= .35: st.warning("High uncertainty — refer this research case for specialist review.")
                elif uncertainty >= .20: st.info("Moderate uncertainty — interpret this model output cautiously.")
                else: st.success("The model reports comparatively low uncertainty.")
                st.caption(f"Model · {selected.stem}")
        except Exception:
            with right: st.error("Analysis could not be completed. Confirm that the selected weights match this configuration.")

elif page == "Results":
    page_header("Experiment results", "Review saved metrics, calibration, predictions, and high-uncertainty cases.")
    run_dir = Path(cfg["training"]["output_dir"]); metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        st.markdown("<div class='hero'><div class='eyebrow'>No finished evaluation</div><h2>Your results workspace is ready</h2><p>Train the selected experiment, then run evaluation. Metrics and figures will load from saved artifacts.</p></div>", unsafe_allow_html=True)
        st.code(f"python -m src.evaluate --config {selected.relative_to(ROOT)}", language="text")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        items = [("accuracy", "Accuracy"), ("recall_sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("ece", "ECE")]
        for col, (key, label) in zip(st.columns(6), items): col.metric(label, f"{metrics.get(key, 0):.3f}")
        overview, uncertainty_tab, cases = st.tabs(["Overview", "Uncertainty", "Cases"])
        with overview:
            reliability = run_dir / "reliability.png"
            if reliability.exists(): st.image(str(reliability), caption="Reliability diagram", width=650)
            st.json({k: v for k, v in metrics.items() if k not in {"reliability", "selective_prediction", "subgroups"}})
        with uncertainty_tab:
            selective = pd.DataFrame(metrics.get("selective_prediction", []))
            if not selective.empty: st.line_chart(selective.set_index("coverage")["accuracy"])
            st.json(metrics.get("uncertainty_quantiles", {}))
        with cases:
            predictions = run_dir / "test_predictions.csv"
            if predictions.exists(): st.dataframe(pd.read_csv(predictions).sort_values("uncertainty", ascending=False), use_container_width=True, hide_index=True)
            else: st.info("Per-case predictions have not been exported for this run.")

else:
    page_header("Guide, model & builders", "How to prepare data, train models, analyze cases, and understand the system.")
    guide_tab, model_tab, builder_tab = st.tabs(["How to use", "Model information", "Builder information"])
    with guide_tab:
        st.subheader("Recommended workflow")
        steps = [("Prepare BUS-BRA", "Place metadata.csv in data/raw and de-identified scans in data/raw/images."),
                 ("Validate the dataset", "Open Dataset & Training and resolve every amber readiness warning."),
                 ("Train baselines first", "Run clinical-only and image-only experiments before multimodal fusion."),
                 ("Train evidential fusion", "Run full fusion, then repeat without BI-RADS for leakage analysis."),
                 ("Evaluate", "Generate discrimination, calibration, uncertainty, subgroup, and per-case artifacts."),
                 ("Analyze a case", "Select matching weights, upload a scan, enter clinical features, and review uncertainty.")]
        for i, (title, copy) in enumerate(steps, 1):
            st.markdown(f"<div class='step'><div class='step-num'>{i}</div><div><b>{title}</b><div class='tiny'>{copy}</div></div></div>", unsafe_allow_html=True)
        st.subheader("Expected metadata")
        st.code("patient_id,image_path,diagnosis,age,lesion_size_mm,birads\nP001,P001_scan.png,benign,43,12.5,3", language="csv")
    with model_tab:
        st.subheader("Multimodal evidential architecture")
        c1, c2 = st.columns(2)
        with c1: st.markdown("<div class='card'><div class='eyebrow'>Image branch</div><div class='card-title'>EfficientNet-B0</div><div class='card-copy'>ImageNet-pretrained features feed a non-negative evidential head that defines a Dirichlet distribution.</div></div>", unsafe_allow_html=True)
        with c2: st.markdown("<div class='card'><div class='eyebrow'>Clinical branch</div><div class='card-title'>Normalized MLP</div><div class='card-copy'>Continuous fields are normalized, categorical fields are encoded, and an MLP produces clinical evidence.</div></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("<div class='card'><div class='eyebrow'>Fusion & training</div><div class='card-title'>Dempster combination + evidential loss</div><div class='card-copy'>Beliefs and uncertainty are combined with a protected Dempster rule. Training uses expected cross-entropy and annealed incorrect-class KL regularization.</div></div>", unsafe_allow_html=True)
        st.subheader("Experiment catalogue")
        st.dataframe(pd.DataFrame([["Clinical-only EDL", "Clinical MLP", "Tabular baseline"], ["Image-only EDL", "EfficientNet-B0", "Imaging baseline"], ["Naive concatenation", "Feature concatenation", "Simple fusion baseline"], ["Full evidential fusion", "Dual Dirichlet + Dempster", "Primary model"], ["Fusion without BI-RADS", "Dual Dirichlet + Dempster", "Leakage ablation"]], columns=["Experiment", "Architecture", "Purpose"]), use_container_width=True, hide_index=True)
        st.caption("References: Sensoy et al. (2018), Evidential Deep Learning; Han et al. (2021/2023), Trusted Multi-View Classification.")
    with builder_tab:
        st.subheader("BreastSight Studio")
        st.write("This project is a reproducible research workbench for multimodal breast-ultrasound classification. The UI is separated from the TensorFlow core so the data pipeline, fusion mathematics, training loop, and evaluation remain testable and reusable.")
        for col, eyebrow, title, copy in zip(st.columns(3), ["ML stack", "Application", "Quality"], ["TensorFlow 2.17 + Keras", "Python + Streamlit", "pytest + YAML"], ["tf.data, EfficientNet, custom evidential layers, GradientTape training.", "Dataset preparation, training, inference, results, and guidance.", "Fusion and leakage tests with reproducible experiment configs."]):
            with col: st.markdown(f"<div class='card'><div class='eyebrow'>{eyebrow}</div><div class='card-title'>{title}</div><div class='card-copy'>{copy}</div></div>", unsafe_allow_html=True)
        st.info("Team - Shivani Ananya KP, Ajisha Threase M, Shruthi CS")
        st.markdown("**Version:** 0.2 research prototype  \n**Dataset:** BUS-BRA")

st.markdown("<div class='footer-note'>BreastSight Studio · Research prototype · Uncertainty reflects model confidence, not clinical risk.</div>", unsafe_allow_html=True)
