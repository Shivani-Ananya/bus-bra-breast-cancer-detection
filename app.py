"""Vercel-compatible FastAPI entrypoint for BreastSight Studio.

The full TensorFlow workbench remains in ``streamlit_app.py`` and is run
locally or on a container host. This lightweight application provides the
public project site without importing TensorFlow into a serverless function.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="BreastSight Studio",
    description="Public project interface for multimodal breast-ultrasound research.",
    version="0.3.0",
)


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BreastSight Studio</title>
<style>
:root{--bg:#06101d;--side:#091827;--panel:#0d1b2d;--panel2:#11243a;--line:#203a55;--teal:#24d0c1;--red:#ff5a63;--text:#f4f8fc;--muted:#9fb1c5;--amber:#f4b942;--green:#39d98a}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}
button,input,select{font:inherit}.shell{display:grid;grid-template-columns:280px 1fr;min-height:100vh}.side{position:sticky;top:0;height:100vh;padding:30px 20px;border-right:1px solid var(--line);background:var(--side)}
.brand{font-size:23px;font-weight:850}.brand span{color:var(--teal)}.sub{color:#79a5c9;font-size:12px;margin:3px 4px 34px}.nav{display:grid;gap:7px}
.nav button{width:100%;border:0;background:transparent;color:#d5e2ee;text-align:left;padding:12px 13px;border-radius:11px;cursor:pointer}.nav button:hover{background:#102b40}.nav button.active{background:#173b50;border-left:3px solid var(--teal);color:#fff}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#445268;margin-right:11px}.active .dot{background:var(--red)}.warning{position:absolute;left:20px;right:20px;bottom:28px;padding:13px 15px;border:1px solid #6e521a;border-radius:13px;background:#2b210d;color:#ffe08a;font-size:12px}
main{padding:42px 48px;max-width:1360px;width:100%}.view{display:none;animation:fade .18s ease}.view.active{display:block}@keyframes fade{from{opacity:.5;transform:translateY(4px)}}
.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.top h1{font-family:Georgia,serif;font-size:36px;line-height:1.2;margin:0}.top p,.muted{color:var(--muted)}.top p{margin:5px 0}.pill{border:1px solid #28617a;border-radius:99px;padding:8px 13px;color:#b9f5ef;background:#0c2a39;font-size:12px;white-space:nowrap}
.hero{margin-top:26px;padding:38px;border:1px solid #1e485c;border-radius:22px;background:radial-gradient(circle at 90% 10%,#164957,#0d1b2d 43%)}.hero h2{font-size:39px;line-height:1.16;margin:9px 0 12px}.hero p{color:#b5c7d7;max-width:790px}.eyebrow{color:var(--teal);font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:17px;margin-top:20px}.grid.two{grid-template-columns:repeat(2,1fr)}.card{padding:22px;border:1px solid var(--line);border-radius:18px;background:var(--panel)}.card h3{margin:7px 0 8px}.card p{color:var(--muted);margin:0}.clickable{cursor:pointer;transition:.15s}.clickable:hover{transform:translateY(-2px);border-color:#34718b}
.drop{display:block;margin-top:14px;padding:30px 18px;border:1.5px dashed #3a6682;border-radius:15px;text-align:center;background:#091725;cursor:pointer}.drop:hover,.drop.drag{border-color:var(--teal);background:#0c2432}.drop input{display:none}.drop strong{display:block;font-size:16px}.drop small{color:var(--muted)}
.file-list{margin-top:12px;color:#b9cad9;font-size:13px;max-height:150px;overflow:auto}.file{display:flex;justify-content:space-between;padding:7px 9px;border-bottom:1px solid #1b334b}.ok{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:17px}.btn{border:1px solid #31627a;border-radius:10px;padding:10px 16px;background:#10283a;color:#fff;cursor:pointer}.btn:hover{filter:brightness(1.2)}.btn.primary{background:var(--teal);border-color:var(--teal);color:#06201d;font-weight:800}.btn:disabled{opacity:.42;cursor:not-allowed}.status{margin-top:18px;padding:15px 17px;border-radius:13px;border:1px solid #24546b;background:#0a2231;color:#bdd0df}.status.warn{border-color:#70561d;background:#2a210e;color:#ffdf89}
.field{display:grid;gap:6px;margin-top:13px}.field label{color:#a9bfd2;font-size:13px}.field input,.field select{width:100%;border:1px solid #29445e;border-radius:10px;background:#081421;color:#fff;padding:11px}.preview{margin-top:14px;min-height:250px;border-radius:16px;background:#07111c;display:grid;place-items:center;overflow:hidden;border:1px solid var(--line)}.preview img{max-width:100%;max-height:420px}.code{margin-top:12px;padding:14px 16px;border-radius:10px;background:#070d16;color:#a5eee7;font-family:ui-monospace,monospace;overflow:auto}.steps{counter-reset:s}.step{position:relative;padding:0 0 22px 45px}.step:before{counter-increment:s;content:counter(s);position:absolute;left:0;top:0;width:29px;height:29px;display:grid;place-items:center;border-radius:50%;background:#163c4c;color:var(--teal);font-weight:800}.step h3{margin:0 0 4px}.team{color:#fff!important;line-height:1.8!important}
@media(max-width:900px){.shell{display:block}.side{position:relative;height:auto}.warning{position:static;margin-top:20px}.nav{grid-template-columns:repeat(2,1fr)}main{padding:25px}.grid,.grid.two{grid-template-columns:1fr}.hero h2{font-size:29px}.top{display:block}.pill{display:inline-block;margin-top:10px}}
/* Light clinical theme: white surfaces with blue and green accents. */
:root{--bg:#f4f8fc;--side:#ffffff;--panel:#ffffff;--panel2:#eef6ff;--line:#d7e5f2;--teal:#16a36a;--red:#1683d8;--text:#102a43;--muted:#5f7488;--amber:#c88712;--green:#16a36a}
body{background:linear-gradient(135deg,#f7fbff,#eef8f4);color:var(--text)}.side{background:#fff;box-shadow:8px 0 30px rgba(20,74,115,.06)}
.brand{color:#123452}.brand span{color:#16a36a}.sub{color:#64819a}.nav button{color:#31516b}.nav button:hover{background:#edf6ff}.nav button.active{background:#e7f3ff;border-left-color:#1683d8;color:#0a5792}.dot{background:#b9ccdc}.active .dot{background:#16a36a}
.warning,.status.warn{background:#fff8e8;border-color:#efd08a;color:#79591b}main{background:transparent}.pill{border-color:#b9dccd;background:#eaf8f1;color:#19744f}.hero{border-color:#c8dfef;background:radial-gradient(circle at 90% 10%,#dff6eb,#fff 44%);box-shadow:0 18px 50px rgba(32,91,129,.09)}.hero p,.card p{color:var(--muted)}
.eyebrow{color:#1683d8}.card{background:#fff;box-shadow:0 10px 30px rgba(35,83,116,.06)}.clickable:hover{border-color:#5ba8df;box-shadow:0 14px 34px rgba(22,131,216,.11)}.drop{background:#f8fbfe;border-color:#8bbfe2}.drop:hover,.drop.drag{background:#edf8f3;border-color:#16a36a}.file{border-bottom-color:#e2edf5}.btn{background:#fff;border-color:#86b7da;color:#15567f}.btn.primary{background:#1683d8;border-color:#1683d8;color:#fff}.status{background:#eef8ff;border-color:#b9daef;color:#365c75}.field input,.field select{background:#fff;border-color:#c9dbe8;color:#16364e}.preview{background:#f7fafc}.code{background:#eaf3fa;color:#175a82}.step:before{background:#e5f6ee;color:#13845a}
</style></head><body><div class="shell">
<aside class="side"><div class="brand">BreastSight <span>Studio</span></div><div class="sub">Multimodal Ultrasound Research Workbench</div>
<nav class="nav"><button data-view="home" class="active"><span class="dot"></span>Home</button><button data-view="dataset"><span class="dot"></span>Dataset & Training</button><button data-view="analysis"><span class="dot"></span>Ultrasound Analysis</button><button data-view="results"><span class="dot"></span>Results</button><button data-view="guide"><span class="dot"></span>Guide & About</button></nav>
<div class="warning">Research prototype only.<br>Not a medical device.</div></aside><main>

<section class="view active" id="home"><div class="top"><div><h1>Breast-ultrasound research,<br>made inspectable.</h1><p>Prepare data, examine model design, and preview de-identified scans.</p></div><div class="pill">● Online workspace</div></div>
<div class="hero"><div class="eyebrow">Multimodal evidential learning</div><h2>From ultrasound data to trustworthy uncertainty</h2><p>An EfficientNet image branch and clinical MLP produce independent Dirichlet evidence. Dempster fusion combines their beliefs into a benign/malignant output with case-level and modality-level uncertainty.</p><div class="actions"><button class="btn primary" data-go="dataset">Prepare dataset</button><button class="btn" data-go="analysis">Open analysis</button></div></div>
<div class="grid"><article class="card clickable" data-go="dataset"><div class="eyebrow">01 · Prepare</div><h3>Dataset & training</h3><p>Drop metadata and ultrasound images, validate inputs, and prepare the local training command.</p></article><article class="card clickable" data-go="analysis"><div class="eyebrow">02 · Analyze</div><h3>Ultrasound workbench</h3><p>Preview a de-identified image and enter clinical features for the inference workflow.</p></article><article class="card clickable" data-go="guide"><div class="eyebrow">03 · Understand</div><h3>Model & guidance</h3><p>Review architecture, fusion, responsible use, and project information.</p></article></div></section>

<section class="view" id="dataset"><div class="top"><div><h1>Dataset & training</h1><p>Add files directly—no computer locations are displayed or required.</p></div><div class="pill" id="dataset-pill">Setup required</div></div>
<div class="grid two"><article class="card"><div class="eyebrow">Metadata</div><h3>Clinical dataset</h3><label class="drop" id="meta-drop">⬆<strong>Drop CSV here</strong><small>or click to select one file</small><input id="meta-file" type="file" accept=".csv,text/csv"></label><div class="file-list" id="meta-info">No metadata selected.</div></article>
<article class="card"><div class="eyebrow">Ultrasound scans</div><h3>Image files</h3><label class="drop" id="image-drop">▧<strong>Drop images here</strong><small>PNG, JPG, JPEG, WEBP — select multiple files</small><input id="image-files" type="file" accept="image/*" multiple></label><div class="file-list" id="image-info">No images selected.</div></article></div>
<article class="card" style="margin-top:17px"><div class="eyebrow">Readiness</div><h3>Input validation</h3><div id="readiness" class="muted">Add a metadata CSV and at least one image to continue.</div><div class="actions"><button class="btn primary" id="prepare" disabled>Prepare training</button><button class="btn" id="clear-files">Clear files</button></div><div id="training-output"></div></article></section>

<section class="view" id="analysis"><div class="top"><div><h1>Ultrasound analysis</h1><p>Preview a de-identified scan and prepare its clinical context.</p></div><div class="pill">No model checkpoint online</div></div>
<div class="grid two"><article class="card"><div class="eyebrow">Image input</div><h3>Select one ultrasound image</h3><label class="drop">▧<strong>Drop a scan here</strong><small>PNG, JPG, JPEG, or WEBP</small><input id="scan-file" type="file" accept="image/*"></label><div class="preview" id="preview"><span class="muted">Image preview appears here</span></div></article>
<article class="card"><div class="eyebrow">Clinical context</div><h3>Patient features</h3><div class="field"><label>Age</label><input id="age" type="number" min="0" max="120" placeholder="Example: 45"></div><div class="field"><label>BI-RADS category</label><select id="birads"><option value="">Select category</option><option>2</option><option>3</option><option>4A</option><option>4B</option><option>4C</option><option>5</option></select></div><div class="field"><label>Lesion size (mm)</label><input id="size" type="number" min="0" step="0.1" placeholder="Example: 16.5"></div><div class="actions"><button class="btn primary" id="validate-case">Validate case</button><button class="btn" id="reset-case">Reset</button></div><div id="case-status"></div></article></div>
<div class="status warn"><b>Why prediction is unavailable here:</b> this Vercel interface does not contain a trained TensorFlow checkpoint. It will never invent a medical result. Run the full local workbench after training a model.</div></section>

<section class="view" id="results"><div class="top"><div><h1>Results</h1><p>Evaluation artifacts appear after a completed local training run.</p></div><div class="pill">Awaiting model run</div></div><div class="hero"><div class="eyebrow">Evaluation workspace</div><h2>No fabricated metrics</h2><p>The online interface deliberately does not show placeholder accuracy or diagnostic values. Train and evaluate the selected experiment locally, then review accuracy, sensitivity, specificity, F1, ROC-AUC, calibration, uncertainty, and subgroup analyses.</p><div class="code">python -m src.evaluate --config configs/baseline_clinical.yaml</div><div class="actions"><button class="btn" data-copy="python -m src.evaluate --config configs/baseline_clinical.yaml">Copy evaluation command</button></div></div></section>

<section class="view" id="guide"><div class="top"><div><h1>Guide & about</h1><p>How to use the research prototype safely and reproducibly.</p></div><div class="pill">Version 0.3</div></div><div class="grid two"><article class="card"><div class="eyebrow">Quick start</div><div class="step"><h3>Prepare inputs</h3><p>Open Dataset & Training, add a metadata CSV, and add ultrasound image files.</p></div><div class="step"><h3>Validate</h3><p>Check the detected columns, row count, and matched image count before training.</p></div><div class="step"><h3>Run locally</h3><p>Use the full Streamlit/TensorFlow workbench for training and real inference.</p></div><div class="code">python -m streamlit run streamlit_app.py</div><div class="actions"><button class="btn" data-copy="python -m streamlit run streamlit_app.py">Copy run command</button></div></article>
<article class="card"><div class="eyebrow">Model information</div><h3>EfficientNet + clinical MLP</h3><p>The image and clinical branches produce separate evidence. Dempster–Shafer fusion combines beliefs while retaining uncertainty for inspection. Patient-level splitting reduces leakage risk.</p><h3 style="margin-top:24px">Project team</h3><p class="team">Shivani Ananya KP<br>Ajisha Threase M<br>Shruthi CS</p></article></div><div class="status warn">For research and educational use only. Do not use this system to diagnose, exclude, or treat disease.</div></section>
</main></div>
<script>
const views=[...document.querySelectorAll('.view')], nav=[...document.querySelectorAll('.nav button')];
function showView(name){views.forEach(v=>v.classList.toggle('active',v.id===name));nav.forEach(b=>b.classList.toggle('active',b.dataset.view===name));history.replaceState(null,'','#'+name);window.scrollTo(0,0)}
nav.forEach(b=>b.onclick=()=>showView(b.dataset.view));document.querySelectorAll('[data-go]').forEach(x=>x.onclick=()=>showView(x.dataset.go));
const initial=location.hash.slice(1);if(views.some(v=>v.id===initial))showView(initial);else if(initial==='workflow')showView('dataset');
let meta=null,images=[];const metaInput=document.querySelector('#meta-file'),imageInput=document.querySelector('#image-files');
function safe(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]))}
function updateReady(){const ready=!!meta&&images.length>0;document.querySelector('#prepare').disabled=!ready;document.querySelector('#dataset-pill').textContent=ready?'Ready to prepare':'Setup required';document.querySelector('#readiness').innerHTML=ready?'<span class="ok">✓ Metadata and images are ready for validation.</span>':'Add a metadata CSV and at least one image to continue.'}
function setMeta(file){meta=file||null;if(!meta){document.querySelector('#meta-info').textContent='No metadata selected.';updateReady();return}const reader=new FileReader();reader.onload=()=>{const lines=String(reader.result).split(/\r?\n/).filter(Boolean),headers=(lines[0]||'').split(',').map(x=>x.trim()).filter(Boolean);document.querySelector('#meta-info').innerHTML='<div class="file"><span>'+safe(meta.name)+'</span><span class="ok">'+Math.max(0,lines.length-1)+' rows</span></div><div class="muted" style="padding:8px">Columns: '+safe(headers.join(', ')||'none detected')+'</div>';updateReady()};reader.readAsText(meta)}
function setImages(files){images=[...(files||[])];document.querySelector('#image-info').innerHTML=images.length?images.slice(0,30).map(f=>'<div class="file"><span>'+safe(f.name)+'</span><span class="ok">ready</span></div>').join(''):'No images selected.';updateReady()}
metaInput.onchange=e=>setMeta(e.target.files[0]);imageInput.onchange=e=>setImages(e.target.files);
function drops(id,cb){const el=document.querySelector(id);['dragenter','dragover'].forEach(n=>el.addEventListener(n,e=>{e.preventDefault();el.classList.add('drag')}));['dragleave','drop'].forEach(n=>el.addEventListener(n,e=>{e.preventDefault();el.classList.remove('drag')}));el.addEventListener('drop',e=>cb(e.dataTransfer.files))}
drops('#meta-drop',f=>setMeta([...f].find(x=>x.name.toLowerCase().endsWith('.csv'))));drops('#image-drop',f=>setImages([...f].filter(x=>x.type.startsWith('image/'))));
document.querySelector('#clear-files').onclick=()=>{metaInput.value='';imageInput.value='';meta=null;images=[];setMeta(null);setImages([]);document.querySelector('#training-output').innerHTML=''};
document.querySelector('#prepare').onclick=()=>{document.querySelector('#training-output').innerHTML='<div class="status"><b class="ok">Inputs prepared.</b> Browser security keeps these files on this device. To train the TensorFlow model, add the same files in the local Streamlit workbench.<div class="code">python -m streamlit run streamlit_app.py</div></div>'};
const scan=document.querySelector('#scan-file');scan.onchange=()=>{const f=scan.files[0];if(!f)return;const url=URL.createObjectURL(f);document.querySelector('#preview').innerHTML='<img alt="Selected ultrasound preview" src="'+url+'">'};
document.querySelector('#validate-case').onclick=()=>{const age=document.querySelector('#age').value,b=document.querySelector('#birads').value,size=document.querySelector('#size').value;document.querySelector('#case-status').innerHTML=age&&b&&size?'<div class="status"><span class="ok">✓ Case inputs are complete.</span> Add a trained checkpoint in the local workbench to run inference.</div>':'<div class="status warn">Complete age, BI-RADS category, and lesion size.</div>'};
document.querySelector('#reset-case').onclick=()=>{scan.value='';['age','size'].forEach(id=>document.querySelector('#'+id).value='');document.querySelector('#birads').value='';document.querySelector('#preview').innerHTML='<span class="muted">Image preview appears here</span>';document.querySelector('#case-status').innerHTML=''};
document.querySelectorAll('[data-copy]').forEach(b=>b.onclick=async()=>{await navigator.clipboard.writeText(b.dataset.copy);const old=b.textContent;b.textContent='Copied';setTimeout(()=>b.textContent=old,1200)});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home() -> str:
    return PAGE


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "BreastSight Studio", "runtime": "vercel-fastapi"}


@app.get("/api/info")
async def info() -> dict[str, object]:
    return {
        "name": "BreastSight Studio",
        "version": "0.3.0",
        "deployment": "public project interface",
        "local_workbench": "streamlit run streamlit_app.py",
        "medical_device": False,
    }
