"""Vercel-compatible FastAPI entrypoint for BreastSight Studio.

The full TensorFlow workbench remains in ``streamlit_app.py`` and is run
locally or on a container host. This lightweight application provides the
public project site without importing TensorFlow into a serverless function.
"""
import json
import os
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from supabase_client import (auth_user, bearer_token, database_probe,
                             settings as supabase_settings, storage_signed_url,
                             user_request)

app = FastAPI(
    title="BreastSight Studio",
    description="Public project interface for multimodal breast-ultrasound research.",
    version="0.3.0",
)


class CaseCreate(BaseModel):
    research_id: str = Field(min_length=1, max_length=100)
    age: float | None = Field(default=None, ge=0, le=120)
    birads: str | None = Field(default=None, max_length=20)
    lesion_size_mm: float | None = Field(default=None, ge=0)
    diagnosis: str | None = Field(default=None, pattern="^(benign|malignant)$")
    notes: str | None = Field(default=None, max_length=2000)


class ImageCreate(BaseModel):
    case_id: str
    storage_path: str = Field(min_length=3, max_length=500)
    original_name: str = Field(min_length=1, max_length=255)
    content_type: str | None = Field(default=None, max_length=100)
    size_bytes: int | None = Field(default=None, ge=0, le=50 * 1024 * 1024)


class PredictionCreate(BaseModel):
    case_id: str
    predicted_class: str = Field(pattern="^(benign|malignant)$")
    malignant_probability: float = Field(ge=0, le=1)
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    image_uncertainty: float | None = Field(default=None, ge=0, le=1)
    clinical_uncertainty: float | None = Field(default=None, ge=0, le=1)


class AnalysisRequest(BaseModel):
    case_id: str
    image_id: str
    age: float = Field(ge=0, le=120)
    birads: str = Field(min_length=1, max_length=20)
    lesion_size_mm: float = Field(ge=0)


def _pdf_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def simple_report_pdf(lines: list[str]) -> bytes:
    """Build a small dependency-free PDF suitable for the serverless runtime."""
    commands = ["BT", "/F1 18 Tf", "54 760 Td", f"({_pdf_escape(lines[0])}) Tj", "/F1 10 Tf"]
    for line in lines[1:]:
        commands.extend(["0 -24 Td", f"({_pdf_escape(line)}) Tj"])
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = BytesIO(); output.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(output.tell()); output.write(f"{index} 0 obj\n".encode()); output.write(obj); output.write(b"\nendobj\n")
    xref = output.tell(); output.write(f"xref\n0 {len(objects)+1}\n".encode()); output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]: output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return output.getvalue()


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
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}.metric{padding:15px;border:1px solid #d7e5f2;border-radius:13px;background:#f8fbfe}.metric b{display:block;font-size:24px;color:#0a5792}.metric span{font-size:12px;color:#5f7488}.result-row{padding:20px;margin-top:14px;border:1px solid #d7e5f2;border-radius:16px;background:#fff}.result-row h3{margin:0}.result-meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.tag{padding:4px 8px;border-radius:99px;background:#eaf8f1;color:#19744f;font-size:11px}@media(max-width:900px){.metric-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="shell">
<aside class="side"><div class="brand">BreastSight <span>Studio</span></div><div class="sub">Multimodal Ultrasound Research Workbench</div>
<nav class="nav"><button data-view="home" class="active"><span class="dot"></span>Home</button><button data-view="account"><span class="dot"></span>Account & Cases</button><button data-view="dataset"><span class="dot"></span>Dataset & Training</button><button data-view="analysis"><span class="dot"></span>Ultrasound Analysis</button><button data-view="results"><span class="dot"></span>Results</button><button data-view="guide"><span class="dot"></span>Guide & About</button></nav>
<div class="warning">Research prototype only.<br>Not a medical device.</div></aside><main>

<section class="view active" id="home"><div class="top"><div><h1>Breast-ultrasound research,<br>made inspectable.</h1><p>Prepare data, examine model design, and preview de-identified scans.</p></div><div class="pill">● Online workspace</div></div>
<div class="hero"><div class="eyebrow">Multimodal evidential learning</div><h2>From ultrasound data to trustworthy uncertainty</h2><p>An EfficientNet image branch and clinical MLP produce independent Dirichlet evidence. Dempster fusion combines their beliefs into a benign/malignant output with case-level and modality-level uncertainty.</p><div class="actions"><button class="btn primary" data-go="dataset">Prepare dataset</button><button class="btn" data-go="analysis">Open analysis</button></div></div>
<div class="grid"><article class="card clickable" data-go="dataset"><div class="eyebrow">01 · Prepare</div><h3>Dataset & training</h3><p>Drop metadata and ultrasound images, validate inputs, and prepare the local training command.</p></article><article class="card clickable" data-go="analysis"><div class="eyebrow">02 · Analyze</div><h3>Ultrasound workbench</h3><p>Preview a de-identified image and enter clinical features for the inference workflow.</p></article><article class="card clickable" data-go="guide"><div class="eyebrow">03 · Understand</div><h3>Model & guidance</h3><p>Review architecture, fusion, responsible use, and project information.</p></article></div></section>

<section class="view" id="account"><div class="top"><div><h1>Account & saved cases</h1><p>Sign in through Supabase to keep de-identified case records between sessions.</p></div><div class="pill" id="auth-pill">Signed out</div></div>
<div class="grid two"><article class="card"><div class="eyebrow">Secure account</div><h3 id="auth-title">Sign in</h3><div class="field"><label>Email</label><input id="auth-email" type="email" autocomplete="email" placeholder="name@example.com"></div><div class="field"><label>Password</label><input id="auth-password" type="password" autocomplete="current-password" minlength="8" placeholder="At least 8 characters"></div><div class="actions"><button class="btn primary" id="sign-in">Sign in</button><button class="btn" id="sign-up">Create account</button><button class="btn" id="sign-out" hidden>Sign out</button></div><div id="auth-status" class="muted" style="margin-top:12px">Credentials are sent directly to Supabase Auth and are never stored in this page.</div></article>
<article class="card"><div class="eyebrow">Persistent records</div><h3>Your saved cases</h3><p>De-identified clinical fields, private ultrasound scans, predictions, and reports remain available to the account that created them.</p><div class="actions"><button class="btn" id="load-cases">Refresh cases</button></div><div class="file-list" id="saved-cases">Sign in to load saved cases.</div><div id="case-detail"></div></article></div>
<div class="status warn">Do not enter names, contact details, hospital numbers, or other direct patient identifiers.</div></section>

<section class="view" id="dataset"><div class="top"><div><h1>Dataset & training</h1><p>Add files directly—no computer locations are displayed or required.</p></div><div class="pill" id="dataset-pill">Setup required</div></div>
<div class="grid two"><article class="card"><div class="eyebrow">Metadata</div><h3>Clinical dataset</h3><label class="drop" id="meta-drop">⬆<strong>Drop CSV here</strong><small>or click to select one file</small><input id="meta-file" type="file" accept=".csv,text/csv"></label><div class="file-list" id="meta-info">No metadata selected.</div></article>
<article class="card"><div class="eyebrow">Ultrasound scans</div><h3>Image files</h3><label class="drop" id="image-drop">▧<strong>Drop images here</strong><small>PNG, JPG, JPEG, WEBP — select multiple files</small><input id="image-files" type="file" accept="image/*" multiple></label><div class="file-list" id="image-info">No images selected.</div></article></div>
<article class="card" style="margin-top:17px"><div class="eyebrow">Readiness</div><h3>Input validation</h3><div id="readiness" class="muted">Add a metadata CSV and at least one image to continue.</div><div class="actions"><button class="btn primary" id="prepare" disabled>Prepare training</button><button class="btn" id="clear-files">Clear files</button></div><div id="training-output"></div></article></section>

<section class="view" id="analysis"><div class="top"><div><h1>Ultrasound analysis</h1><p>Save a private scan with its clinical context, then run the trained model.</p></div><div class="pill">Secure case workflow</div></div>
<div class="grid two"><article class="card"><div class="eyebrow">Image input</div><h3>Select one ultrasound image</h3><label class="drop">▧<strong>Drop a scan here</strong><small>PNG, JPG, JPEG, or WEBP</small><input id="scan-file" type="file" accept="image/*"></label><div class="preview" id="preview"><span class="muted">Image preview appears here</span></div></article>
<article class="card"><div class="eyebrow">Clinical context</div><h3>Patient features</h3><div class="field"><label>De-identified research ID</label><input id="research-id" maxlength="100" placeholder="Example: STUDY-CASE-001"></div><div class="field"><label>Age</label><input id="age" type="number" min="0" max="120" placeholder="Example: 45"></div><div class="field"><label>BI-RADS category</label><select id="birads"><option value="">Select category</option><option>2</option><option>3</option><option>4A</option><option>4B</option><option>4C</option><option>5</option></select></div><div class="field"><label>Lesion size (mm)</label><input id="size" type="number" min="0" step="0.1" placeholder="Example: 16.5"></div><div class="actions"><button class="btn primary" id="save-case">Save case & scan</button><button class="btn" id="analyze-case" disabled>Analyze</button><button class="btn" id="download-report" disabled>Download PDF</button><button class="btn" id="reset-case">Reset</button></div><div id="case-status"></div></article></div>
<div class="status warn"><b>Research use only:</b> predictions and uncertainty are experimental and must not be used as a diagnosis or treatment decision.</div></section>

<section class="view" id="results"><div class="top"><div><h1>Results</h1><p>Held-out evaluation for the completed source-aware experiment grid.</p></div><div class="pill" id="results-pill">Loading results…</div></div><div id="results-dashboard"><div class="hero"><div class="eyebrow">Evaluation workspace</div><h2>Loading verified metrics</h2><p>The dashboard will show only metrics produced by saved checkpoints on untouched test groups.</p></div></div></section>

<section class="view" id="guide"><div class="top"><div><h1>Guide & about</h1><p>How to use the research prototype safely and reproducibly.</p></div><div class="pill">Version 0.3</div></div><div class="grid two"><article class="card"><div class="eyebrow">Quick start</div><div class="step"><h3>Prepare inputs</h3><p>Open Dataset & Training, add a metadata CSV, and add ultrasound image files.</p></div><div class="step"><h3>Validate</h3><p>Check the detected columns, row count, and matched image count before training.</p></div><div class="step"><h3>Run locally</h3><p>Use the full Streamlit/TensorFlow workbench for training and real inference.</p></div><div class="code">python -m streamlit run streamlit_app.py</div><div class="actions"><button class="btn" data-copy="python -m streamlit run streamlit_app.py">Copy run command</button></div></article>
<article class="card"><div class="eyebrow">Model information</div><h3>EfficientNet + clinical MLP</h3><p>The image and clinical branches produce separate evidence. Dempster–Shafer fusion combines beliefs while retaining uncertainty for inspection. Patient-level splitting reduces leakage risk.</p><h3 style="margin-top:24px">Project team</h3><p class="team">Shivani Ananya KP<br>Ajisha Threase M<br>Shruthi CS</p></article></div><div class="status warn">For research and educational use only. Do not use this system to diagnose, exclude, or treat disease.</div></section>
</main></div>
<script>
const views=[...document.querySelectorAll('.view')], nav=[...document.querySelectorAll('.nav button')];
function showView(name){views.forEach(v=>v.classList.toggle('active',v.id===name));nav.forEach(b=>b.classList.toggle('active',b.dataset.view===name));history.replaceState(null,'','#'+name);window.scrollTo(0,0);if(name==='results')loadExperimentResults()}
nav.forEach(b=>b.onclick=()=>showView(b.dataset.view));document.querySelectorAll('[data-go]').forEach(x=>x.onclick=()=>showView(x.dataset.go));
const initial=location.hash.slice(1);if(views.some(v=>v.id===initial))showView(initial);else if(initial==='workflow')showView('dataset');
let meta=null,images=[];const metaInput=document.querySelector('#meta-file'),imageInput=document.querySelector('#image-files');
function safe(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]))}
let authSession=null,supabaseConfig=null;
try{authSession=JSON.parse(sessionStorage.getItem('breastsight-auth')||'null')}catch(_){authSession=null}
async function getSupabaseConfig(){if(!supabaseConfig){const response=await fetch('/api/supabase/public-config');if(!response.ok)throw new Error('Supabase configuration is unavailable.');supabaseConfig=await response.json()}return supabaseConfig}
function setAuthMessage(message,kind=''){document.querySelector('#auth-status').innerHTML='<span class="'+kind+'">'+safe(message)+'</span>'}
function updateAuthUI(){const signed=!!authSession?.access_token,email=authSession?.user?.email||'';document.querySelector('#auth-pill').textContent=signed?'Signed in':'Signed out';document.querySelector('#auth-title').textContent=signed?'Signed in as '+email:'Sign in';document.querySelector('#sign-in').hidden=signed;document.querySelector('#sign-up').hidden=signed;document.querySelector('#sign-out').hidden=!signed;document.querySelector('#auth-email').disabled=signed;document.querySelector('#auth-password').disabled=signed;if(signed)setAuthMessage('Session active. Case access is protected by Row-Level Security.','ok')}
async function authRequest(path){const email=document.querySelector('#auth-email').value.trim(),password=document.querySelector('#auth-password').value;if(!email||password.length<8){setAuthMessage('Enter a valid email and a password of at least 8 characters.','bad');return null}const cfg=await getSupabaseConfig(),response=await fetch(cfg.url+'/auth/v1/'+path,{method:'POST',headers:{'apikey':cfg.anon_key,'Content-Type':'application/json'},body:JSON.stringify({email,password})}),data=await response.json();if(!response.ok){setAuthMessage(data.msg||data.error_description||data.message||'Authentication failed.','bad');return null}return data}
document.querySelector('#sign-in').onclick=async()=>{try{const data=await authRequest('token?grant_type=password');if(!data)return;authSession=data;sessionStorage.setItem('breastsight-auth',JSON.stringify(data));updateAuthUI();await loadCases()}catch(error){setAuthMessage(error.message,'bad')}};
document.querySelector('#sign-up').onclick=async()=>{try{const data=await authRequest('signup');if(!data)return;if(data.access_token){authSession=data;sessionStorage.setItem('breastsight-auth',JSON.stringify(data));updateAuthUI();await loadCases()}else setAuthMessage('Account created. Check your email to confirm the address before signing in.','ok')}catch(error){setAuthMessage(error.message,'bad')}};
document.querySelector('#sign-out').onclick=()=>{authSession=null;sessionStorage.removeItem('breastsight-auth');document.querySelector('#saved-cases').textContent='Sign in to load saved cases.';updateAuthUI()};
async function loadCases(){const list=document.querySelector('#saved-cases');if(!authSession?.access_token){list.textContent='Sign in to load saved cases.';return}list.textContent='Loading…';const response=await fetch('/api/cases',{headers:{'Authorization':'Bearer '+authSession.access_token}}),data=await response.json();if(!response.ok){list.innerHTML='<span class="bad">'+safe(data.detail||'Could not load cases.')+'</span>';return}list.innerHTML=data.length?data.map(item=>'<button class="file" style="width:100%;background:none;border:0;color:inherit;text-align:left;cursor:pointer" data-case="'+safe(item.id)+'"><span><b>'+safe(item.research_id)+'</b><br><span class="muted">Age '+safe(item.age??'—')+' · BI-RADS '+safe(item.birads??'—')+' · '+safe(item.lesion_size_mm??'—')+' mm</span></span><span class="ok">View</span></button>').join(''):'No cases saved yet.';list.querySelectorAll('[data-case]').forEach(button=>button.onclick=()=>showCase(button.dataset.case))}
async function showCase(id){const box=document.querySelector('#case-detail');box.innerHTML='<div class="status">Loading case…</div>';const response=await apiFetch('/api/cases/'+id),item=await response.json();if(!response.ok){box.innerHTML='<div class="status warn">'+safe(item.detail||'Could not open case.')+'</div>';return}const prediction=item.predictions[0],image=item.images[0];box.innerHTML='<div class="status"><b>'+safe(item.research_id)+'</b><br>'+safe(item.images.length)+' private scan(s) · '+safe(item.predictions.length)+' analysis result(s)'+(prediction?'<br><b>'+safe(prediction.predicted_class)+'</b> · malignant probability '+(prediction.malignant_probability*100).toFixed(1)+'% · uncertainty '+(prediction.uncertainty*100).toFixed(1)+'%':'<br>No analysis result yet.')+'<div class="actions">'+(image?'<button class="btn" id="view-scan">View scan</button>':'')+'<button class="btn" id="history-report">Download PDF</button></div><div id="history-preview"></div></div>';if(image)document.querySelector('#view-scan').onclick=async()=>{const r=await apiFetch('/api/images/'+image.id+'/url'),d=await r.json();document.querySelector('#history-preview').innerHTML=r.ok?'<div class="preview"><img alt="Private ultrasound" src="'+safe(d.url)+'"></div>':'<span class="bad">Could not load scan.</span>'};document.querySelector('#history-report').onclick=()=>downloadReport(id,item.research_id)}
function apiFetch(url,options={}){options.headers={...(options.headers||{}),'Authorization':'Bearer '+(authSession?.access_token||'')};return fetch(url,options)}
async function downloadReport(id,name){const response=await apiFetch('/api/cases/'+id+'/report.pdf');if(!response.ok){const data=await response.json();throw new Error(data.detail||'Could not create report.')}const blob=await response.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='BreastSight-'+name+'.pdf';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
document.querySelector('#load-cases').onclick=()=>loadCases();updateAuthUI();
let resultsLoaded=false;
function percent(value){return value==null?'—':(Number(value)*100).toFixed(1)+'%'}
async function loadExperimentResults(){if(resultsLoaded)return;const dashboard=document.querySelector('#results-dashboard'),pill=document.querySelector('#results-pill');try{const response=await fetch('/api/experiment-results'),data=await response.json();if(!response.ok||!data.experiments?.length)throw new Error(data.detail||'Results are not available yet.');resultsLoaded=true;pill.textContent=data.status||'Completed';dashboard.innerHTML='<div class="status warn"><b>Exploratory research results.</b> '+safe(data.disclaimer)+'</div>'+data.experiments.map(item=>'<article class="result-row"><div class="top"><div><div class="eyebrow">'+safe(item.kind)+'</div><h3>'+safe(item.name)+'</h3></div><span class="tag">'+safe(item.test_source)+'</span></div><div class="metric-grid"><div class="metric"><b>'+percent(item.accuracy)+'</b><span>Accuracy</span></div><div class="metric"><b>'+percent(item.sensitivity)+'</b><span>Sensitivity</span></div><div class="metric"><b>'+percent(item.specificity)+'</b><span>Specificity</span></div><div class="metric"><b>'+percent(item.roc_auc)+'</b><span>ROC-AUC</span></div><div class="metric"><b>'+percent(item.f1)+'</b><span>F1 score</span></div><div class="metric"><b>'+percent(item.mean_uncertainty)+'</b><span>Mean uncertainty</span></div><div class="metric"><b>'+safe(item.test_n)+'</b><span>Test cases</span></div><div class="metric"><b>'+safe(item.epochs)+'</b><span>Training epochs</span></div></div><div class="result-meta">'+(item.sources||[]).map(source=>'<span class="tag">'+safe(source.name)+': n='+safe(source.n)+', AUC '+percent(source.roc_auc)+'</span>').join('')+'</div></article>').join('')+'<div class="status"><b>Interpretation</b><br>'+safe(data.interpretation)+'</div>'}catch(error){pill.textContent='Results pending';dashboard.innerHTML='<div class="hero"><div class="eyebrow">Evaluation workspace</div><h2>Experiment evaluation in progress</h2><p>'+safe(error.message)+'</p></div>'}}
function updateReady(){const ready=!!meta&&images.length>0;document.querySelector('#prepare').disabled=!ready;document.querySelector('#dataset-pill').textContent=ready?'Ready to prepare':'Setup required';document.querySelector('#readiness').innerHTML=ready?'<span class="ok">✓ Metadata and images are ready for validation.</span>':'Add a metadata CSV and at least one image to continue.'}
function setMeta(file){meta=file||null;if(!meta){document.querySelector('#meta-info').textContent='No metadata selected.';updateReady();return}const reader=new FileReader();reader.onload=()=>{const lines=String(reader.result).split(/\r?\n/).filter(Boolean),headers=(lines[0]||'').split(',').map(x=>x.trim()).filter(Boolean);document.querySelector('#meta-info').innerHTML='<div class="file"><span>'+safe(meta.name)+'</span><span class="ok">'+Math.max(0,lines.length-1)+' rows</span></div><div class="muted" style="padding:8px">Columns: '+safe(headers.join(', ')||'none detected')+'</div>';updateReady()};reader.readAsText(meta)}
function setImages(files){images=[...(files||[])];document.querySelector('#image-info').innerHTML=images.length?images.slice(0,30).map(f=>'<div class="file"><span>'+safe(f.name)+'</span><span class="ok">ready</span></div>').join(''):'No images selected.';updateReady()}
metaInput.onchange=e=>setMeta(e.target.files[0]);imageInput.onchange=e=>setImages(e.target.files);
function drops(id,cb){const el=document.querySelector(id);['dragenter','dragover'].forEach(n=>el.addEventListener(n,e=>{e.preventDefault();el.classList.add('drag')}));['dragleave','drop'].forEach(n=>el.addEventListener(n,e=>{e.preventDefault();el.classList.remove('drag')}));el.addEventListener('drop',e=>cb(e.dataTransfer.files))}
drops('#meta-drop',f=>setMeta([...f].find(x=>x.name.toLowerCase().endsWith('.csv'))));drops('#image-drop',f=>setImages([...f].filter(x=>x.type.startsWith('image/'))));
document.querySelector('#clear-files').onclick=()=>{metaInput.value='';imageInput.value='';meta=null;images=[];setMeta(null);setImages([]);document.querySelector('#training-output').innerHTML=''};
document.querySelector('#prepare').onclick=()=>{document.querySelector('#training-output').innerHTML='<div class="status"><b class="ok">Inputs prepared.</b> Browser security keeps these files on this device. To train the TensorFlow model, add the same files in the local Streamlit workbench.<div class="code">python -m streamlit run streamlit_app.py</div></div>'};
let activeCase=null,activeImage=null;const scan=document.querySelector('#scan-file');scan.onchange=()=>{const f=scan.files[0];if(!f)return;const url=URL.createObjectURL(f);document.querySelector('#preview').innerHTML='<img alt="Selected ultrasound preview" src="'+url+'">'};
document.querySelector('#save-case').onclick=async()=>{const researchId=document.querySelector('#research-id').value.trim(),age=document.querySelector('#age').value,birads=document.querySelector('#birads').value,size=document.querySelector('#size').value,file=scan.files[0],status=document.querySelector('#case-status');if(!authSession?.access_token){status.innerHTML='<div class="status warn">Sign in under Account & Cases before saving.</div>';return}if(!researchId||!age||!birads||!size||!file){status.innerHTML='<div class="status warn">Add one ultrasound scan and complete all clinical fields.</div>';return}if(file.size>50*1024*1024||!['image/png','image/jpeg','image/webp'].includes(file.type)){status.innerHTML='<div class="status warn">Use a PNG, JPEG, or WEBP image smaller than 50 MB.</div>';return}status.innerHTML='<div class="status">Saving case and private scan…</div>';const response=await apiFetch('/api/cases',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({research_id:researchId,age:Number(age),birads,lesion_size_mm:Number(size)})}),data=await response.json();if(!response.ok){status.innerHTML='<div class="status warn">'+safe(data.detail||'Could not save this case.')+'</div>';return}activeCase=data[0];const cfg=await getSupabaseConfig(),clean=file.name.replace(/[^a-zA-Z0-9._-]/g,'_'),path=authSession.user.id+'/'+activeCase.id+'/'+crypto.randomUUID()+'-'+clean,upload=await fetch(cfg.url+'/storage/v1/object/breastsight-private/'+path,{method:'POST',headers:{'apikey':cfg.anon_key,'Authorization':'Bearer '+authSession.access_token,'Content-Type':file.type,'x-upsert':'false'},body:file});if(!upload.ok){status.innerHTML='<div class="status warn">Case saved, but the scan upload failed.</div>';return}const registered=await apiFetch('/api/images',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case_id:activeCase.id,storage_path:path,original_name:file.name,content_type:file.type,size_bytes:file.size})}),imageData=await registered.json();if(!registered.ok){status.innerHTML='<div class="status warn">Scan uploaded, but it could not be linked to the case.</div>';return}activeImage=imageData[0];status.innerHTML='<div class="status"><span class="ok">✓ Case and private scan retained in Supabase.</span> You may now run analysis or download a case report.</div>';document.querySelector('#analyze-case').disabled=false;document.querySelector('#download-report').disabled=false;loadCases()};
document.querySelector('#analyze-case').onclick=async()=>{if(!activeCase||!activeImage)return;const status=document.querySelector('#case-status');status.innerHTML='<div class="status">Running trained model…</div>';const response=await apiFetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case_id:activeCase.id,image_id:activeImage.id,age:Number(document.querySelector('#age').value),birads:document.querySelector('#birads').value,lesion_size_mm:Number(document.querySelector('#size').value)})}),data=await response.json();status.innerHTML=response.ok?'<div class="status"><b>'+safe(data.predicted_class)+'</b><br>Malignant probability: '+(data.malignant_probability*100).toFixed(1)+'%<br>Uncertainty: '+(data.uncertainty*100).toFixed(1)+'%<br><span class="muted">Research result only—not a diagnosis.</span></div>':'<div class="status warn">'+safe(data.detail||'Analysis failed.')+'</div>';if(response.ok)loadCases()};
document.querySelector('#download-report').onclick=()=>activeCase&&downloadReport(activeCase.id,activeCase.research_id).catch(error=>document.querySelector('#case-status').innerHTML='<div class="status warn">'+safe(error.message)+'</div>');
document.querySelector('#reset-case').onclick=()=>{activeCase=activeImage=null;scan.value='';['research-id','age','size'].forEach(id=>document.querySelector('#'+id).value='');document.querySelector('#birads').value='';document.querySelector('#preview').innerHTML='<span class="muted">Image preview appears here</span>';document.querySelector('#case-status').innerHTML='';document.querySelector('#analyze-case').disabled=true;document.querySelector('#download-report').disabled=true};
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


@app.get("/api/experiment-results")
async def experiment_results() -> object:
    path = Path(__file__).with_name("experiment_results.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="The experiment grid is still being evaluated.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/supabase/status")
def supabase_status() -> dict[str, object]:
    cfg = supabase_settings()
    return {
        "configured": cfg.configured,
        "connected": database_probe(),
        "database": "supabase-postgresql",
        "storage_bucket": "breastsight-private",
        "authentication_required": True,
        "service_role_configured": bool(cfg.service_role_key),
    }


@app.get("/api/supabase/public-config")
def supabase_public_config() -> dict[str, str]:
    cfg = supabase_settings()
    if not cfg.configured:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    # The publishable/anon key is intentionally public. Authorization remains
    # enforced by Supabase Auth and PostgreSQL Row-Level Security.
    return {"url": cfg.url, "anon_key": cfg.anon_key}


@app.get("/api/cases")
def list_cases(limit: int = Query(default=50, ge=1, le=200),
               authorization: str | None = Header(default=None)) -> object:
    token = bearer_token(authorization)
    return user_request("GET", "cases", token, query={
        "select": "id,research_id,age,birads,lesion_size_mm,diagnosis,created_at,updated_at",
        "order": "created_at.desc",
        "limit": str(limit),
    })


@app.post("/api/cases", status_code=201)
def create_case(case: CaseCreate,
                authorization: str | None = Header(default=None)) -> object:
    token = bearer_token(authorization)
    return user_request("POST", "cases", token, payload=case.model_dump(exclude_none=True))


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str, authorization: str | None = Header(default=None)) -> object:
    token = bearer_token(authorization)
    cases = user_request("GET", "cases", token, query={
        "id": f"eq.{case_id}",
        "select": "id,research_id,age,birads,lesion_size_mm,diagnosis,created_at,updated_at",
        "limit": "1",
    })
    if not cases:
        raise HTTPException(status_code=404, detail="Case not found.")
    images = user_request("GET", "images", token, query={
        "case_id": f"eq.{case_id}",
        "select": "id,case_id,storage_path,original_name,content_type,size_bytes,created_at",
        "order": "created_at.desc",
    })
    predictions = user_request("GET", "predictions", token, query={
        "case_id": f"eq.{case_id}",
        "select": "id,case_id,predicted_class,malignant_probability,uncertainty,image_uncertainty,clinical_uncertainty,created_at",
        "order": "created_at.desc",
    })
    return {**cases[0], "images": images, "predictions": predictions}


@app.post("/api/images", status_code=201)
def register_image(image: ImageCreate,
                   authorization: str | None = Header(default=None)) -> object:
    token = bearer_token(authorization)
    user = auth_user(token)
    if not image.storage_path.startswith(f"{user['id']}/"):
        raise HTTPException(status_code=403, detail="The image path must belong to the signed-in user.")
    cases = user_request("GET", "cases", token, query={"id": f"eq.{image.case_id}", "select": "id", "limit": "1"})
    if not cases:
        raise HTTPException(status_code=404, detail="Case not found.")
    return user_request("POST", "images", token, payload=image.model_dump(exclude_none=True))


@app.get("/api/images/{image_id}/url")
def image_url(image_id: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    token = bearer_token(authorization)
    rows = user_request("GET", "images", token, query={
        "id": f"eq.{image_id}", "select": "storage_path", "limit": "1"
    })
    if not rows:
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"url": storage_signed_url(rows[0]["storage_path"], token)}


@app.post("/api/analyze")
def analyze_case(request_data: AnalysisRequest,
                 authorization: str | None = Header(default=None)) -> object:
    token = bearer_token(authorization)
    service_url = os.getenv("MODEL_API_URL", "").rstrip("/")
    service_key = os.getenv("MODEL_API_KEY", "")
    if not service_url:
        raise HTTPException(status_code=503, detail="The trained-model service is not deployed yet.")
    cases = user_request("GET", "cases", token, query={"id": f"eq.{request_data.case_id}", "select": "id", "limit": "1"})
    images = user_request("GET", "images", token, query={"id": f"eq.{request_data.image_id}", "case_id": f"eq.{request_data.case_id}", "select": "id,storage_path", "limit": "1"})
    if not cases or not images:
        raise HTTPException(status_code=404, detail="Case or ultrasound image not found.")
    payload = request_data.model_dump(exclude={"case_id", "image_id"})
    payload["image_url"] = storage_signed_url(images[0]["storage_path"], token, expires_in=300)
    api_request = Request(f"{service_url}/predict", data=json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {service_key}"
    })
    try:
        with urlopen(api_request, timeout=55) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        raise HTTPException(status_code=502, detail="The model service rejected this analysis.") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="The model service is temporarily unavailable.") from exc
    prediction = PredictionCreate(case_id=request_data.case_id, **result)
    saved = user_request("POST", "predictions", token, payload=prediction.model_dump(exclude_none=True))
    return saved[0] if isinstance(saved, list) else saved


@app.get("/api/cases/{case_id}/report.pdf")
def case_report(case_id: str, authorization: str | None = Header(default=None)) -> Response:
    detail = case_detail(case_id, authorization)
    latest = detail["predictions"][0] if detail["predictions"] else None
    lines = [
        "BreastSight Studio - Research Report",
        f"Research case: {detail['research_id']}",
        f"Created: {detail['created_at']}",
        f"Age: {detail.get('age') or 'Not provided'}",
        f"BI-RADS: {detail.get('birads') or 'Not provided'}",
        f"Lesion size: {detail.get('lesion_size_mm') or 'Not provided'} mm",
        f"Ultrasound files: {len(detail['images'])}",
        f"Predicted class: {latest['predicted_class'] if latest else 'Not analyzed'}",
        f"Malignant probability: {latest['malignant_probability']:.1%}" if latest else "Malignant probability: Not available",
        f"Uncertainty: {latest['uncertainty']:.1%}" if latest and latest.get('uncertainty') is not None else "Uncertainty: Not available",
        "Model: BreastSight fusion evidential research prototype",
        "Research use only. This report is not a medical diagnosis.",
    ]
    filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in detail["research_id"])
    return Response(simple_report_pdf(lines), media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="BreastSight-{filename}.pdf"'
    })
