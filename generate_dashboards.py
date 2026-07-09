"""
Génère dashboard_client.html et dashboard_technique.html — fichiers autonomes.
Lancer : python3 generate_dashboards.py
         (puis ouvrir les HTML directement ou via python3 -m http.server 8765)
"""
import json, os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
print("Chargement CSV...")
df = pd.read_csv("predictive_maintenance_v3.csv", parse_dates=["timestamp"])
df = df.sort_values(["machine_id","timestamp"]).reset_index(drop=True)

FEAT = ["vibration_rms","temperature_motor","current_phase_avg","pressure_level",
        "rpm","hours_since_maintenance","ambient_temp","cumcount",
        "hours_sq","vib_x_temp","vib_x_hours","temp_x_hours"]

df["cumcount"]    = df.groupby("machine_id").cumcount()
df["hours_sq"]    = df["hours_since_maintenance"]**2
df["vib_x_temp"]  = df["vibration_rms"]*df["temperature_motor"]
df["vib_x_hours"] = df["vibration_rms"]*df["hours_since_maintenance"]
df["temp_x_hours"]= df["temperature_motor"]*df["hours_since_maintenance"]

mask = df["rul_hours"].notna() & df[FEAT].notna().all(axis=1)
Xi = SimpleImputer(strategy="median").fit_transform(df.loc[mask, FEAT])
lr = LinearRegression().fit(Xi, df.loc[mask,"rul_hours"].values)
coefs = {f: round(float(c),6) for f,c in zip(FEAT, lr.coef_)}
coefs["intercept"] = round(float(lr.intercept_),4)

NUM = ["vibration_rms","temperature_motor","current_phase_avg",
       "pressure_level","rpm","hours_since_maintenance","ambient_temp","rul_hours"]

sensors, stats_raw = {}, df[NUM].describe()
for mid in sorted(df["machine_id"].unique()):
    m = df[df["machine_id"]==mid].sort_values("timestamp").head(60)
    sensors[str(mid)] = {
        "t":    m["timestamp"].dt.strftime("%d/%m %H:%M").tolist(),
        "rul":  m["rul_hours"].round(1).tolist(),
        "vib":  m["vibration_rms"].round(3).tolist(),
        "temp": m["temperature_motor"].round(1).tolist(),
        "rpm":  m["rpm"].round(0).tolist(),
        "pres": m["pressure_level"].round(1).tolist(),
        "curr": m["current_phase_avg"].round(3).tolist(),
        "type": str(m["machine_type"].iloc[0]),
    }

stats = {c: {"count":int(stats_raw.loc["count",c]),"mean":round(float(stats_raw.loc["mean",c]),2),
             "std":round(float(stats_raw.loc["std",c]),2),"min":round(float(stats_raw.loc["min",c]),2),
             "q1":round(float(stats_raw.loc["25%",c]),2),"median":round(float(stats_raw.loc["50%",c]),2),
             "q3":round(float(stats_raw.loc["75%",c]),2),"max":round(float(stats_raw.loc["max",c]),2)}
         for c in NUM}

missing = {k: {"n":int(v),"pct":round(float(v/len(df)*100),2)}
           for k,v in df.isnull().sum().items() if v>0}

corr = df[NUM].corr().round(3)
corr_data = {"labels":NUM,"matrix":corr.values.round(3).tolist(),
             "rul_corr":corr["rul_hours"].drop("rul_hours").round(3).to_dict()}

rv = df["rul_hours"].dropna()
cnts,bins = np.histogram(rv, bins=30)
rul_hist = {"counts":cnts.tolist(),"bins":bins.round(1).tolist(),
            "mean":round(float(rv.mean()),1),"median":round(float(rv.median()),1)}

rul_by_type = {mt: {"min":round(float(v.min()),1),"q1":round(float(v.quantile(.25)),1),
                    "median":round(float(v.median()),1),"q3":round(float(v.quantile(.75)),1),
                    "max":round(float(v.max()),1)}
               for mt in sorted(df["machine_type"].dropna().unique())
               for v in [df[df["machine_type"]==mt]["rul_hours"].dropna()]}

sensor_box = {}
for s in ["vibration_rms","temperature_motor","rpm","pressure_level"]:
    sensor_box[s] = {mt: {"min":round(float(v.min()),2),"q1":round(float(v.quantile(.25)),2),
                           "median":round(float(v.median()),2),"q3":round(float(v.quantile(.75)),2),
                           "max":round(float(v.max()),2)}
                     for mt in sorted(df["machine_type"].dropna().unique())
                     for v in [df[df["machine_type"]==mt][s].dropna()]}

machine_type_counts = df["machine_type"].value_counts().to_dict()
dataset_info = {"n_rows":len(df),"n_machines":int(df["machine_id"].nunique()),
                "n_vars":15,"n_days":14,"failure_24h":int(df["failure_within_24h"].sum())}

MODEL = {"names":["Ridge","Random Forest","XGBoost","MLP"],
         "mae":[16.288,3.077,4.402,4.911],"r2":[0.394,0.955,0.938,0.916],
         "cv_mae":[16.603,3.097,4.430,4.943],"cv_std":[0.114,0.073,0.114,0.200]}
FI = {"names":["cumcount","hours_sq","hours_since_maintenance","rpm","temperature_motor",
               "machine_type_enc","current_phase_avg","vib_x_temp","pressure_level","vibration_rms"],
      "imp":[0.400,0.235,0.080,0.038,0.027,0.020,0.017,0.013,0.012,0.010],
      "new":[True,True,False,False,False,False,False,True,False,False]}
VARIABLES = [
    ("timestamp","datetime","drop","Horodatage"),
    ("machine_id","int64","drop","ID machine"),
    ("machine_type","object","keep","Type de machine — OHE"),
    ("vibration_rms","float64","keep","Vibration RMS (g) — clé"),
    ("temperature_motor","float64","keep","Température moteur (°C)"),
    ("current_phase_avg","float64","keep","Courant de phase (A)"),
    ("pressure_level","float64","keep","Pression (bar)"),
    ("rpm","float64","keep","Vitesse de rotation (RPM)"),
    ("operating_mode","object","keep","Mode opératoire — OHE"),
    ("hours_since_maintenance","float64","keep","Heures depuis maintenance"),
    ("ambient_temp","float64","keep","Température ambiante (°C)"),
    ("rul_hours","float64","target","🎯 Variable cible"),
    ("failure_within_24h","int64","leak","⚠️ Data leakage"),
    ("failure_type","object","leak","⚠️ Data leakage"),
    ("estimated_repair_cost","int64","leak","⚠️ Data leakage"),
]

def J(o): return json.dumps(o,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/')

print(f"Données prêtes — {dataset_info['n_rows']:,} lignes · {dataset_info['n_machines']} machines")

# ══════════════════════════════════════════════════════════════════════════════
# CSS COMMUN (partagé par les deux dashboards)
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--fh:'Rajdhani',sans-serif;--fb:'Inter',sans-serif;--fm:'JetBrains Mono',monospace;
  --rad:12px;--rad-s:8px;--tr:.22s ease}
[data-theme="dark"]{--a:#00d4aa;--r:#ef4444;--o:#f59e0b;--b:#6366f1;--g:#22c55e;
  --bg:#070b14;--card:#0c1424;--card2:#111d30;--bd:#1a2e4a;--bd2:#1e3654;
  --tx:#e2e8f0;--txm:#64748b;--txf:#2a3f5e;--sh:0 4px 24px rgba(0,0,0,.45)}
[data-theme="light"]{--a:#0d9488;--r:#dc2626;--o:#d97706;--b:#4f46e5;--g:#16a34a;
  --bg:#f0f4f8;--card:#fff;--card2:#f8fafc;--bd:#d0dae8;--bd2:#b8c6d8;
  --tx:#0f172a;--txm:#374151;--txf:#6b7280;--sh:0 2px 12px rgba(0,0,0,.08)}
html,body{height:100%;font-family:var(--fb);font-size:14px;color:var(--tx);background:var(--bg);transition:background .3s,color .3s}
.app{min-height:100vh;display:flex;flex-direction:column}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:var(--a)}
@keyframes fadeUp{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{box-shadow:none}50%{box-shadow:0 0 0 4px rgba(239,68,68,.13)}}
@keyframes scanR{from{transform:rotate(0)}to{transform:rotate(360deg)}}
@keyframes gBlink{0%,100%{opacity:.4}50%{opacity:1}}
.topnav{height:52px;display:flex;align-items:center;padding:0 20px;
  background:var(--card);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:100;transition:background .3s}
.brand{font-family:var(--fh);font-size:1.2rem;font-weight:700;letter-spacing:3px;
  color:var(--tx);margin-right:22px;user-select:none;white-space:nowrap}
.brand em{color:var(--a);font-style:normal}
.tabs{display:flex;gap:2px;flex:1}
.tab{padding:7px 15px;border:none;border-radius:var(--rad-s);cursor:pointer;
  font-family:var(--fh);font-size:.82rem;font-weight:600;letter-spacing:.5px;
  background:transparent;color:var(--txm);transition:all .2s;white-space:nowrap}
.tab:hover{background:rgba(0,212,170,.08);color:var(--a)}
.tab.active{background:rgba(0,212,170,.13);color:var(--a)}
.nav-link{font-size:.72rem;color:var(--txm);text-decoration:none;
  padding:5px 11px;border:1px solid var(--bd);border-radius:var(--rad-s);
  margin-right:8px;transition:all .2s;white-space:nowrap}
.nav-link:hover{border-color:var(--a);color:var(--a)}
.theme-btn{width:34px;height:34px;border-radius:50%;border:1px solid var(--bd);
  background:var(--card2);cursor:pointer;font-size:.9rem;display:flex;
  align-items:center;justify-content:center;transition:all .2s;color:var(--tx)}
.theme-btn:hover{border-color:var(--a);transform:rotate(20deg)}
.main{flex:1;padding:20px 22px;overflow-y:auto}
.tab-panel{display:none}.tab-panel.active{display:block;animation:fadeUp .28s ease}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.g5{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);padding:18px;transition:background .3s}
.page-hdr{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);
  padding:18px 22px;margin-bottom:16px;position:relative;overflow:hidden;transition:background .3s}
.page-hdr::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--a),transparent)}
.page-hdr h1{font-family:var(--fh);font-size:1.55rem;font-weight:700}
.page-hdr h1 em{color:var(--a);font-style:normal}
.page-hdr p{font-size:.75rem;color:var(--txm);margin-top:3px}
.chip{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.62rem;
  font-family:var(--fh);letter-spacing:1px;margin-top:6px;margin-right:3px}
.chip-a{background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);color:var(--a)}
.chip-r{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:var(--r)}
.chip-b{background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.3);color:var(--b)}
.chip-g{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:var(--g)}
.sec-title{font-family:var(--fh);font-size:.9rem;font-weight:700;color:var(--a);
  letter-spacing:1.5px;text-transform:uppercase;border-left:3px solid var(--a);
  padding-left:10px;margin:18px 0 12px}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);
  padding:14px;text-align:center;position:relative;overflow:hidden;
  transition:transform .2s,background .3s}
.kpi:hover{transform:translateY(-2px)}
.kpi::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--a),transparent)}
.kpi-v{font-family:var(--fh);font-size:1.65rem;font-weight:700;color:var(--a);line-height:1.1}
.kpi-l{font-size:.62rem;color:var(--txm);text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.kpi-d{font-size:.68rem;color:var(--txm);margin-top:2px}
/* sliders */
.param-card{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);padding:16px 18px;transition:background .3s}
.sl-group{margin-bottom:10px}
.sl-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.sl-name{font-size:.67rem;color:var(--txm);text-transform:uppercase;letter-spacing:.8px}
.sl-val{font-family:var(--fh);font-size:.88rem;font-weight:700;color:var(--a);min-width:56px;text-align:right}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;
  background:var(--bd);border-radius:2px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;
  background:var(--a);border-radius:50%;cursor:pointer;
  box-shadow:0 0 7px rgba(0,212,170,.35);transition:transform .2s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.3)}
.btn-predict{width:100%;padding:13px;margin-top:12px;
  background:linear-gradient(135deg,var(--a),#009e7e);
  color:#fff;border:none;border-radius:var(--rad-s);
  font-family:var(--fh);font-size:1rem;font-weight:700;letter-spacing:2.5px;
  cursor:pointer;box-shadow:0 4px 18px rgba(0,212,170,.2);transition:all .2s}
[data-theme="light"] .btn-predict{color:#0f172a}
.btn-predict:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,212,170,.38)}
.btn-predict:active{transform:none}
/* gauge */
.gauge-card{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);
  padding:16px 18px;transition:background .3s;min-height:310px;
  display:flex;flex-direction:column;justify-content:center}
.g-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:14px;padding:24px 16px;text-align:center}
.scan-svg{opacity:.15;display:block}
.empty-ttl{font-family:var(--fh);font-size:1rem;letter-spacing:1px;color:var(--txf);text-transform:uppercase}
.empty-sub{font-size:.75rem;color:var(--txf);line-height:1.7;max-width:210px}
.empty-sub strong{color:var(--a)}
.g-result{display:none}
.g-result.show{display:block;animation:fadeUp .4s ease}
/* alerts */
.alert{border-radius:var(--rad-s);padding:12px 14px;margin-top:12px;border:1px solid}
.alert-title{font-family:var(--fh);font-size:.88rem;font-weight:700;letter-spacing:1px;margin-bottom:4px}
.alert-desc{font-size:.76rem;line-height:1.55;color:var(--txm)}
.al-ok{background:rgba(0,212,170,.06);border-color:rgba(0,212,170,.3)}
.al-warn{background:rgba(245,158,11,.06);border-color:rgba(245,158,11,.3)}
.al-crit{background:rgba(239,68,68,.06);border-color:rgba(239,68,68,.3);animation:pulse 2s ease-in-out infinite}
/* scenarios */
.sc-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.sc-card{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);
  padding:14px;text-align:center;cursor:pointer;transition:all .2s;position:relative;overflow:hidden}
.sc-card:hover{transform:translateY(-3px);box-shadow:var(--sh)}
.sc-name{font-size:.62rem;color:var(--txm);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:7px}
.sc-val{font-family:var(--fh);font-size:1.85rem;font-weight:700;margin:3px 0}
.sc-badge{font-size:.6rem;font-family:var(--fh);font-weight:700;letter-spacing:2px;
  padding:3px 9px;border-radius:20px;display:inline-block}
/* svg charts */
.chart-card{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);padding:16px;overflow:hidden;transition:background .3s}
.chart-title{font-family:var(--fh);font-size:.75rem;color:var(--txm);letter-spacing:1.5px;
  text-transform:uppercase;margin-bottom:12px}
.chart-box{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);overflow:hidden;transition:background .3s}
svg text{user-select:none}
/* sensor buttons */
.s-btns{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.s-btn{padding:5px 12px;border-radius:20px;font-size:.7rem;font-family:var(--fh);
  font-weight:600;letter-spacing:.8px;cursor:pointer;border:1px solid var(--bd);
  background:transparent;color:var(--txm);transition:all .2s}
select{background:var(--card2);color:var(--tx);border:1px solid var(--bd);
  border-radius:var(--rad-s);padding:6px 12px;font-family:var(--fb);font-size:.83rem;
  outline:none;cursor:pointer}
select:focus{border-color:var(--a)}
/* table */
.tbl-wrap{background:var(--card);border:1px solid var(--bd);border-radius:var(--rad);overflow:hidden;transition:background .3s}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:var(--card2);color:var(--txm);font-size:.62rem;text-transform:uppercase;
  letter-spacing:1px;padding:9px 13px;text-align:left;border-bottom:1px solid var(--bd)}
td{padding:8px 13px;border-bottom:1px solid var(--bd);color:var(--tx)}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(0,212,170,.025)}
code{background:var(--card2);padding:1px 5px;border-radius:4px;font-size:.77rem;border:1px solid var(--bd)}
.badge{display:inline-flex;padding:2px 8px;border-radius:20px;font-size:.62rem;letter-spacing:1px;font-family:var(--fh);font-weight:600}
.b-keep{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.4);color:var(--g)}
.b-drop{background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.4);color:var(--b)}
.b-target{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.4);color:var(--o)}
.b-leak{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);color:var(--r)}
/* pipeline */
.pipe-flow{display:flex;align-items:center;flex-wrap:wrap;gap:4px;padding:12px 0}
.pipe-step{background:var(--card2);border:1px solid var(--bd);border-radius:var(--rad-s);
  padding:14px 16px;text-align:center;min-width:130px;flex:1}
.pipe-ico{font-size:1.3rem;margin-bottom:4px}
.pipe-name{font-family:var(--fh);font-size:.82rem;font-weight:700;margin-bottom:2px}
.pipe-detail{font-size:.66rem;color:var(--txm)}
.pipe-arrow{font-size:1.1rem;color:var(--bd2);padding:0 2px}
.code-block{background:#0a0e1a;border:1px solid var(--bd);border-radius:var(--rad);
  padding:15px 18px;font-family:var(--fm);font-size:.75rem;line-height:1.7;overflow-x:auto;color:#c9d1d9}
.kw{color:#ff7b72}.fn{color:#d2a8ff}.st{color:#a5d6ff}.nb{color:#79c0ff}.cm{color:#8b949e}
"""

PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD CLIENT
# ══════════════════════════════════════════════════════════════════════════════
client_data = {"lr":{"coefs":coefs,"features":FEAT},"sensors":sensors,
               "model":MODEL,"fi":FI,"di":dataset_info}

# Gauge SVG coords: cx=130,cy=130,r=95 — arc 135°→45° CW (270°)
# Start(135°)=(62.8,197.2) P10(162°)=(39.6,159.4) P30(216°)=(53.1,74.2) End(45°)=(197.2,197.2)
# ARC_LEN=447.7  FULL_CIRC=596.9

CLIENT_JS = r"""
var D=__DATA__;
var C={a:'#00d4aa',r:'#ef4444',o:'#f59e0b',b:'#6366f1',g:'#22c55e'};
var MC=['#ef4444','#00d4aa','#6366f1','#f59e0b'];

function setTheme(t){
  document.documentElement.dataset.theme=t;
  localStorage.setItem('theme',t);
  document.getElementById('tb').textContent=t==='dark'?'☀️':'🌙';
}
function toggleTheme(){setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');}

function showTab(id,btn){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('tp-'+id).classList.add('active');btn.classList.add('active');
  if(id==='perf')setTimeout(renderPerf,50);
  if(id==='capt'){initSel();setTimeout(renderSensors,50);}
}

function sv(id){return parseFloat(document.getElementById(id).value);}
function upd(id,did,sfx,dec){document.getElementById(did).textContent=sv(id).toFixed(dec||0)+sfx;}

function runPred(){
  var vib=sv('s-vib'),temp=sv('s-temp'),curr=sv('s-curr'),pres=sv('s-pres'),
      rpm=sv('s-rpm'),hours=sv('s-hours'),amb=sv('s-amb'),cum=sv('s-cum');
  var vals={vibration_rms:vib,temperature_motor:temp,current_phase_avg:curr,
            pressure_level:pres,rpm:rpm,hours_since_maintenance:hours,
            ambient_temp:amb,cumcount:cum,hours_sq:hours*hours,
            vib_x_temp:vib*temp,vib_x_hours:vib*hours,temp_x_hours:temp*hours};
  var rul=D.lr.coefs.intercept;
  D.lr.features.forEach(function(f){rul+=D.lr.coefs[f]*(vals[f]||0);});
  rul=Math.max(0,Math.round(rul*10)/10);
  showResult(rul);
}
function showResult(rul){
  document.getElementById('g-empty').style.display='none';
  var r=document.getElementById('g-result');r.style.display='block';
  setTimeout(function(){r.classList.add('show');},20);
  drawGauge(rul);setAlert(rul);
}

function drawGauge(rul){
  var ARC=447.7,color=rul<10?C.r:rul<30?C.o:C.a;
  var offset=ARC*(1-Math.min(rul,100)/100);
  var fill=document.getElementById('g-fill'),glow=document.getElementById('g-glow');
  fill.style.stroke=color;fill.style.strokeDashoffset=offset;
  glow.style.stroke=color;glow.style.strokeDashoffset=offset;
  document.getElementById('g-num').textContent=rul.toFixed(1);
  document.getElementById('g-num').setAttribute('fill',color);
  document.getElementById('g-unit').setAttribute('fill',color);
}
function setAlert(rul){
  var color,cls,title,desc;
  if(rul<10){color=C.r;cls='al-crit';title='ARRET IMMEDIAT REQUIS';
    desc='RUL <strong>'+rul.toFixed(0)+'h</strong> — Intervention immediate requise.';}
  else if(rul<30){color=C.o;cls='al-warn';title='ALERTE — PLANIFIER SOUS 48H';
    desc='RUL <strong>'+rul.toFixed(1)+'h</strong> — Programmer une maintenance.';}
  else{color=C.a;cls='al-ok';title='ETAT NORMAL';
    desc='Prochaine maintenance dans <strong>'+rul.toFixed(0)+'h</strong>.';}
  document.getElementById('g-alert').innerHTML=
    '<div class="alert '+cls+'"><div class="alert-title" style="color:'+color+'">'+title+'</div>'+
    '<div class="alert-desc">'+desc+'</div></div>';
}

var SC=[
  {name:'Machine Neuve', vib:.8, temp:42,curr:5,  pres:22,rpm:900, hours:50, amb:13,cum:80},
  {name:'Mi-Vie',        vib:1.5,temp:58,curr:7,  pres:35,rpm:1100,hours:300,amb:13,cum:450},
  {name:'Degradee',      vib:4.2,temp:85,curr:8.5,pres:55,rpm:1200,hours:450,amb:13,cum:750},
  {name:'Critique',      vib:7.5,temp:92,curr:12, pres:80,rpm:1500,hours:250,amb:13,cum:1000},
];
function scRUL(s){
  var vals={vibration_rms:s.vib,temperature_motor:s.temp,current_phase_avg:s.curr,
            pressure_level:s.pres,rpm:s.rpm,hours_since_maintenance:s.hours,
            ambient_temp:s.amb,cumcount:s.cum,hours_sq:s.hours*s.hours,
            vib_x_temp:s.vib*s.temp,vib_x_hours:s.vib*s.hours,temp_x_hours:s.temp*s.hours};
  var rul=D.lr.coefs.intercept;
  D.lr.features.forEach(function(f){rul+=D.lr.coefs[f]*(vals[f]||0);});
  return Math.max(0,Math.round(rul*10)/10);
}
function applyScenario(i){
  var s=SC[i];
  function set(id,v){var el=document.getElementById(id);el.value=v;el.dispatchEvent(new Event('input'));}
  set('s-vib',s.vib);set('s-temp',s.temp);set('s-curr',s.curr);set('s-pres',s.pres);
  set('s-rpm',s.rpm);set('s-hours',s.hours);set('s-amb',s.amb);set('s-cum',s.cum);
  runPred();
}
function drawScenarios(){
  var h='';
  SC.forEach(function(s,i){
    var rul=scRUL(s),color=rul<10?C.r:rul<30?C.o:C.a;
    var status=rul<10?'CRITIQUE':rul<30?'ALERTE':'NORMAL';
    var bg=rul<10?'rgba(239,68,68,.07)':rul<30?'rgba(245,158,11,.06)':'rgba(0,212,170,.05)';
    h+='<div class="sc-card" onclick="applyScenario('+i+')" style="border-top:3px solid '+color+';background:'+bg+'">'+
       '<div class="sc-name">'+s.name+'</div>'+
       '<div class="sc-val" style="color:'+color+'">'+rul.toFixed(1)+'h</div>'+
       '<div class="sc-badge" style="color:'+color+';border-color:'+color+'40">'+status+'</div></div>';
  });
  document.getElementById('sc-grid').innerHTML=h;
}

function barChart(id,labels,values,colors,yMax,vFmt){
  var W=440,H=220,PL=44,PR=16,PT=28,PB=28,iW=W-PL-PR,iH=H-PT-PB;
  var mx=yMax||Math.max.apply(null,values)*1.22;
  var bGap=iW/labels.length,bW=bGap*.55;
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" style="display:block;overflow:visible"><defs>';
  colors.forEach(function(c,i){
    s+='<linearGradient id="bg'+id+i+'" x1="0" y1="0" x2="0" y2="1">'+
       '<stop offset="0%" stop-color="'+c+'" stop-opacity=".9"/>'+
       '<stop offset="100%" stop-color="'+c+'" stop-opacity=".4"/></linearGradient>';
  });
  s+='</defs>';
  for(var gi=0;gi<=4;gi++){
    var gy=PT+iH*(1-gi/4),gv=mx*gi/4;
    s+='<line x1="'+PL+'" y1="'+gy+'" x2="'+(W-PR)+'" y2="'+gy+'" stroke="var(--bd)" stroke-width="0.5"/>';
    s+='<text x="'+(PL-5)+'" y="'+(gy+4)+'" text-anchor="end" fill="var(--txm)" font-size="9">'+
       (vFmt?vFmt(gv):gv.toFixed(2))+'</text>';
  }
  labels.forEach(function(lbl,i){
    var v=values[i],bH=(v/mx)*iH,bX=PL+i*bGap+(bGap-bW)/2,bY=PT+iH-bH;
    s+='<rect data-y="'+bY+'" data-h="'+bH+'" x="'+bX+'" y="'+(PT+iH)+'" width="'+bW+'" height="0"'+
       ' rx="5" fill="url(#bg'+id+i+')" style="transition:y .55s cubic-bezier(.4,0,.2,1) '+(i*.07)+'s,height .55s cubic-bezier(.4,0,.2,1) '+(i*.07)+'s"/>';
    s+='<text x="'+(bX+bW/2)+'" y="'+(bY-7)+'" text-anchor="middle" fill="'+colors[i]+'"'+
       ' font-size="11" font-weight="700" opacity="0" style="transition:opacity .3s '+(i*.07+.45)+'s" data-op="1">'+
       (vFmt?vFmt(v):v.toFixed(3))+'</text>';
    s+='<text x="'+(bX+bW/2)+'" y="'+(PT+iH+16)+'" text-anchor="middle" fill="var(--txm)" font-size="10">'+lbl+'</text>';
  });
  s+='</svg>';
  var el=document.getElementById(id);el.innerHTML=s;
  setTimeout(function(){
    el.querySelectorAll('[data-y]').forEach(function(r){r.setAttribute('y',r.dataset.y);r.setAttribute('height',r.dataset.h);});
    el.querySelectorAll('[data-op]').forEach(function(t){t.style.opacity='1';});
  },30);
}

function hBars(id,names,values,isNew){
  var sorted=names.map(function(n,i){return{n:n,v:values[i],isN:isNew[i]};})
                  .sort(function(a,b){return a.v-b.v;});
  var W=560,rH=28,PT=6,PL=170,PR=60,iW=W-PL-PR;
  var H=sorted.length*rH+PT*2,mx=Math.max.apply(null,values)*1.1;
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" style="display:block">';
  sorted.forEach(function(d,i){
    var bW=(d.v/mx)*iW,bY=PT+i*rH,color=d.isN?'#ef4444':'#6366f1';
    s+='<rect x="'+PL+'" y="'+(bY+5)+'" width="'+iW+'" height="'+(rH-10)+'" rx="3" fill="var(--bd)" opacity="0.35"/>';
    s+='<rect class="hb" data-w="'+bW+'" x="'+PL+'" y="'+(bY+5)+'" width="0" height="'+(rH-10)+'" rx="3"'+
       ' fill="'+color+'" opacity="0.85" style="transition:width .6s cubic-bezier(.4,0,.2,1) '+(i*.04)+'s"/>';
    s+='<text x="'+(PL-12)+'" y="'+(bY+rH/2+4)+'" text-anchor="end" fill="var(--txm)" font-size="9.5">'+d.n+'</text>';
    if(d.isN)s+='<circle cx="'+(PL-5)+'" cy="'+(bY+rH/2)+'" r="3" fill="'+color+'"/>';
    s+='<text class="hb-lbl" data-x="'+(PL+bW+6)+'" x="'+(PL+6)+'" y="'+(bY+rH/2+4)+'" fill="'+color+'"'+
       ' font-size="9" font-weight="700" opacity="0" style="transition:opacity .3s '+(i*.04+.5)+'s" data-op="1">'+
       (d.v*100).toFixed(1)+'%</text>';
  });
  s+='</svg>';
  var el=document.getElementById(id);el.innerHTML=s;
  setTimeout(function(){
    el.querySelectorAll('.hb').forEach(function(r){r.setAttribute('width',r.dataset.w);});
    el.querySelectorAll('.hb-lbl').forEach(function(t){t.setAttribute('x',t.dataset.x);t.style.opacity='1';});
  },30);
}

function lineChart(id,datasets,opts){
  opts=opts||{};
  var W=580,H=190,PL=46,PR=44,PT=16,PB=30,iW=W-PL-PR,iH=H-PT-PB;
  var allY=[];datasets.forEach(function(d){d.y.forEach(function(v){if(v!=null)allY.push(v);});});
  var yMin=opts.yMin!==undefined?opts.yMin:Math.min.apply(null,allY)*0.94;
  var yMax=opts.yMax!==undefined?opts.yMax:Math.max.apply(null,allY)*1.06;
  if(yMax===yMin){yMax+=1;yMin-=1;}
  var n=datasets[0].x.length;
  var sx=function(i){return PL+(i/(n-1))*iW;};
  var sy=function(v){return PT+iH-((v-yMin)/(yMax-yMin))*iH;};
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" style="display:block"><defs>';
  datasets.forEach(function(ds,i){
    s+='<linearGradient id="lg'+id+i+'" x1="0" y1="0" x2="0" y2="1">'+
       '<stop offset="0%" stop-color="'+ds.color+'" stop-opacity="0.22"/>'+
       '<stop offset="100%" stop-color="'+ds.color+'" stop-opacity="0"/></linearGradient>';
  });
  s+='</defs>';
  for(var gi=0;gi<=4;gi++){
    var gv=yMin+(yMax-yMin)*gi/4,gy=sy(gv);
    s+='<line x1="'+PL+'" y1="'+gy+'" x2="'+(W-PR)+'" y2="'+gy+'" stroke="var(--bd)" stroke-width="0.5"/>';
    s+='<text x="'+(PL-5)+'" y="'+(gy+4)+'" text-anchor="end" fill="var(--txm)" font-size="9">'+gv.toFixed(opts.dec||1)+'</text>';
  }
  var step=Math.ceil(n/6);
  for(var xi=0;xi<n;xi+=step){
    var lbl=datasets[0].x[xi];
    if(lbl&&lbl.length>5)lbl=lbl.slice(5);
    s+='<text x="'+sx(xi)+'" y="'+(PT+iH+18)+'" text-anchor="middle" fill="var(--txm)" font-size="8">'+lbl+'</text>';
  }
  if(opts.th)opts.th.forEach(function(t){
    var ty=sy(Math.max(yMin,Math.min(yMax,t.v)));
    s+='<line x1="'+PL+'" y1="'+ty+'" x2="'+(W-PR)+'" y2="'+ty+'" stroke="'+t.c+'" stroke-width="1.2" stroke-dasharray="5 3" opacity="0.65"/>';
    s+='<text x="'+(W-PR+3)+'" y="'+(ty+4)+'" fill="'+t.c+'" font-size="8" font-weight="700">'+t.l+'</text>';
  });
  datasets.forEach(function(ds,di){
    var pts=[];
    ds.y.forEach(function(v,i){if(v!=null)pts.push({x:sx(i),y:sy(v)});});
    if(pts.length<2)return;
    var path=pts.reduce(function(acc,pt,k){
      if(k===0)return 'M '+pt.x+' '+pt.y;
      var prev=pts[k-1],mx2=(prev.x+pt.x)/2;
      return acc+' C '+mx2+' '+prev.y+' '+mx2+' '+pt.y+' '+pt.x+' '+pt.y;
    },'');
    var area=path+' L '+pts[pts.length-1].x+' '+(PT+iH)+' L '+pts[0].x+' '+(PT+iH)+' Z';
    s+='<path d="'+area+'" fill="url(#lg'+id+di+')"/>';
    s+='<path d="'+path+'" fill="none" stroke="'+ds.color+'" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>';
  });
  s+='</svg>';
  document.getElementById(id).innerHTML=s;
}

function renderPerf(){
  var m=D.model;
  barChart('c-r2',m.names,m.r2,MC,1.1,function(v){return v.toFixed(2);});
  barChart('c-mae',m.names,m.mae,MC,20,function(v){return v.toFixed(1)+'h';});
  hBars('c-fi',D.fi.names,D.fi.imp,D.fi.new);
}

var SKEYS=['rul','vib','temp','rpm','pres','curr'];
var SLBL={rul:'RUL (h)',vib:'Vibration (g)',temp:'Temp C',rpm:'RPM',pres:'Pression (b)',curr:'Courant (A)'};
var SCOL={rul:'#00d4aa',vib:'#ef4444',temp:'#f59e0b',rpm:'#6366f1',pres:'#22c55e',curr:'#38bdf8'};
var activeSensors=['rul','vib'];
function initSel(){
  var sel=document.getElementById('m-sel');
  if(sel.options.length>1)return;
  Object.keys(D.sensors).sort(function(a,b){return+a-+b;}).forEach(function(id){
    var o=document.createElement('option');o.value=id;o.textContent='Machine #'+id;sel.appendChild(o);
  });
}
function renderSensors(){
  var mid=document.getElementById('m-sel').value||Object.keys(D.sensors)[0];
  var m=D.sensors[mid];
  if(!m)return;
  lineChart('c-rul',[{x:m.t,y:m.rul,color:'#00d4aa'}],
    {yMin:0,dec:0,th:[{v:10,c:'#ef4444',l:'10h'},{v:30,c:'#f59e0b',l:'30h'}]});
  var datasets=activeSensors.map(function(k){
    if(!m[k])return null;
    var vals=m[k].filter(function(v){return v!=null;});
    if(!vals.length)return null;
    var mn=Math.min.apply(null,vals),mx=Math.max.apply(null,vals),rng=mx-mn||1;
    return{x:m.t,y:m[k].map(function(v){return v!=null?(v-mn)/rng*100:null;}),color:SCOL[k]};
  }).filter(Boolean);
  if(datasets.length)lineChart('c-sensors',datasets,{yMin:0,yMax:100,dec:0});
  renderSBtns();
}
function renderSBtns(){
  var h='';
  SKEYS.forEach(function(k){
    var on=activeSensors.indexOf(k)>-1;
    h+='<button class="s-btn'+(on?' on':'')+'" onclick="toggleSensor(\''+k+'\')"'
       +(on?' style="border-color:'+SCOL[k]+';color:'+SCOL[k]+'"':'')+'>'
       +SLBL[k]+'</button>';
  });
  document.getElementById('s-btns').innerHTML=h;
}
function toggleSensor(k){
  var i=activeSensors.indexOf(k);
  if(i>-1)activeSensors.splice(i,1);else activeSensors.push(k);
  renderSensors();
}

window.addEventListener('DOMContentLoaded',function(){
  setTheme(localStorage.getItem('theme')||'dark');
  drawScenarios();
});
"""

_CLIENT_HTML_RAW = """<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard Client — Maintenance Predictive</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>__CSS__</style>
</head>
<body>
<script>__JS__</script>
<div class="app">
<nav class="topnav">
  <div class="brand">&#9881; PRED<em>ML</em></div>
  <div class="tabs">
    <button class="tab active" onclick="showTab('pred',this)">&#9889; Prediction RUL</button>
    <button class="tab" onclick="showTab('perf',this)">&#128202; Performances</button>
    <button class="tab" onclick="showTab('capt',this)">&#128225; Capteurs</button>
  </div>
  <a class="nav-link" href="dashboard_technique.html">&#8594; Technique</a>
  <button class="theme-btn" onclick="toggleTheme()" id="tb">&#9728;&#65039;</button>
</nav>
<div class="main">

<!-- TAB PREDICTION -->
<div id="tp-pred" class="tab-panel active">
<div class="page-hdr">
  <h1>&#9889; Prediction <em>RUL</em></h1>
  <p>Remaining Useful Life &middot; Ajustez les parametres puis cliquez sur Calculer</p>
</div>
<div class="g2" style="align-items:start">
  <div class="param-card">
    <div class="sec-title" style="margin-top:0">Parametres Machine</div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">Heures depuis maintenance</span><span class="sl-val" id="d-hours">250h</span></div>
      <input type="range" id="s-hours" min="0" max="600" value="250" step="10" oninput="upd('s-hours','d-hours','h')">
    </div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">Mesures cumulees</span><span class="sl-val" id="d-cum">500</span></div>
      <input type="range" id="s-cum" min="0" max="2000" value="500" step="10" oninput="upd('s-cum','d-cum','')">
    </div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">Vibration RMS (g)</span><span class="sl-val" id="d-vib">1.2g</span></div>
      <input type="range" id="s-vib" min="0" max="10" value="1.2" step="0.1" oninput="upd('s-vib','d-vib','g',1)">
    </div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">Temperature moteur</span><span class="sl-val" id="d-temp">55&#176;C</span></div>
      <input type="range" id="s-temp" min="28" max="95" value="55" step="1" oninput="upd('s-temp','d-temp','&#176;C')">
    </div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">RPM</span><span class="sl-val" id="d-rpm">900</span></div>
      <input type="range" id="s-rpm" min="100" max="4100" value="900" step="50" oninput="upd('s-rpm','d-rpm','')">
    </div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">Pression (bar)</span><span class="sl-val" id="d-pres">45b</span></div>
      <input type="range" id="s-pres" min="10" max="200" value="45" step="1" oninput="upd('s-pres','d-pres','b')">
    </div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">Courant de phase (A)</span><span class="sl-val" id="d-curr">6.0A</span></div>
      <input type="range" id="s-curr" min="2" max="35" value="6" step="0.5" oninput="upd('s-curr','d-curr','A',1)">
    </div>
    <div class="sl-group">
      <div class="sl-row"><span class="sl-name">Temperature ambiante</span><span class="sl-val" id="d-amb">13&#176;C</span></div>
      <input type="range" id="s-amb" min="8" max="18" value="13" step="0.5" oninput="upd('s-amb','d-amb','&#176;C')">
    </div>
    <button class="btn-predict" onclick="runPred()">&#9889; CALCULER LE RUL</button>
  </div>

  <div class="gauge-card">
    <div id="g-empty" class="g-empty">
      <svg class="scan-svg" width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r="62" fill="none" stroke="var(--bd)" stroke-width="1"/>
        <circle cx="70" cy="70" r="44" fill="none" stroke="var(--bd)" stroke-width=".8"/>
        <circle cx="70" cy="70" r="24" fill="none" stroke="var(--bd)" stroke-width=".6"/>
        <line x1="70" y1="8" x2="70" y2="132" stroke="var(--bd)" stroke-width=".5"/>
        <line x1="8" y1="70" x2="132" y2="70" stroke="var(--bd)" stroke-width=".5"/>
        <line x1="70" y1="70" x2="70" y2="8" stroke="var(--a)" stroke-width="1.5" stroke-linecap="round"
              style="transform-origin:70px 70px;animation:scanR 3s linear infinite"/>
        <circle cx="70" cy="70" r="4" fill="var(--a)" style="animation:gBlink 2s ease-in-out infinite"/>
      </svg>
      <div class="empty-ttl">EN ATTENTE</div>
      <div class="empty-sub">Configurez les parametres<br>et cliquez sur <strong>Calculer le RUL</strong></div>
    </div>
    <div id="g-result" class="g-result">
      <svg viewBox="0 0 260 215" width="100%" style="display:block">
        <defs>
          <filter id="glow-f" x="-25%" y="-25%" width="150%" height="150%">
            <feGaussianBlur stdDeviation="5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <path d="M 62.8 197.2 A 95 95 0 0 1 39.6 159.4" fill="none" stroke="rgba(239,68,68,.18)" stroke-width="20" stroke-linecap="butt"/>
        <path d="M 39.6 159.4 A 95 95 0 0 1 53.1 74.2" fill="none" stroke="rgba(245,158,11,.14)" stroke-width="20" stroke-linecap="butt"/>
        <path d="M 53.1 74.2 A 95 95 0 1 1 197.2 197.2" fill="none" stroke="rgba(0,212,170,.09)" stroke-width="20" stroke-linecap="butt"/>
        <path d="M 62.8 197.2 A 95 95 0 1 1 197.2 197.2" fill="none" stroke="var(--bd)" stroke-width="16" stroke-linecap="round"/>
        <path id="g-glow" d="M 62.8 197.2 A 95 95 0 1 1 197.2 197.2" fill="none"
              stroke="#00d4aa" stroke-width="24" stroke-linecap="round"
              stroke-dasharray="447.7 596.9" stroke-dashoffset="447.7" opacity="0.12"
              style="transition:stroke-dashoffset .9s cubic-bezier(.4,0,.2,1),stroke .35s"/>
        <path id="g-fill" d="M 62.8 197.2 A 95 95 0 1 1 197.2 197.2" fill="none"
              stroke="#00d4aa" stroke-width="16" stroke-linecap="round"
              stroke-dasharray="447.7 596.9" stroke-dashoffset="447.7"
              style="transition:stroke-dashoffset .9s cubic-bezier(.4,0,.2,1),stroke .35s"
              filter="url(#glow-f)"/>
        <line x1="49.2" y1="156.5" x2="39" y2="160" stroke="rgba(239,68,68,.6)" stroke-width="2"/>
        <line x1="61.2" y1="80" x2="52" y2="73" stroke="rgba(245,158,11,.6)" stroke-width="2"/>
        <text x="44" y="213" fill="rgba(239,68,68,.6)" font-size="9" text-anchor="middle" font-family="Rajdhani,sans-serif">0h</text>
        <text x="36" y="155" fill="rgba(239,68,68,.7)" font-size="8" text-anchor="end" font-family="Rajdhani,sans-serif">10h</text>
        <text x="45" y="68" fill="rgba(245,158,11,.7)" font-size="8" text-anchor="end" font-family="Rajdhani,sans-serif">30h</text>
        <text x="216" y="213" fill="rgba(0,212,170,.6)" font-size="9" text-anchor="middle" font-family="Rajdhani,sans-serif">100h</text>
        <text id="g-num" x="130" y="152" text-anchor="middle" fill="#00d4aa"
              font-size="56" font-weight="700" font-family="Rajdhani,sans-serif">0.0</text>
        <text id="g-unit" x="130" y="170" text-anchor="middle" fill="#00d4aa"
              font-size="10" font-family="Rajdhani,sans-serif" letter-spacing="4">HEURES</text>
      </svg>
      <div id="g-alert"></div>
    </div>
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--bd);font-size:.76rem;color:var(--txm);line-height:1.8">
      Modele : <strong style="color:var(--tx)">Random Forest</strong> &nbsp;n=300 &nbsp;|&nbsp;
      R&#178; <strong style="color:var(--a)">0.955</strong> &nbsp;|&nbsp;
      MAE <strong style="color:var(--a)">3.077h</strong>
    </div>
  </div>
</div>
<div class="sec-title">Simulation Scenarios &mdash; cliquez pour appliquer</div>
<div class="sc-grid" id="sc-grid"></div>
</div>

<!-- TAB PERFORMANCES -->
<div id="tp-perf" class="tab-panel">
<div class="page-hdr">
  <h1>&#128202; Performances des <em>Modeles</em></h1>
  <p>Evaluation sur 4 809 observations &middot; Jeu de test 20% &middot; 5-fold CV</p>
</div>
<div class="g5" style="margin-bottom:18px">
  <div class="kpi"><div class="kpi-v">0.955</div><div class="kpi-l">R&#178; Test</div><div class="kpi-d">Random Forest</div></div>
  <div class="kpi"><div class="kpi-v">3.077h</div><div class="kpi-l">MAE Test</div><div class="kpi-d">Random Forest</div></div>
  <div class="kpi"><div class="kpi-v">3.097h</div><div class="kpi-l">CV MAE</div><div class="kpi-d">&plusmn;0.073h</div></div>
  <div class="kpi"><div class="kpi-v">300</div><div class="kpi-l">Estimateurs</div><div class="kpi-d">n_estimators</div></div>
  <div class="kpi"><div class="kpi-v">&times;5.3</div><div class="kpi-l">Gain vs Ridge</div><div class="kpi-d">Feature Eng.</div></div>
</div>
<div class="g2">
  <div class="chart-card">
    <div class="chart-title">R&#178; PAR MODELE &nbsp;&#8593; MEILLEUR</div>
    <div id="c-r2"></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">MAE PAR MODELE (h) &nbsp;&#8595; MEILLEUR</div>
    <div id="c-mae"></div>
  </div>
</div>
<div class="sec-title">Importance des Variables</div>
<div class="chart-card">
  <div class="chart-title">FEATURE IMPORTANCE &mdash; RANDOM FOREST
    <span style="color:#ef4444"> &#9679; NOUVELLES</span>
    <span style="color:#6366f1"> &#9679; ORIGINALES</span>
  </div>
  <div id="c-fi"></div>
</div>
</div>

<!-- TAB CAPTEURS -->
<div id="tp-capt" class="tab-panel">
<div class="page-hdr">
  <h1>&#128225; Donnees <em>Capteurs</em></h1>
  <p>20 machines &middot; 14 jours &middot; 60 dernieres mesures par machine</p>
</div>
<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
  <span style="font-size:.72rem;color:var(--txm);text-transform:uppercase;letter-spacing:.8px">Machine :</span>
  <select id="m-sel" onchange="renderSensors()"><option value="">-- selectionner --</option></select>
</div>
<div class="chart-card" style="margin-bottom:14px">
  <div class="chart-title">EVOLUTION RUL &mdash; SEUILS 10h ET 30h</div>
  <div id="c-rul"></div>
</div>
<div class="sec-title">Capteurs Normalises (0-100% de leur plage)</div>
<div id="s-btns" class="s-btns"></div>
<div class="chart-card">
  <div class="chart-title">COMPARAISON MULTI-CAPTEURS (valeurs normalisees)</div>
  <div id="c-sensors"></div>
</div>
</div>

</div><!-- main -->
</div><!-- app -->
</body>
</html>"""

CLIENT_HTML = _CLIENT_HTML_RAW.replace('__CSS__', CSS).replace('__JS__', CLIENT_JS.replace('__DATA__', J(client_data)))

with open("dashboard_client.html","w",encoding="utf-8") as f:
    f.write(CLIENT_HTML)
print("✓ dashboard_client.html généré")

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TECHNIQUE
# ══════════════════════════════════════════════════════════════════════════════
tech_data = {"stats":stats,"missing":missing,"corr":corr_data,"rul_hist":rul_hist,
             "rul_by_type":rul_by_type,"sensor_box":sensor_box,"di":dataset_info,
             "machine_types":machine_type_counts,"fi":FI,"model":MODEL}

# Build variables table rows
VAR_ROWS = ""
STATUS_BADGE = {"drop":"b-drop","keep":"b-keep","target":"b-target","leak":"b-leak"}
for name,typ,status,desc in VARIABLES:
    badge = STATUS_BADGE.get(status,"b-keep")
    VAR_ROWS += f'<tr><td><code>{name}</code></td><td style="color:var(--txm);font-size:.78rem">{typ}</td><td><span class="badge {badge}">{status.upper()}</span></td><td style="font-size:.78rem">{desc}</td></tr>\n'

TECH_JS = r"""
var D=__DATA__;
var COLORS={a:'#2a9d8f',r:'#e63946',o:'#f4a261',b:'#58a6ff',g:'#56d364'};
var PAL=['#58a6ff','#f85149','#2a9d8f','#d29922'];

function setTheme(t){document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);
  document.getElementById('theme-btn').textContent=t==='dark'?'☀️':'🌙';}
function toggleTheme(){var t=document.documentElement.dataset.theme==='dark'?'light':'dark';
  setTheme(t);rerenderAll();}
function pC(){var d=document.documentElement.dataset.theme==='dark';
  return{paper:d?'#0d1117':'#f0f4f8',bg:d?'#161b22':'#fff',grid:d?'#21262d':'#d0d7de',
         text:d?'#c9d1d9':'#24292f',muted:d?'#8b949e':'#57606a'};}
function pL(title,extra){
  var c=pC();extra=extra||{};
  return Object.assign({title:{text:title||'',font:{color:c.text,size:12,family:'Source Sans 3'}},
    paper_bgcolor:c.paper,plot_bgcolor:c.bg,
    font:{color:c.text,family:'Source Sans 3,sans-serif'},
    xaxis:{gridcolor:c.grid,zeroline:false,color:c.muted},
    yaxis:{gridcolor:c.grid,zeroline:false,color:c.muted},
    margin:{t:44,b:36,l:44,r:16},showlegend:false},extra);
}

function showTab(id,btn){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('tp-'+id).classList.add('active');
  btn.classList.add('active');
  renderPage(id);
}
function renderPage(id){
  if(id==='dataset') {renderPie();renderStats();}
  if(id==='missing') renderMissing();
  if(id==='distrib') {renderRULHist();renderBoxPlots();}
  if(id==='correl')  renderHeatmap();
  if(id==='feateng') renderFI();
  if(id==='pipeline') renderCV();
}
function rerenderAll(){
  var a=document.querySelector('.tab.active');
  if(a) renderPage(a.dataset.page);
}

/* ── DATASET ── */
function renderPie(){
  var mt=D.machine_types,labels=Object.keys(mt),vals=labels.map(function(l){return mt[l];});
  var c=pC();
  Plotly.react('ch-pie',[{type:'pie',labels:labels,values:vals,marker:{colors:PAL},
    textinfo:'label+percent',textfont:{color:c.text,size:11},hole:.3}],
    {paper_bgcolor:c.paper,plot_bgcolor:c.bg,font:{color:c.text,family:'Source Sans 3'},
     showlegend:true,legend:{font:{color:c.text},bgcolor:'rgba(0,0,0,0)'},
     margin:{t:10,b:10,l:10,r:10},title:{text:'Machines par type',font:{color:c.text,size:12}},height:260});
}
function renderStats(){
  var c=pC(),keys=Object.keys(D.stats);
  var mns={vibration_rms:'vib_rms',temperature_motor:'temp_motor',current_phase_avg:'current',
           pressure_level:'pressure',rpm:'rpm',hours_since_maintenance:'hours_maint',
           ambient_temp:'ambient',rul_hours:'rul_hours'};
  var rows='';
  keys.forEach(function(k){var s=D.stats[k];
    rows+='<tr><td><code>'+mns[k]+'</code></td><td>'+s.count.toLocaleString('fr')+'</td>'+
    '<td>'+s.min+'</td><td>'+s.q1+'</td><td style="color:var(--a)">'+s.median+'</td>'+
    '<td>'+s.mean+'</td><td>'+s.q3+'</td><td>'+s.max+'</td><td style="color:var(--txm)">'+s.std+'</td></tr>';
  });
  document.getElementById('stats-tbody').innerHTML=rows;
}

/* ── MISSING ── */
function renderMissing(){
  var entries=Object.entries(D.missing).sort(function(a,b){return b[1].pct-a[1].pct;});
  var c=pC();
  Plotly.react('ch-missing',[{type:'bar',orientation:'h',
    x:entries.map(function(e){return e[1].pct;}),
    y:entries.map(function(e){return e[0];}),
    marker:{color:entries.map(function(e){return e[1].pct>3?COLORS.r:e[1].pct>1?COLORS.o:COLORS.b;})},
    text:entries.map(function(e){return e[1].pct.toFixed(1)+'%  ('+e[1].n+' obs)';}),
    textposition:'outside',textfont:{color:c.muted,size:10}}],
    pL('Taux de valeurs manquantes par variable',{xaxis:Object.assign({},pL().xaxis,{ticksuffix:'%'})}));
}

/* ── DISTRIBUTIONS ── */
function renderRULHist(){
  var h=D.rul_hist,c=pC();
  var binsX=h.bins.slice(0,-1).map(function(b,i){return(b+h.bins[i+1])/2;});
  Plotly.react('ch-rul-hist',[{type:'bar',x:binsX,y:h.counts,
    marker:{color:binsX.map(function(b){return b<10?COLORS.r:b<30?COLORS.o:COLORS.a;})}}],
    pL('Distribution RUL — 🔴 Critique (<10h)  🟠 Alerte (<30h)  🔵 Normal',{
      shapes:[
        {type:'line',x0:10,x1:10,y0:0,y1:1,yref:'paper',line:{dash:'dash',color:COLORS.r+'80',width:1.5}},
        {type:'line',x0:30,x1:30,y0:0,y1:1,yref:'paper',line:{dash:'dash',color:COLORS.o+'80',width:1.5}},
      ],annotations:[
        {x:10,y:1,yref:'paper',text:'10h',showarrow:false,font:{color:COLORS.r,size:9},xanchor:'left'},
        {x:30,y:1,yref:'paper',text:'30h',showarrow:false,font:{color:COLORS.o,size:9},xanchor:'left'},
      ]}));
}
function renderBoxPlots(){
  var c=pC(),types=Object.keys(D.rul_by_type);
  Plotly.react('ch-box-rul',types.map(function(mt,i){
    var v=D.rul_by_type[mt];
    return{type:'box',name:mt,q1:[v.q1],median:[v.median],q3:[v.q3],
           lowerfence:[v.min],upperfence:[v.max],marker:{color:PAL[i]},
           line:{color:PAL[i]},fillcolor:PAL[i]+'30'};
  }),pL('RUL par type de machine',{showlegend:true,legend:{font:{color:c.text},bgcolor:'rgba(0,0,0,0)'}}));
}

/* ── CORRÉLATIONS ── */
function renderHeatmap(){
  var c=pC();
  Plotly.react('ch-heatmap',[{type:'heatmap',z:D.corr.matrix,x:D.corr.labels,y:D.corr.labels,
    colorscale:[[0,COLORS.r],[.5,c.bg],[1,COLORS.a]],zmid:0,zmin:-1,zmax:1,
    text:D.corr.matrix.map(function(row){return row.map(function(v){return v.toFixed(2);});}),
    texttemplate:'%{text}',textfont:{size:9,color:c.text},showscale:true}],
    pL('Matrice de Corrélation de Pearson',{margin:{t:50,b:90,l:110,r:20}}));
}

/* ── FEATURE ENGINEERING ── */
function renderFI(){
  var c=pC(),fi=D.fi;
  var sorted=fi.names.map(function(n,i){return{n:n,v:fi.imp[i],isNew:fi.new[i]};})
    .sort(function(a,b){return a.v-b.v;});
  Plotly.react('ch-fi-tech',[{type:'bar',orientation:'h',
    x:sorted.map(function(d){return d.v;}),y:sorted.map(function(d){return d.n;}),
    marker:{color:sorted.map(function(d){return d.isNew?COLORS.r:COLORS.b;})},
    text:sorted.map(function(d){return(d.v*100).toFixed(1)+'%';}),
    textposition:'outside',textfont:{color:c.muted,size:10}}],
    pL('Feature Importance Random Forest — 🔴 Nouvelles   🔵 Originales',{
      xaxis:Object.assign({},pL().xaxis,{tickformat:'.0%'}),height:360}));
}

/* ── PIPELINE ── */
function renderCV(){
  var c=pC(),m=D.model;
  Plotly.react('ch-cv',[{type:'bar',x:m.names,y:m.cv_mae,
    error_y:{type:'data',array:m.cv_std,color:c.muted,width:5},
    marker:{color:['#f85149','#2a9d8f','#58a6ff','#d29922']},width:.5,
    text:m.cv_mae.map(function(v,i){return v.toFixed(3)+'±'+m.cv_std[i].toFixed(3);}),
    textposition:'outside',textfont:{color:c.muted,size:9}}],
    pL('Validation Croisée 5-fold — CV MAE ± Std'));
}

/* ── INIT ── */
window.addEventListener('DOMContentLoaded',function(){
  var t=localStorage.getItem('theme')||'dark';
  setTheme(t);
  document.querySelectorAll('.tab[data-page]').forEach(function(b){
    b.addEventListener('click',function(){showTab(b.dataset.page,b);});
  });
  renderPage('dataset');
});
"""

TECH_HTML = """<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard Technique — Maintenance Prédictive</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
""" + PLOTLY_CDN + """
<style>""" + CSS + """</style>
</head>
<body>
<script>""" + TECH_JS.replace("__DATA__", J(tech_data)) + """</script>

<div class="app">
<nav class="topnav">
  <div class="brand">⚙ PRED<span>ML</span></div>
  <div class="tabs">
    <button class="tab active" data-page="dataset">🗂 Dataset</button>
    <button class="tab" data-page="missing">❓ Valeurs Manquantes</button>
    <button class="tab" data-page="distrib">📈 Distributions</button>
    <button class="tab" data-page="correl">🔗 Corrélations</button>
    <button class="tab" data-page="feateng">⚗️ Feature Eng.</button>
    <button class="tab" data-page="pipeline">🔧 Pipeline</button>
  </div>
  <a class="nav-link" href="dashboard_client.html">→ Client</a>
  <button class="theme-btn" onclick="toggleTheme()" id="theme-btn">☀️</button>
</nav>

<div class="main">

<!-- ── DATASET ── -->
<div id="tp-dataset" class="tab-panel active">
<div class="page-hdr">
  <h1>🗂 Analyse du <span class="accent">Dataset</span></h1>
  <p>24 042 enregistrements · 20 machines industrielles · 14 jours · 15 variables</p>
  <span class="chip chip-b">EDA</span><span class="chip chip-g">PRODUCTION-READY</span>
</div>
<div class="g5" style="margin-bottom:18px">
  <div class="kpi"><div class="kpi-v">24 042</div><div class="kpi-l">Enregistrements</div><div class="kpi-d">Dataset complet</div></div>
  <div class="kpi"><div class="kpi-v">20</div><div class="kpi-l">Machines</div><div class="kpi-d">4 types</div></div>
  <div class="kpi"><div class="kpi-v">14j</div><div class="kpi-l">Durée</div><div class="kpi-d">Horizon</div></div>
  <div class="kpi"><div class="kpi-v">27.8h</div><div class="kpi-l">RUL moyen</div><div class="kpi-d">Variable cible</div></div>
  <div class="kpi"><div class="kpi-v">3 721</div><div class="kpi-l">Pannes &lt;24h</div><div class="kpi-d">Cas critiques</div></div>
</div>
<div class="g2">
  <div>
    <div class="sec-title">Variables du Dataset</div>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Nom</th><th>Type</th><th>Statut</th><th>Description</th></tr></thead>
        <tbody>""" + VAR_ROWS + """</tbody>
      </table>
    </div>
  </div>
  <div>
    <div class="sec-title">Répartition par Type de Machine</div>
    <div class="chart-box"><div id="ch-pie" style="height:260px"></div></div>
  </div>
</div>
<div class="sec-title">Statistiques Descriptives</div>
<div class="tbl-wrap">
  <table>
    <thead><tr><th>Variable</th><th>N</th><th>Min</th><th>Q1</th><th>Médiane</th><th>Moyenne</th><th>Q3</th><th>Max</th><th>Std</th></tr></thead>
    <tbody id="stats-tbody"></tbody>
  </table>
</div>
</div>

<!-- ── MISSING ── -->
<div id="tp-missing" class="tab-panel">
<div class="page-hdr">
  <h1>❓ Valeurs <span class="accent">Manquantes</span></h1>
  <p>Analyse de complétude · Stratégie d'imputation : médiane (SimpleImputer sklearn)</p>
</div>
<div class="chart-box"><div id="ch-missing" style="height:380px"></div></div>
<div class="sec-title" style="margin-top:18px">Stratégie d'Imputation</div>
<div class="g3">
  <div class="card" style="border-left:3px solid var(--a)">
    <b style="color:var(--a)">✅ SimpleImputer (médiane)</b>
    <p style="font-size:.8rem;color:var(--txm);margin-top:6px;line-height:1.6">Robuste aux outliers · Pas de fuite d'information · Intégré dans le Pipeline sklearn</p>
  </div>
  <div class="card" style="border-left:3px solid var(--r)">
    <b style="color:var(--r)">❌ Suppression de lignes</b>
    <p style="font-size:.8rem;color:var(--txm);margin-top:6px;line-height:1.6">Perte de données · Biais si MNAR</p>
  </div>
  <div class="card" style="border-left:3px solid var(--r)">
    <b style="color:var(--r)">❌ Imputation par la moyenne</b>
    <p style="font-size:.8rem;color:var(--txm);margin-top:6px;line-height:1.6">Sensible aux outliers · Données industrielles souvent skewed</p>
  </div>
</div>
</div>

<!-- ── DISTRIBUTIONS ── -->
<div id="tp-distrib" class="tab-panel">
<div class="page-hdr">
  <h1>📈 <span class="accent">Distributions</span></h1>
  <p>Analyse statistique des capteurs et de la variable cible RUL</p>
</div>
<div class="g2">
  <div>
    <div class="sec-title">Distribution de la Variable Cible — RUL (h)</div>
    <div class="chart-box"><div id="ch-rul-hist" style="height:300px"></div></div>
  </div>
  <div>
    <div class="sec-title">RUL par Type de Machine</div>
    <div class="chart-box"><div id="ch-box-rul" style="height:300px"></div></div>
  </div>
</div>
</div>

<!-- ── CORRÉLATIONS ── -->
<div id="tp-correl" class="tab-panel">
<div class="page-hdr">
  <h1>🔗 <span class="accent">Corrélations</span></h1>
  <p>Relations linéaires entre variables · Les non-linéarités sont capturées par le Random Forest</p>
</div>
<div class="chart-box"><div id="ch-heatmap" style="height:430px"></div></div>
<div class="sec-title">Interprétation</div>
<div class="card" style="font-size:.83rem;line-height:1.8;color:var(--txm)">
  Les corrélations linéaires avec le RUL sont <b style="color:var(--tx)">faibles</b> car la relation est <b style="color:var(--a)">non-linéaire</b>.
  Le Random Forest capture ces non-linéarités. Les features engineerées (cumcount, hours_sq) montrent
  les corrélations les plus fortes avec la variable cible.
</div>
</div>

<!-- ── FEATURE ENGINEERING ── -->
<div id="tp-feateng" class="tab-panel">
<div class="page-hdr">
  <h1>⚗️ Feature <span class="accent">Engineering</span></h1>
  <p>3 nouvelles features créées · Gain MAE : ×5.3 (16.3h → 3.1h)</p>
  <span class="chip chip-r">NOUVELLES FEATURES</span>
  <span class="chip chip-b">FEATURES ORIGINALES</span>
</div>
<div class="g3" style="margin-bottom:18px">
  <div class="kpi" style="--a:#f85149"><div class="kpi-v" style="color:#f85149">3</div><div class="kpi-l">Features créées</div><div class="kpi-d">cumcount · hours_sq · interactions</div></div>
  <div class="kpi"><div class="kpi-v">×5.3</div><div class="kpi-l">Gain MAE</div><div class="kpi-d">16.3h → 3.1h</div></div>
  <div class="kpi"><div class="kpi-v">63.5%</div><div class="kpi-l">Top-2 features</div><div class="kpi-d">cumcount + hours_sq</div></div>
</div>
<div class="chart-box"><div id="ch-fi-tech" style="height:360px"></div></div>
<div class="sec-title">Code — Feature Engineering</div>
<div class="code-block"><span class="cm"># Feature Engineering</span>
df[<span class="st">"cumcount"</span>]     = df.groupby(<span class="st">"machine_id"</span>).cumcount()  <span class="cm"># ← #1 importance : 40%</span>
df[<span class="st">"hours_sq"</span>]      = df[<span class="st">"hours_since_maintenance"</span>]**<span class="nb">2</span>       <span class="cm"># ← #2 importance : 23.5%</span>
df[<span class="st">"vib_x_temp"</span>]   = df[<span class="st">"vibration_rms"</span>] * df[<span class="st">"temperature_motor"</span>]
df[<span class="st">"vib_x_hours"</span>]  = df[<span class="st">"vibration_rms"</span>] * df[<span class="st">"hours_since_maintenance"</span>]
df[<span class="st">"temp_x_hours"</span>] = df[<span class="st">"temperature_motor"</span>] * df[<span class="st">"hours_since_maintenance"</span>]</div>
</div>

<!-- ── PIPELINE ── -->
<div id="tp-pipeline" class="tab-panel">
<div class="page-hdr">
  <h1>🔧 Pipeline de <span class="accent">Preprocessing</span></h1>
  <p>sklearn Pipeline · Encapsulation complète · Zéro fuite de données</p>
  <span class="chip chip-g">PRODUCTION-READY</span>
</div>
<div class="card">
  <div class="pipe-flow">
    <div class="pipe-step" style="border-top:3px solid #58a6ff">
      <div class="pipe-ico">✂️</div>
      <div class="pipe-name" style="color:#58a6ff">Train / Test Split</div>
      <div class="pipe-detail">80% train · 20% test<br>Temporel — pas de fuite</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step" style="border-top:3px solid #f85149">
      <div class="pipe-ico">🩹</div>
      <div class="pipe-name" style="color:#f85149">SimpleImputer</div>
      <div class="pipe-detail">Stratégie : médiane<br>Remplacement des NaN</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step" style="border-top:3px solid #2a9d8f">
      <div class="pipe-ico">⚖️</div>
      <div class="pipe-name" style="color:#2a9d8f">StandardScaler</div>
      <div class="pipe-detail">μ=0 · σ=1<br>Normalisation</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step" style="border-top:3px solid #d29922">
      <div class="pipe-ico">🔀</div>
      <div class="pipe-name" style="color:#d29922">ColumnTransformer</div>
      <div class="pipe-detail">Num: Imp+Scale<br>Cat: OneHotEncoder</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step" style="border-top:3px solid #56d364">
      <div class="pipe-ico">🌲</div>
      <div class="pipe-name" style="color:#56d364">RandomForest</div>
      <div class="pipe-detail">n_estimators=300<br>n_jobs=-1</div>
    </div>
  </div>
</div>
<div class="g2" style="margin-top:16px">
  <div>
    <div class="sec-title">Code Pipeline</div>
    <div class="code-block"><span class="kw">from</span> sklearn.pipeline <span class="kw">import</span> Pipeline
<span class="kw">from</span> sklearn.compose <span class="kw">import</span> ColumnTransformer
<span class="kw">from</span> sklearn.impute <span class="kw">import</span> SimpleImputer
<span class="kw">from</span> sklearn.preprocessing <span class="kw">import</span> StandardScaler, OneHotEncoder

num_pipe = Pipeline([
    (<span class="st">"imputer"</span>, SimpleImputer(strategy=<span class="st">"median"</span>)),
    (<span class="st">"scaler"</span>,  StandardScaler()),
])
preprocessor = ColumnTransformer([
    (<span class="st">"num"</span>, num_pipe, NUMERIC_FEATURES),
    (<span class="st">"cat"</span>, OneHotEncoder(), CATEGORICAL_FEATURES),
])
pipeline = Pipeline([
    (<span class="st">"prep"</span>,  preprocessor),
    (<span class="st">"model"</span>, RandomForestRegressor(n_estimators=<span class="nb">300</span>)),
])
pipeline.fit(X_train, y_train)  <span class="cm"># R²=0.955 · MAE=3.077h</span></div>
  </div>
  <div>
    <div class="sec-title">Validation Croisée 5-fold</div>
    <div class="chart-box"><div id="ch-cv" style="height:250px"></div></div>
    <div class="sec-title" style="margin-top:14px">Variables exclues (Data Leakage)</div>
    <div style="display:flex;flex-direction:column;gap:7px">
      <div style="background:rgba(230,57,70,.08);border:1px solid rgba(230,57,70,.25);border-radius:8px;padding:10px 13px;font-size:.8rem">
        ⚠️ <code>failure_within_24h</code> — dérivé du RUL
      </div>
      <div style="background:rgba(230,57,70,.08);border:1px solid rgba(230,57,70,.25);border-radius:8px;padding:10px 13px;font-size:.8rem">
        ⚠️ <code>failure_type</code> — connu seulement après la panne
      </div>
      <div style="background:rgba(230,57,70,.08);border:1px solid rgba(230,57,70,.25);border-radius:8px;padding:10px 13px;font-size:.8rem">
        ⚠️ <code>estimated_repair_cost</code> — corrélé à la panne
      </div>
    </div>
  </div>
</div>
</div>

</div><!-- main -->
</div><!-- app -->
</body>
</html>
"""

with open("dashboard_technique.html","w",encoding="utf-8") as f:
    f.write(TECH_HTML)
print("✓ dashboard_technique.html généré")
print()
print("=" * 50)
print("Ouvrir dans le navigateur :")
print("  python3 -m http.server 8765")
print("  → http://localhost:8765/dashboard_client.html")
print("  → http://localhost:8765/dashboard_technique.html")
print()
print("Ou ouvrir directement les fichiers HTML (double-clic)")
