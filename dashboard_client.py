"""
Dashboard Client — Maintenance Prédictive Industrielle
Interface décisionnelle pour les responsables maintenance

Lancer : streamlit run dashboard_client.py
"""

import math
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, os

st.set_page_config(
    page_title="Maintenance Prédictive — RUL",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Source Sans 3', sans-serif; }
h1,h2,h3,h4 { font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; }

.stApp { background: #090b0f; color: #f0f2f5; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10141c 0%, #0c0f16 100%) !important;
    border-right: 1px solid #1e2736 !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e2736 !important; }
section[data-testid="stSidebar"] strong { color: #2a9d8f !important; }

/* ── st.metric override ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#141820,#111520) !important;
    border: 1px solid #1e2736 !important;
    border-radius: 12px !important; padding: 14px !important;
}
div[data-testid="stMetricLabel"] p { color: #8892a4 !important; font-size:.75rem !important; text-transform:uppercase; letter-spacing:1px; }
div[data-testid="stMetricValue"]   { color: #2a9d8f !important; font-family:'Rajdhani',sans-serif !important; font-size:1.8rem !important; font-weight:700 !important; }
div[data-testid="stMetricDelta"]   { color: #56d364 !important; font-size:.8rem !important; }

/* ── Button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2a9d8f,#1d7a6e) !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
    letter-spacing: 2px !important; font-size: 1rem !important;
    padding: 12px !important;
    box-shadow: 0 4px 20px #2a9d8f35 !important;
    transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px #2a9d8f55 !important;
}

/* ── Expander ── */
details { background: #141820 !important; border: 1px solid #1e2736 !important; border-radius: 10px !important; }
details summary { color: #e2e8f0 !important; font-family: 'Rajdhani', sans-serif !important; font-size: 1rem !important; font-weight: 600 !important; letter-spacing: 1px !important; }

/* ── Section title ── */
.sec-title {
    font-family: 'Rajdhani', sans-serif; font-size: 1.2rem; font-weight: 700;
    color: #2a9d8f; border-left: 3px solid #2a9d8f;
    padding-left: 12px; margin: 24px 0 16px 0; letter-spacing: 1px;
}

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg,#0f1a2e 0%,#0c1a1a 60%,#090b0f 100%);
    border: 1px solid #1e2736; border-radius: 16px;
    padding: 24px 32px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.page-header::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,transparent,#2a9d8f,transparent);
}
.page-header h1 { font-family:'Rajdhani',sans-serif; font-size:2rem; font-weight:700;
    color:#f0f2f5; margin:0; letter-spacing:2px; }
.page-header h1 span { color:#2a9d8f; }
.page-header p { font-size:.85rem; color:#8892a4; margin:4px 0 0 0; letter-spacing:1px; }
.header-chip {
    display:inline-block; background:#2a9d8f18; border:1px solid #2a9d8f55;
    border-radius:20px; padding:3px 14px; font-size:.72rem; color:#2a9d8f;
    font-family:'Rajdhani',sans-serif; letter-spacing:2px; margin-top:10px;
}

/* ── Sidebar logo ── */
.sb-logo { font-family:'Rajdhani',sans-serif; font-size:1.5rem; font-weight:700; color:#f0f2f5; letter-spacing:2px; }
.sb-logo span { color:#2a9d8f; }
.sb-sub  { font-size:.65rem; color:#4a5568; letter-spacing:2px; text-transform:uppercase; margin-bottom:16px; }
.model-chip { background:#2a9d8f14; border:1px solid #2a9d8f40; border-radius:10px; padding:12px 14px; }
.model-chip .mn  { font-size:.95rem; font-weight:700; color:#2a9d8f; font-family:'Rajdhani',sans-serif; }
.model-chip .mst { font-size:.78rem; color:#8892a4; margin-top:3px; }
.model-chip .mst b { color:#e2e8f0; font-weight:600; }
.fleet-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
.fc { display:inline-block; padding:3px 10px; border-radius:12px; font-size:.65rem; font-family:'Rajdhani',sans-serif; letter-spacing:1px; }
.fc-ok   { background:#2a9d8f18; color:#2a9d8f; border:1px solid #2a9d8f50; }
.fc-warn { background:#f4a26118; color:#f4a261; border:1px solid #f4a26150; }
.fc-crit { background:#e6394618; color:#e63946; border:1px solid #e6394650; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">⚙ PRED<span>ML</span></div>
    <div class="sb-sub">Maintenance Prédictive · Industrielle</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🎯  Prédiction RUL",
        "📊  Performances Modèles",
        "📡  Données Capteurs",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div class="model-chip">
        <div class="mn">🏆 Random Forest</div>
        <div class="mst">R² <b>0.955</b></div>
        <div class="mst">MAE <b>3.077 h</b></div>
        <div class="mst">CV MAE <b>3.097 ± 0.073 h</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:.68rem;color:#4a5568;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Flotte surveillée</div>
    <div class="fleet-row">
        <span class="fc fc-ok">✓ 12 OK</span>
        <span class="fc fc-warn">⚠ 5 Alerte</span>
        <span class="fc fc-crit">✕ 3 Critique</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:.62rem;color:#2a3040;letter-spacing:1px">v2.0 · 24 042 obs · 20 machines</div>
    """, unsafe_allow_html=True)

# ─── Data & model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    p = "models/best_model_pipeline.pkl"
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("predictive_maintenance_v3.csv", parse_dates=["timestamp"])
        return df.sort_values(["machine_id","timestamp"]).reset_index(drop=True)
    except:
        return None

pipeline = load_model()
df_raw   = load_data()

TEAL="#2a9d8f"; RED="#e63946"; ORANGE="#f4a261"; BLUE="#58a6ff"; PURPLE="#a78bfa"
DARK="#090b0f"; CARD="#141820"; BORDER="#1e2736"

def dark_fig(fig, h=350):
    fig.update_layout(height=h, paper_bgcolor=DARK, plot_bgcolor=CARD,
        font=dict(color="#f0f2f5",family="Source Sans 3"),
        margin=dict(t=40,b=20,l=10,r=10),
        xaxis=dict(gridcolor=BORDER,zeroline=False),
        yaxis=dict(gridcolor=BORDER,zeroline=False))
    return fig

def prepare_input(vib, temp, curr, pres, rpm_v, hours, ambient, cumcount):
    return pd.DataFrame([{
        "vibration_rms":vib,"temperature_motor":temp,"current_phase_avg":curr,
        "pressure_level":pres,"rpm":rpm_v,"hours_since_maintenance":hours,
        "ambient_temp":ambient,"cumcount":cumcount,"hours_sq":hours**2,
        "vib_x_temp":vib*temp,"vib_x_hours":vib*hours,"temp_x_hours":temp*hours,
    }])

# ─── SVG Gauge HTML Component ─────────────────────────────────────────────────
def make_gauge_html(rul, color, bg="#090b0f"):
    pct  = min(max(rul, 0) / 100, 1.0)
    cx, cy, r = 130, 120, 100

    def pt(deg):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)

    sx, sy = pt(225)
    ex, ey = pt(315)
    arc    = 2 * math.pi * r * 0.75
    offset = arc * (1 - pct)
    path   = f"M {sx:.3f} {sy:.3f} A {r} {r} 0 1 1 {ex:.3f} {ey:.3f}"

    # compute threshold tick positions (10h and 30h)
    def tick(h):
        a  = 225 - (h/100)*270
        tx, ty = pt(a)
        ix, iy = cx + (r-18)*math.cos(math.radians(a)), cy - (r-18)*math.sin(math.radians(a))
        return tx, ty, ix, iy

    t10x, t10y, i10x, i10y = tick(10)
    t30x, t30y, i30x, i30y = tick(30)

    zone_color = RED if rul < 10 else (ORANGE if rul < 30 else TEAL)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Source+Sans+3&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ height:100%; background:{bg}; }}
  body {{ display:flex; justify-content:center; align-items:center; overflow:hidden; }}

  .gauge-track  {{ fill:none; stroke:#1a2030; stroke-width:20; stroke-linecap:round; }}
  .gauge-zone-crit  {{ fill:none; stroke:#e6394618; stroke-width:20; stroke-linecap:butt; }}
  .gauge-zone-warn  {{ fill:none; stroke:#f4a26114; stroke-width:20; stroke-linecap:butt; }}
  .gauge-zone-ok    {{ fill:none; stroke:#2a9d8f10; stroke-width:20; stroke-linecap:butt; }}

  .gauge-prog {{
    fill: none;
    stroke: {color};
    stroke-width: 20;
    stroke-linecap: round;
    stroke-dasharray: {arc:.3f};
    stroke-dashoffset: {arc:.3f};
    transition: stroke-dashoffset 1.5s cubic-bezier(.34,1.56,.64,1);
    filter: drop-shadow(0 0 10px {color}70);
  }}
  .gauge-inner-ring {{ fill:none; stroke:#1a2030; stroke-width:1.5; }}
  .gauge-bg         {{ fill:{bg}; }}
  .val-text {{
    font-family:'Rajdhani',sans-serif; font-size:52px; font-weight:700;
    fill:{color}; text-anchor:middle; dominant-baseline:middle;
  }}
  .lbl-text {{
    font-family:'Source Sans 3',sans-serif; font-size:10px;
    fill:#8892a4; text-anchor:middle; text-transform:uppercase; letter-spacing:2.5px;
  }}
  .axis-lbl {{ font-size:8.5px; fill:#3a4050; text-anchor:middle; font-family:monospace; }}
  .tick-line {{ stroke:#3a4050; stroke-width:1.5; stroke-linecap:round; }}
</style>
</head>
<body>
<svg viewBox="0 0 260 210" width="280" height="225" overflow="visible">
  <defs>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- Background track -->
  <path d="{path}" class="gauge-track"/>

  <!-- Zone coloring (subtle overlay arcs) -->
  <!-- critical 0-10% zone -->
  <path d="{path}" fill="none" stroke="#e6394618" stroke-width="20"
        stroke-dasharray="{arc*0.10:.2f} {arc:.2f}" stroke-dashoffset="0" stroke-linecap="butt"/>
  <!-- warning 10-30% zone -->
  <path d="{path}" fill="none" stroke="#f4a26112" stroke-width="20"
        stroke-dasharray="{arc*0.20:.2f} {arc:.2f}" stroke-dashoffset="{-arc*0.10:.2f}" stroke-linecap="butt"/>
  <!-- ok 30-100% zone -->
  <path d="{path}" fill="none" stroke="#2a9d8f0c" stroke-width="20"
        stroke-dasharray="{arc*0.70:.2f} {arc:.2f}" stroke-dashoffset="{-arc*0.30:.2f}" stroke-linecap="butt"/>

  <!-- Progress arc -->
  <path id="gp" d="{path}" class="gauge-prog" filter="url(#glow)"/>

  <!-- Threshold ticks -->
  <line x1="{t10x:.2f}" y1="{t10y:.2f}" x2="{i10x:.2f}" y2="{i10y:.2f}" class="tick-line" stroke="{RED}90"/>
  <line x1="{t30x:.2f}" y1="{t30y:.2f}" x2="{i30x:.2f}" y2="{i30y:.2f}" class="tick-line" stroke="{ORANGE}90"/>

  <!-- Inner decorative rings -->
  <circle cx="{cx}" cy="{cy}" r="74" class="gauge-inner-ring"/>
  <circle cx="{cx}" cy="{cy}" r="72" class="gauge-bg"/>

  <!-- RUL counter -->
  <text id="gval" x="{cx}" y="{cy + 4}" class="val-text">0.0h</text>
  <text x="{cx}" y="{cy + 36}" class="lbl-text">Durée de vie restante</text>

  <!-- Axis labels -->
  <text x="{sx - 6:.0f}" y="{sy + 18:.0f}" class="axis-lbl">0h</text>
  <text x="{ex + 6:.0f}" y="{ey + 18:.0f}" class="axis-lbl">100h</text>

  <!-- Threshold labels -->
  <text x="{i10x:.0f}" y="{i10y - 6:.0f}" fill="{RED}80" font-size="7.5" text-anchor="middle" font-family="monospace">10h</text>
  <text x="{i30x:.0f}" y="{i30y - 6:.0f}" fill="{ORANGE}80" font-size="7.5" text-anchor="middle" font-family="monospace">30h</text>

  <!-- Center dot -->
  <circle cx="{cx}" cy="{cy}" r="4" fill="{color}" opacity="0.7"/>
</svg>

<script>
window.addEventListener('load', () => {{
  setTimeout(() => {{
    document.getElementById('gp').style.strokeDashoffset = '{offset:.3f}';
  }}, 120);

  const target = {rul:.2f};
  const dur    = 1500;
  const t0     = performance.now();
  const ease   = t => 1 - Math.pow(1 - t, 3);

  (function tick(now) {{
    const f   = Math.min((now - t0) / dur, 1);
    const val = target * ease(f);
    document.getElementById('gval').textContent = val.toFixed(1) + 'h';
    if (f < 1) requestAnimationFrame(tick);
  }})(t0);
}});
</script>
</body>
</html>"""

# ─── Scenarios HTML Component ─────────────────────────────────────────────────
def make_scenarios_html(scenarios, bg="#090b0f"):
    """
    scenarios : list of (name, rul, color, status_label)
    """
    r_m = 34
    cx_m, cy_m = 44, 42
    sx_m = cx_m + r_m * math.cos(math.radians(225))
    sy_m = cy_m - r_m * math.sin(math.radians(225))
    ex_m = cx_m + r_m * math.cos(math.radians(315))
    ey_m = cy_m - r_m * math.sin(math.radians(315))
    arc_m = 2 * math.pi * r_m * 0.75
    path_m = f"M {sx_m:.2f} {sy_m:.2f} A {r_m} {r_m} 0 1 1 {ex_m:.2f} {ey_m:.2f}"

    cards_html = ""
    js_inits   = ""
    for i, (name, rul, color, status) in enumerate(scenarios):
        pct    = min(max(rul,0)/100, 1.0)
        offset = arc_m * (1 - pct)
        cards_html += f"""
        <div class="sc-card" style="--col:{color}; --border:{color}30">
            <div class="sc-name">{name}</div>
            <div class="sc-gauge-wrap">
                <svg viewBox="0 0 88 78" width="96" height="85">
                    <path d="{path_m}" fill="none" stroke="#1a2030" stroke-width="7" stroke-linecap="round"/>
                    <path id="sp{i}" d="{path_m}" fill="none" stroke="{color}" stroke-width="7"
                          stroke-linecap="round"
                          stroke-dasharray="{arc_m:.2f}" stroke-dashoffset="{arc_m:.2f}"
                          style="transition:stroke-dashoffset 1.5s cubic-bezier(.34,1.56,.64,1);
                                 filter:drop-shadow(0 0 5px {color}60)"/>
                    <circle cx="{cx_m}" cy="{cy_m}" r="22" fill="{bg}"/>
                    <text id="sv{i}" x="{cx_m}" y="{cy_m + 5}" fill="{color}"
                          font-family="Rajdhani,sans-serif" font-size="15" font-weight="700"
                          text-anchor="middle">0h</text>
                </svg>
            </div>
            <div class="sc-status" style="color:{color}">{status}</div>
        </div>
        """
        js_inits += f"""
        setTimeout(() => {{
            document.getElementById('sp{i}').style.strokeDashoffset = '{offset:.2f}';
        }}, {100 + i*120});
        animateNum('sv{i}', {rul:.1f}, {1200 + i*120});
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ height:100%; background:{bg}; font-family:'Rajdhani',sans-serif; }}
  body {{ display:flex; justify-content:center; align-items:center; padding:8px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    width: 100%;
  }}
  .sc-card {{
    background: linear-gradient(135deg,#141820,#111520);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 10px 12px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
    animation: fadeUp .5s ease both;
  }}
  .sc-card:nth-child(1) {{ animation-delay:.05s }}
  .sc-card:nth-child(2) {{ animation-delay:.15s }}
  .sc-card:nth-child(3) {{ animation-delay:.25s }}
  .sc-card:nth-child(4) {{ animation-delay:.35s }}
  .sc-card:hover {{ transform:translateY(-4px); box-shadow:0 8px 24px #00000050; }}
  .sc-name {{ font-size:.72rem; color:#8892a4; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }}
  .sc-gauge-wrap {{ display:flex; justify-content:center; }}
  .sc-status {{ font-size:.7rem; letter-spacing:2px; text-transform:uppercase; margin-top:6px; font-weight:700; }}
  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(16px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
</style>
</head>
<body>
<div class="grid">
{cards_html}
</div>
<script>
function animateNum(id, target, dur) {{
  const el = document.getElementById(id);
  const t0 = performance.now();
  const ease = t => 1 - Math.pow(1-t,3);
  (function tick(now) {{
    const f = Math.min((now-t0)/dur, 1);
    el.textContent = (target*ease(f)).toFixed(1)+'h';
    if (f<1) requestAnimationFrame(tick);
  }})(t0);
}}

window.addEventListener('load', () => {{
  {js_inits}
}});
</script>
</body>
</html>"""

# ─── Radar chart (Plotly — kept for data accuracy) ────────────────────────────
def radar_fig(vals, color):
    cats = ["Vibration","Température","Pression","RPM","Courant"]
    r_vals = vals + [vals[0]]
    cats_loop = cats + [cats[0]]
    c_rgba = (
        f"rgba(42,157,143,0.2)"  if color==TEAL   else
        f"rgba(230,57,70,0.2)"   if color==RED     else
        f"rgba(244,162,97,0.2)"
    )
    fig = go.Figure(go.Scatterpolar(
        r=r_vals, theta=cats_loop, fill="toself",
        fillcolor=c_rgba, line=dict(color=color, width=2.5),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0,1], gridcolor=BORDER, tickfont=dict(color="#4a5568",size=9)),
            bgcolor=CARD, angularaxis=dict(gridcolor=BORDER,tickfont=dict(color="#e2e8f0"))
        ),
        showlegend=False, height=200, paper_bgcolor=DARK,
        font=dict(color="#f0f2f5"),
        margin=dict(t=16,b=16,l=16,r=16),
        title=dict(text="Profil capteurs (normalisé)", font=dict(size=11,color="#8892a4"))
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
page_key = page.split("  ")[-1]

# ─── PAGE : Prédiction RUL ────────────────────────────────────────────────────
if page_key == "Prédiction RUL":
    st.markdown("""
    <div class="page-header">
        <h1>⚙ Prédiction <span>RUL</span></h1>
        <p>Remaining Useful Life · Estimation temps réel</p>
        <div class="header-chip">INTERFACE OPÉRATIONNELLE</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown('<div class="sec-title">Paramètres Machine</div>', unsafe_allow_html=True)
        with st.expander("⚙️  Contexte opérationnel", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                machine_type = st.selectbox("Type de machine", ["CNC","Pump","Compressor","Robotic Arm"])
                op_mode      = st.selectbox("Mode opératoire", ["idle","normal","peak"])
                cumcount     = st.number_input("Nb mesures (cumcount)", 0, 2000, 500, help="Proxy du vieillissement")
            with c2:
                hours   = st.number_input("Heures depuis maintenance (h)", 0.0, 600.0, 250.0, 10.0)
                ambient = st.number_input("Température ambiante (°C)", 8.0, 18.0, 13.0, 0.5)

        with st.expander("📡  Valeurs capteurs", expanded=True):
            c3, c4 = st.columns(2)
            with c3:
                vib  = st.slider("Vibration RMS (g)",        0.0, 10.0,  1.2, 0.1)
                temp = st.slider("Température moteur (°C)", 28.0, 95.0, 55.0, 1.0)
                pres = st.slider("Pression (bar)",           10.0,200.0, 45.0, 1.0)
            with c4:
                rpm_v = st.slider("RPM",                    100.0,4100.0,900.0,50.0)
                curr  = st.slider("Courant de phase (A)",    2.0, 35.0,  6.0,  0.1)

        predict_btn = st.button("⚡ LANCER LA PRÉDICTION", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="sec-title">Résultat</div>', unsafe_allow_html=True)

        if predict_btn:
            if not pipeline:
                st.error("Modèle non disponible.")
            else:
                X    = prepare_input(vib, temp, curr, pres, rpm_v, hours, ambient, cumcount)
                rul  = float(max(0, pipeline.predict(X)[0]))

                if rul < 10:
                    color    = RED
                    status   = "CRITIQUE"
                    msg      = "INTERVENTION IMMÉDIATE REQUISE"
                    reco     = f"Arrêter la machine dans les **{rul:.0f}h** maximum. Contacter la maintenance d'urgence immédiatement."
                    border_c = "#e6394640"
                    bg_c     = "#2d0a0a"
                elif rul < 30:
                    color    = ORANGE
                    status   = "ALERTE"
                    msg      = "PLANIFIER UNE INTERVENTION SOUS 48H"
                    reco     = f"Planifier un arrêt dans les **48 heures**. Surveiller l'évolution des capteurs vibration et température."
                    border_c = "#f4a26140"
                    bg_c     = "#2d1a00"
                else:
                    color    = TEAL
                    status   = "NORMAL"
                    msg      = "MACHINE EN ÉTAT OPÉRATIONNEL"
                    reco     = f"Machine opérationnelle. Prochaine maintenance dans **{rul:.0f}h**. Surveillance périodique maintenue."
                    border_c = "#2a9d8f40"
                    bg_c     = "#00291a"

                # ── SVG Gauge (HTML/JS) ──
                components.html(make_gauge_html(rul, color), height=240)

                # ── Alert card (HTML) ──
                urgency   = max(0, min(100, 100 - rul))
                urg_color = RED if urgency > 70 else ORANGE if urgency > 50 else TEAL
                st.markdown(f"""
                <div style="background:{bg_c};border:2px solid {border_c};
                    border-radius:14px;padding:18px 20px;margin:6px 0">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                        <span style="font-family:Rajdhani,sans-serif;font-size:1.05rem;
                            font-weight:700;color:{color};letter-spacing:1.5px">{msg}</span>
                        <span style="background:{color}20;border:1px solid {color}60;
                            border-radius:12px;padding:2px 12px;font-size:.7rem;
                            color:{color};font-family:Rajdhani,sans-serif;letter-spacing:2px">{status}</span>
                    </div>
                    <div style="font-size:.82rem;color:#c9d1d9;line-height:1.6">{reco}</div>
                    <div style="margin-top:12px">
                        <div style="display:flex;justify-content:space-between;
                            font-size:.68rem;color:#8892a4;margin-bottom:4px">
                            <span>Niveau d'urgence</span>
                            <span style="color:{urg_color};font-weight:700">{urgency:.0f}%</span>
                        </div>
                        <div style="background:#1e2736;border-radius:6px;height:7px;overflow:hidden">
                            <div style="width:{urgency:.0f}%;height:100%;
                                background:linear-gradient(90deg,{TEAL},{urg_color});
                                border-radius:6px"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Radar (Plotly) ──
                vals = [vib/10,(temp-28)/67,pres/200,rpm_v/4100,curr/35]
                st.plotly_chart(radar_fig(vals, color), use_container_width=True)

                with st.expander("🔬 Features calculées"):
                    st.dataframe(pd.DataFrame({
                        "Feature":["cumcount","hours_sq","vib×temp","vib×hours","temp×hours"],
                        "Valeur" :[cumcount,round(hours**2,1),round(vib*temp,2),
                                   round(vib*hours,2),round(temp*hours,2)],
                        "Rôle"   :["Vieillissement proxy","Usure non-linéaire",
                                   "Stress thermomécanique","Fatigue vibratoire","Fatigue thermique"]
                    }), hide_index=True, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#141820;border:1px dashed #1e2736;border-radius:14px;
                        padding:50px;text-align:center;color:#8892a4;margin-top:8px">
                <div style="font-size:3rem;margin-bottom:14px">⚙️</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:1.2rem;color:#f0f2f5;
                    letter-spacing:1px;margin-bottom:6px">Système prêt</div>
                <div style="font-size:.85rem">Configurez les paramètres et lancez la prédiction</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Multi-scénarios ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Simulation Multi-Scénarios</div>', unsafe_allow_html=True)

    if pipeline:
        sc_defs = [
            ("Machine Neuve",   0.8, 42.0,  5.0, 22.0,  900.0,13.0,  50.0,  80),
            ("Mi-Vie",          1.5, 58.0,  7.0, 35.0, 1100.0,13.0, 300.0, 450),
            ("Dégradée",        4.2, 85.0,  8.5, 55.0, 1200.0,13.0, 450.0, 750),
            ("Critique",        7.5, 92.0, 12.0, 80.0, 1500.0,13.0, 250.0,1000),
        ]
        sc_computed = []
        for name, *params in sc_defs:
            v,te,cu,pr,rpm,am,hr,cc = params
            inp   = prepare_input(v,te,cu,pr,rpm,am,hr,cc)
            rul_s = float(max(0, pipeline.predict(inp)[0]))
            c     = RED if rul_s<10 else ORANGE if rul_s<30 else TEAL
            label = "CRITIQUE" if rul_s<10 else "ALERTE" if rul_s<30 else "NORMAL"
            sc_computed.append((name, rul_s, c, label))

        components.html(make_scenarios_html(sc_computed), height=210)

# ─── PAGE : Performances Modèles ─────────────────────────────────────────────
elif page_key == "Performances Modèles":
    st.markdown("""
    <div class="page-header">
        <h1>📊 Performances des <span>Modèles</span></h1>
        <p>Comparaison rigoureuse · Jeu de test 20% · 4 809 observations</p>
        <div class="header-chip">ÉVALUATION RIGOUREUSE</div>
    </div>
    """, unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("MAE",      "3.077 h",       "-6.7 h vs baseline")
    k2.metric("RMSE",     "5.575 h",       "-9.5 h vs baseline")
    k3.metric("R²",       "0.955",         "+0.284 vs baseline")
    k4.metric("CV MAE",   "3.097±0.073 h", "Très stable")
    k5.metric("Gain FE",  "+42%",          "Feature Engineering")

    results = {
        "Ridge Regression"   :{"MAE":16.288,"RMSE":20.468,"R²":0.394,"CV MAE":16.603},
        "Random Forest ✅"    :{"MAE": 3.077,"RMSE": 5.575,"R²":0.955,"CV MAE": 3.097},
        "XGBoost"            :{"MAE": 4.402,"RMSE": 6.542,"R²":0.938,"CV MAE": 4.430},
        "MLP (Deep Learning)":{"MAE": 4.911,"RMSE": 7.604,"R²":0.916,"CV MAE": 4.943},
    }
    res_df   = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Modèle"})
    pal      = [RED, TEAL, BLUE, ORANGE]

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_r2 = go.Figure()
        for i, row in res_df.iterrows():
            fig_r2.add_trace(go.Bar(x=[row["Modèle"]], y=[row["R²"]],
                name=row["Modèle"], marker_color=pal[i], marker_line_width=0, width=0.5))
        fig_r2.add_hline(y=0.9, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                         annotation_text="Objectif R²=0.90", annotation_font_color="#8892a4")
        fig_r2.update_layout(title="R² par modèle (↑ meilleur)", showlegend=False,
                              yaxis=dict(range=[0,1.05],title="R²"), bargap=0.35)
        dark_fig(fig_r2); st.plotly_chart(fig_r2, use_container_width=True)
    with c2:
        fig_mae = go.Figure()
        for i, row in res_df.iterrows():
            fig_mae.add_trace(go.Bar(x=[row["Modèle"]], y=[row["MAE"]],
                name=row["Modèle"], marker_color=pal[i], marker_line_width=0, width=0.5))
        fig_mae.update_layout(title="MAE par modèle (↓ meilleur)", showlegend=False,
                              yaxis=dict(title="MAE (h)"), bargap=0.35)
        dark_fig(fig_mae); st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown('<div class="sec-title">Classement des Modèles</div>', unsafe_allow_html=True)
    ranked = res_df.sort_values("MAE").reset_index(drop=True)
    medals = ["🥇","🥈","🥉","4️⃣"]
    labels = ["1er","2e","3e","4e"]
    pc = st.columns(4, gap="small")
    for i, row in ranked.iterrows():
        with pc[i]:
            c = pal[res_df.index[res_df["Modèle"]==row["Modèle"]].tolist()[0]]
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#141820,#111520);
                border:1px solid {c}35;border-radius:12px;padding:18px 12px;
                text-align:center;transition:transform .2s" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
                <div style="font-size:1.8rem">{medals[i]}</div>
                <div style="font-size:.7rem;color:{c};letter-spacing:1px;margin:4px 0">{labels[i]}</div>
                <div style="font-size:.82rem;color:#f0f2f5;font-family:Rajdhani,sans-serif;margin-bottom:6px">{row["Modèle"]}</div>
                <div style="font-size:.72rem;color:#8892a4">MAE <span style="color:{c};font-weight:700">{row["MAE"]:.3f}h</span></div>
                <div style="font-size:.72rem;color:#8892a4">R² <span style="color:{c};font-weight:700">{row["R²"]:.3f}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Importance des Variables — Random Forest</div>', unsafe_allow_html=True)
    fi = {"Feature":["cumcount","hours_sq","hours_since_maintenance","rpm","temperature_motor",
                     "machine_type_enc","current_phase_avg","vib_x_temp","pressure_level","vibration_rms"],
          "Importance":[0.400,0.235,0.080,0.038,0.027,0.020,0.017,0.013,0.012,0.010],
          "Type":["Nouvelle","Nouvelle","Originale","Originale","Originale",
                  "Originale","Originale","Nouvelle","Originale","Originale"]}
    fi_df = pd.DataFrame(fi).sort_values("Importance", ascending=True)
    bc = [RED if t=="Nouvelle" else BLUE for t in fi_df["Type"]]
    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"], orientation='h',
        marker_color=bc, marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in fi_df["Importance"]],
        textposition="outside", textfont=dict(color="#8892a4",size=10)
    ))
    fig_fi.update_layout(title="Feature Importance (🔴 nouvelles  🔵 originales)",
                         xaxis=dict(title="Importance",tickformat=".0%"), bargap=0.3)
    dark_fig(fig_fi, h=400); st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="sec-title">Stabilité — CV 5-fold</div>', unsafe_allow_html=True)
    cv_df = pd.DataFrame({"Modèle":["Ridge","Random Forest","XGBoost","MLP"],
                          "CV MAE Mean":[16.603,3.097,4.430,4.943],
                          "CV MAE Std":[0.114,0.073,0.114,0.200]})
    fig_cv = go.Figure()
    for i, row in cv_df.iterrows():
        fig_cv.add_trace(go.Bar(
            x=[row["Modèle"]], y=[row["CV MAE Mean"]],
            name=row["Modèle"], marker_color=pal[i], marker_line_width=0, width=0.5,
            error_y=dict(type="data",array=[row["CV MAE Std"]],color="rgba(255,255,255,0.4)"),
        ))
    fig_cv.update_layout(title="CV MAE ± Std (5-fold) — plus petit = meilleur et plus stable",
                         showlegend=False, bargap=0.3)
    dark_fig(fig_cv); st.plotly_chart(fig_cv, use_container_width=True)

# ─── PAGE : Données Capteurs ──────────────────────────────────────────────────
elif page_key == "Données Capteurs":
    st.markdown("""
    <div class="page-header">
        <h1>📡 Données <span>Capteurs</span></h1>
        <p>Surveillance temps réel · 20 machines industrielles · 14 jours</p>
        <div class="header-chip">MONITORING CONTINU</div>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset non trouvé."); st.stop()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Enregistrements", f"{len(df_raw):,}",      "Dataset complet")
    k2.metric("Machines",        df_raw["machine_id"].nunique(), "4 types")
    k3.metric("RUL moyen",       f"{df_raw['rul_hours'].mean():.1f}h", "")
    k4.metric("Pannes < 24h",    f"{df_raw['failure_within_24h'].sum():,}", "observations critiques")

    st.markdown('<div class="sec-title">Évolution RUL — Machine sélectionnée</div>', unsafe_allow_html=True)
    machine_sel = st.selectbox("Sélectionner une machine", sorted(df_raw["machine_id"].unique()))
    m_df = df_raw[df_raw["machine_id"]==machine_sel].sort_values("timestamp").head(800)

    fig_ts = go.Figure()
    fig_ts.add_hrect(y0=0, y1=10, fillcolor=RED, opacity=0.07, line_width=0)
    fig_ts.add_hrect(y0=10, y1=30, fillcolor=ORANGE, opacity=0.05, line_width=0)
    fig_ts.add_trace(go.Scatter(x=m_df["timestamp"], y=m_df["rul_hours"],
        mode="lines", name="RUL", line=dict(color=TEAL,width=2.5),
        fill="tozeroy", fillcolor="rgba(42,157,143,0.07)"))
    fig_ts.add_hline(y=10, line_dash="dash", line_color=RED,    line_width=1.5,
                     annotation_text="Seuil critique 10h", annotation_font_color=RED)
    fig_ts.add_hline(y=30, line_dash="dash", line_color=ORANGE, line_width=1.5,
                     annotation_text="Seuil alerte 30h",   annotation_font_color=ORANGE)
    fig_ts.update_layout(title=f"RUL — Machine #{machine_sel}",
                         xaxis_title="Temps", yaxis_title="RUL (h)", showlegend=False)
    dark_fig(fig_ts, h=320); st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="sec-title">Capteurs en temps réel</div>', unsafe_allow_html=True)
    slbls = {"vibration_rms":"Vibration RMS (g)","temperature_motor":"Temp. moteur (°C)",
             "rpm":"RPM","pressure_level":"Pression (bar)","current_phase_avg":"Courant (A)"}
    sensor_sel = st.multiselect("Capteurs à afficher", list(slbls.keys()),
                                default=["vibration_rms","temperature_motor"],
                                format_func=lambda x: slbls[x])
    if sensor_sel:
        fig_s = make_subplots(rows=len(sensor_sel), cols=1,
                              subplot_titles=[slbls[s] for s in sensor_sel],
                              shared_xaxes=True, vertical_spacing=0.07)
        pal_s = [TEAL,ORANGE,RED,BLUE,PURPLE]
        for i, col_s in enumerate(sensor_sel):
            fig_s.add_trace(go.Scatter(x=m_df["timestamp"], y=m_df[col_s],
                mode="lines", line=dict(color=pal_s[i%len(pal_s)],width=1.8),
                fill="tozeroy", fillcolor=f"rgba(255,255,255,0.02)"),
                row=i+1, col=1)
        fig_s.update_layout(height=100+190*len(sensor_sel), paper_bgcolor=DARK, plot_bgcolor=CARD,
            font=dict(color="#f0f2f5"), showlegend=False, margin=dict(t=30,b=20,l=20,r=20))
        for i in range(1, len(sensor_sel)+1):
            fig_s.update_xaxes(gridcolor=BORDER, row=i, col=1)
            fig_s.update_yaxes(gridcolor=BORDER, row=i, col=1)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="sec-title">Comparaison par type de machine</div>', unsafe_allow_html=True)
    cap = st.selectbox("Capteur", list(slbls.keys()), format_func=lambda x: slbls[x])
    fig_box = px.box(df_raw, x="machine_type", y=cap, color="machine_type",
                     color_discrete_sequence=[TEAL,ORANGE,RED,BLUE], notched=True,
                     title=f"{slbls[cap]} — Distribution par type de machine")
    dark_fig(fig_box); st.plotly_chart(fig_box, use_container_width=True)
