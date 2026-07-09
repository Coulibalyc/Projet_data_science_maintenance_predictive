"""
Dashboard Technique — EDA & Preprocessing
Pour les Data Scientists / équipes techniques

Lancer : streamlit run dashboard_technique.py --server.port 8502
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Dashboard Technique — EDA & Preprocessing",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'IBM Plex Sans', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }
.stApp { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161b22 0%,#0e1318 100%) !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: #21262d !important; }

/* ── st.metric ── */
div[data-testid="stMetric"] {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 10px !important; padding: 14px !important;
}
div[data-testid="stMetricLabel"] p { color:#8b949e !important; font-size:.75rem !important; text-transform:uppercase; letter-spacing:1px; }
div[data-testid="stMetricValue"]   { color:#58a6ff !important; font-family:'JetBrains Mono' !important; font-size:1.8rem !important; font-weight:700 !important; }
div[data-testid="stMetricDelta"]   { color:#56d364 !important; }

/* ── Expander ── */
details { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 10px !important; }
details summary { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; }

/* ── Page header ── */
.tech-header {
    background: linear-gradient(135deg,#161b22 0%,#0d1117 100%);
    border: 1px solid #21262d; border-radius: 12px;
    padding: 22px 28px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.tech-header::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,transparent,#58a6ff,#56d364,transparent);
}
.tech-header .th { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#e6edf3; margin:0; }
.tech-header .th .ab { color:#58a6ff; }
.tech-header .th .ag { color:#56d364; }
.tech-header .ts { font-size:.8rem; color:#8b949e; font-family:'JetBrains Mono'; margin-top:4px; }
.tech-chip  { display:inline-block; background:#58a6ff14; border:1px solid #58a6ff40; border-radius:6px;
              padding:2px 10px; font-size:.7rem; color:#58a6ff; font-family:'JetBrains Mono'; margin-top:8px; margin-right:6px; }
.tech-chip-g{ display:inline-block; background:#56d36414; border:1px solid #56d36440; border-radius:6px;
              padding:2px 10px; font-size:.7rem; color:#56d364; font-family:'JetBrains Mono'; margin-top:8px; margin-right:6px; }

/* ── Section header ── */
.sec-hdr {
    font-size:1rem; font-weight:600; color:#58a6ff;
    border-bottom:1px solid #21262d; padding-bottom:8px; margin:20px 0 14px 0;
}

/* ── Tags ── */
.tag { display:inline-block; padding:2px 10px; border-radius:6px;
       font-size:.72rem; font-family:'JetBrains Mono'; margin:2px; }
.tg-new  { background:#1f4a1f; color:#56d364; border:1px solid #56d36460; }
.tg-orig { background:#1a2f5a; color:#58a6ff; border:1px solid #58a6ff60; }
.tg-drop { background:#4a1f1f; color:#f85149; border:1px solid #f8514960; }
.tg-tgt  { background:#2a1f4a; color:#d2a8ff; border:1px solid #d2a8ff60; }

/* ── Sidebar ── */
.sb-tech { font-family:'JetBrains Mono'; font-size:1.1rem; font-weight:700; color:#e6edf3; }
.sb-tech .sb { color:#58a6ff; }
.sb-tech .sg { color:#56d364; }
.sb-stat { font-size:.75rem; color:#8b949e; font-family:'JetBrains Mono'; }
.sb-stat span { color:#58a6ff; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-tech"><span class="sb">EDA</span><span class="sg">&amp;</span>ML</div>
    <div style="font-size:.65rem;color:#8b949e;letter-spacing:1.5px;margin-bottom:12px">DASHBOARD TECHNIQUE</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Section", [
        "📋  Analyse du Dataset",
        "❓  Valeurs Manquantes",
        "📊  Distributions",
        "🔗  Corrélations",
        "⚗️  Feature Engineering",
        "🔧  Pipeline Preprocessing",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">Statistiques</div>
    <div class="sb-stat">Obs. &nbsp;<span>24 042</span></div>
    <div class="sb-stat">Variables &nbsp;<span>15</span></div>
    <div class="sb-stat">Machines &nbsp;<span>20</span></div>
    <div class="sb-stat">Jours &nbsp;<span>14</span></div>
    <div class="sb-stat">NaN moyen &nbsp;<span>3.3%</span></div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div style="font-size:.65rem;color:#8b949e">🏆 Random Forest · R²=0.955 · MAE=3.077h</div>""",
                unsafe_allow_html=True)

# ─── Données ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("predictive_maintenance_v3.csv", parse_dates=["timestamp"])
        df = df.sort_values(["machine_id","timestamp"]).reset_index(drop=True)
        df["cumcount"]     = df.groupby("machine_id").cumcount()
        df["hours_sq"]     = df["hours_since_maintenance"]**2
        df["vib_x_temp"]   = df["vibration_rms"]*df["temperature_motor"]
        df["vib_x_hours"]  = df["vibration_rms"]*df["hours_since_maintenance"]
        df["temp_x_hours"] = df["temperature_motor"]*df["hours_since_maintenance"]
        return df
    except:
        return None

df     = load_data()
DARK   = "#0d1117"; CARD = "#161b22"; CARD2 = "#13181f"; BORDER = "#21262d"
BLUE   = "#58a6ff"; GREEN = "#56d364"; RED = "#f85149"; ORANGE = "#d29922"; PURPLE = "#d2a8ff"

def dark_fig(fig, h=350):
    fig.update_layout(height=h, paper_bgcolor=DARK, plot_bgcolor=CARD2,
        font=dict(color="#e6edf3",family="IBM Plex Sans"),
        margin=dict(t=40,b=20,l=20,r=20),
        xaxis=dict(gridcolor=BORDER,zeroline=False),
        yaxis=dict(gridcolor=BORDER,zeroline=False))
    return fig

# ─── Pipeline flow HTML component ─────────────────────────────────────────────
def make_pipeline_html():
    steps = [
        ("01", "Train/Test Split",      "train_test_split()",        "test_size=0.2 · random_state=42",     "#58a6ff"),
        ("02", "SimpleImputer",         "strategy='median'",         "NaN → médiane du TRAIN uniquement",   "#56d364"),
        ("03", "StandardScaler",        "fit_transform(X_train)",    "mean/std appris sur TRAIN seulement", "#56d364"),
        ("04", "ColumnTransformer",     "12 features numériques",    "Orchestration des transformateurs",   "#d2a8ff"),
        ("05", "RandomForestRegressor", "n_estimators=300",          "R²=0.955 · MAE=3.077h",               "#d29922"),
    ]

    cards = ""
    for i, (num, name, code, note, color) in enumerate(steps):
        arrow = '<div class="arrow">→</div>' if i < len(steps)-1 else ""
        cards += f"""
        <div class="pipe-card" style="--col:{color};animation-delay:{i*0.12}s">
            <div class="step-num" style="color:{color};border-color:{color}40">{num}</div>
            <div class="step-name" style="color:{color}">{name}</div>
            <code class="step-code">{code}</code>
            <div class="step-note">{note}</div>
        </div>
        {arrow}
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ height:100%; background:#0d1117; color:#e6edf3; font-family:'IBM Plex Sans',sans-serif; }}
  body {{ display:flex; align-items:center; padding:16px; overflow-x:auto; }}

  .pipeline-wrap {{ display:flex; align-items:center; gap:0; min-width:max-content; margin:auto; }}

  .pipe-card {{
    background: linear-gradient(135deg,#161b22 0%,#13181f 100%);
    border: 1px solid #21262d;
    border-top: 3px solid var(--col);
    border-radius: 12px;
    padding: 16px 14px;
    width: 168px;
    text-align: center;
    animation: slideUp .5s ease both;
    transition: transform .2s, box-shadow .2s;
    flex-shrink: 0;
  }}
  .pipe-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 28px #00000060, 0 0 20px var(--col)18;
  }}

  .step-num {{
    font-family:'JetBrains Mono',monospace; font-size:.8rem; font-weight:700;
    border:1px solid; border-radius:20px; display:inline-block;
    padding:2px 10px; margin-bottom:10px; letter-spacing:1px;
  }}
  .step-name {{
    font-family:'IBM Plex Sans',sans-serif; font-size:.88rem; font-weight:700;
    margin-bottom:8px; line-height:1.25;
  }}
  .step-code {{
    display:block; background:#0d1117; border:1px solid #21262d;
    border-radius:6px; padding:4px 8px; font-size:.72rem;
    color:#c9d1d9; font-family:'JetBrains Mono',monospace;
    margin-bottom:8px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }}
  .step-note {{
    font-size:.7rem; color:#8b949e; line-height:1.4;
  }}

  .arrow {{
    font-size:1.4rem; color:#21262d; padding:0 4px;
    flex-shrink:0; align-self:center;
    animation: fadeIn .4s ease both;
    animation-delay:.6s;
  }}

  .guarantee-row {{
    display:flex; gap:10px; margin-top:14px; width:100%; min-width:max-content;
  }}
  .guar {{
    background:#161b22; border:1px solid; border-radius:10px;
    padding:12px 14px; flex:1; min-width:160px;
  }}
  .guar-icon {{ font-size:1.2rem; margin-bottom:4px; }}
  .guar-title {{ font-size:.8rem; font-weight:700; margin-bottom:3px; }}
  .guar-desc  {{ font-size:.72rem; color:#8b949e; line-height:1.4; }}

  @keyframes slideUp {{
    from {{ opacity:0; transform:translateY(20px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes fadeIn {{
    from {{ opacity:0; }} to {{ opacity:1; }}
  }}
</style>
</head>
<body>
<div>
  <div class="pipeline-wrap">
    {cards}
  </div>
  <div class="guarantee-row" style="animation:slideUp .5s ease .65s both;opacity:0">
    <div class="guar" style="border-color:#56d36440">
      <div class="guar-icon">✅</div>
      <div class="guar-title" style="color:#56d364">Split avant preprocessing</div>
      <div class="guar-desc">Données brutes séparées AVANT toute transformation</div>
    </div>
    <div class="guar" style="border-color:#56d36440">
      <div class="guar-icon">✅</div>
      <div class="guar-title" style="color:#56d364">Pipeline sklearn</div>
      <div class="guar-desc">Imputer/Scaler ajustés sur X_train uniquement via fit_transform</div>
    </div>
    <div class="guar" style="border-color:#f8514940">
      <div class="guar-icon">🚫</div>
      <div class="guar-title" style="color:#f85149">Variables leakage exclues</div>
      <div class="guar-desc">failure_within_24h · failure_type · estimated_repair_cost</div>
    </div>
    <div class="guar" style="border-color:#d2a8ff40">
      <div class="guar-icon">🔁</div>
      <div class="guar-title" style="color:#d2a8ff">CV 5-fold</div>
      <div class="guar-desc">Preprocessing refit à chaque fold — zéro leakage inter-fold</div>
    </div>
  </div>
</div>
</body>
</html>"""

# ─── Feature bars HTML component ─────────────────────────────────────────────
def make_feat_bars_html(features_info):
    """
    features_info: list of (name, importance_pct, tag, code, desc, justif)
    """
    max_imp = max(f[1] for f in features_info)
    rows = ""
    for i, (name, imp, tag, code, desc, justif) in enumerate(features_info):
        is_new = "🔴" in tag
        color  = "#f85149" if is_new else "#58a6ff"
        pct_w  = (imp / max_imp) * 100
        tag_cls = "new" if is_new else "orig"
        rows += f"""
        <div class="feat-row" style="animation-delay:{i*0.1}s">
          <div class="feat-header">
            <div class="feat-name">
              <code>{name}</code>
              <span class="ftag ftag-{tag_cls}">{tag}</span>
            </div>
            <div class="feat-pct" style="color:{color}">{imp:.1f}%</div>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="--w:{pct_w:.1f}%;--col:{color}"></div>
          </div>
          <div class="feat-detail">
            <div class="feat-code"><code>{code}</code></div>
            <div class="feat-just">{justif}</div>
          </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ background:#0d1117; color:#e6edf3; font-family:'IBM Plex Sans',sans-serif; }}
  body {{ padding:16px; }}

  .feat-row {{
    margin-bottom:18px;
    animation: fadeUp .45s ease both;
  }}
  .feat-header {{
    display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;
  }}
  .feat-name {{ display:flex; align-items:center; gap:8px; }}
  .feat-name code {{
    font-family:'JetBrains Mono',monospace; font-size:.85rem;
    color:#e6edf3; background:#161b22; padding:2px 8px;
    border-radius:5px; border:1px solid #21262d;
  }}
  .feat-pct {{ font-family:'JetBrains Mono',monospace; font-size:.88rem; font-weight:700; }}

  .ftag {{ display:inline-block; padding:1px 8px; border-radius:5px;
           font-size:.68rem; font-family:'JetBrains Mono'; }}
  .ftag-new  {{ background:#1f4a1f; color:#56d364; border:1px solid #56d36460; }}
  .ftag-orig {{ background:#1a2f5a; color:#58a6ff; border:1px solid #58a6ff60; }}

  .bar-bg {{
    background:#161b22; border-radius:6px; height:10px; overflow:hidden;
    border:1px solid #21262d;
  }}
  .bar-fill {{
    height:100%; border-radius:6px;
    background: linear-gradient(90deg, var(--col)60, var(--col));
    width: 0%;
    animation: grow .9s ease both var(--delay, 0s);
    box-shadow: 0 0 8px var(--col)50;
  }}

  .feat-detail {{
    display:flex; gap:12px; margin-top:6px;
    font-size:.75rem; color:#8b949e;
  }}
  .feat-code {{ flex-shrink:0; }}
  .feat-code code {{
    font-family:'JetBrains Mono'; font-size:.7rem; color:#58a6ff;
    background:#0d1117; padding:2px 7px; border-radius:4px; border:1px solid #21262d;
  }}
  .feat-just {{ line-height:1.5; }}

  .legend {{
    display:flex; gap:14px; margin-bottom:14px; font-size:.75rem;
  }}
  .leg-item {{ display:flex; align-items:center; gap:5px; }}
  .leg-dot  {{ width:10px; height:10px; border-radius:50%; }}

  @keyframes grow {{
    from {{ width:0%; }} to {{ width:var(--w); }}
  }}
  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(12px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
</style>
</head>
<body>
<div class="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#f85149"></div>Nouvelle feature</div>
  <div class="leg-item"><div class="leg-dot" style="background:#58a6ff"></div>Feature originale</div>
</div>
{rows}
</body>
</html>"""

page_key = page.split("  ")[-1]

# ══════════════════════════════════════════════════════════════════════════════
if page_key == "Analyse du Dataset":
    st.markdown("""
    <div class="tech-header">
        <div class="th">📋 Analyse du <span class="ab">Dataset</span></div>
        <div class="ts">Vue d'ensemble · Dataset industriel de maintenance prédictive</div>
        <span class="tech-chip">24 042 observations</span>
        <span class="tech-chip">15 variables</span>
        <span class="tech-chip-g">20 machines</span>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset non trouvé. Placer `predictive_maintenance_v3.csv` dans le dossier."); st.stop()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Enregistrements", f"{len(df):,}")
    c2.metric("Variables", "15")
    c3.metric("Machines",  df["machine_id"].nunique())
    c4.metric("Jours",     "14")
    c5.metric("NaN moyen", "3.3%")

    st.markdown('<div class="sec-hdr">📂 Variables du dataset</div>', unsafe_allow_html=True)
    vars_info = {
        "timestamp"             :("datetime","drop",  "Horodatage — non généralisable"),
        "machine_id"            :("int64",   "drop",  "ID machine — non généralisable"),
        "machine_type"          :("object",  "keep",  "Type de machine — encodé"),
        "vibration_rms"         :("float64", "keep",  "Vibration RMS (g) — capteur clé"),
        "temperature_motor"     :("float64", "keep",  "Température moteur (°C)"),
        "current_phase_avg"     :("float64", "keep",  "Courant de phase moyen (A)"),
        "pressure_level"        :("float64", "keep",  "Pression (bar)"),
        "rpm"                   :("float64", "keep",  "Vitesse de rotation (RPM)"),
        "operating_mode"        :("object",  "keep",  "Mode opératoire — encodé"),
        "hours_since_maintenance":("float64","keep",  "Heures depuis maintenance — très important"),
        "ambient_temp"          :("float64", "keep",  "Température ambiante (°C)"),
        "rul_hours"             :("float64", "target","🎯 VARIABLE CIBLE"),
        "failure_within_24h"    :("int64",   "drop",  "⚠️ Data leakage — dérivé du RUL"),
        "failure_type"          :("object",  "drop",  "⚠️ Data leakage — connu après panne"),
        "estimated_repair_cost" :("int64",   "drop",  "⚠️ Data leakage — corrélé à la panne"),
    }
    rows = [{"Variable":k,"Type":v[0],
             "Statut":{"drop":"🔴 Supprimée","keep":"🟢 Gardée","target":"🎯 Cible"}[v[1]],
             "Description":v[2]} for k,v in vars_info.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1,2], gap="large")
    with col1:
        counts = {"Gardées":8,"Supprimées (leakage)":3,"Supprimées (id)":2,"Cible":1,"Nouvelles FE":5}
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.keys()), values=list(counts.values()),
            marker_colors=[GREEN,RED,ORANGE,PURPLE,BLUE], hole=0.55,
            textfont=dict(size=10,family="JetBrains Mono")
        ))
        fig_pie.update_layout(title=dict(text="Répartition des variables",font=dict(size=11,color="#8b949e")),
            height=280, paper_bgcolor=DARK, font=dict(color="#e6edf3"),
            legend=dict(font=dict(size=9)), margin=dict(t=40,b=10,l=0,r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown('<div class="sec-hdr">📊 Statistiques descriptives</div>', unsafe_allow_html=True)
        num_cols = ["vibration_rms","temperature_motor","current_phase_avg",
                    "pressure_level","rpm","hours_since_maintenance","ambient_temp","rul_hours"]
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Valeurs Manquantes":
    st.markdown("""
    <div class="tech-header">
        <div class="th">❓ Valeurs <span class="ab">Manquantes</span></div>
        <div class="ts">Analyse · Stratégie d'imputation · Anti-leakage</div>
        <span class="tech-chip">NaN moyen: 3.3%</span>
        <span class="tech-chip-g">Stratégie: Médiane</span>
    </div>
    """, unsafe_allow_html=True)

    if df is None: st.stop()
    missing     = df.isnull().sum()
    missing_pct = (missing/len(df)*100).round(2)
    miss_df     = pd.DataFrame({"Variable":missing.index,"Count":missing.values,"Pct (%)":missing_pct.values})
    miss_df     = miss_df[miss_df["Count"]>0].sort_values("Pct (%)", ascending=False)

    c1, c2 = st.columns([3,2], gap="large")
    with c1:
        st.markdown('<div class="sec-hdr">📈 Taux de valeurs manquantes</div>', unsafe_allow_html=True)
        fig = px.bar(miss_df, x="Variable", y="Pct (%)", color="Pct (%)",
                     color_continuous_scale=[[0,"#1f4a1f"],[0.5,"#d29922"],[1,"#f85149"]],
                     text="Pct (%)")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside",
                          marker_line_width=0, width=0.5)
        fig.add_hline(y=5, line_dash="dash", line_color=RED,
                      annotation_text="Seuil critique 5%", annotation_font_color=RED)
        fig.update_layout(coloraxis_showscale=False, bargap=0.35, xaxis_title="", yaxis_title="NaN (%)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="sec-hdr">📋 Détail</div>', unsafe_allow_html=True)
        st.dataframe(miss_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        st.success("**Stratégie : Médiane** via `SimpleImputer(strategy='median')`\n\nRobuste aux outliers capteurs. Ajustée uniquement sur X_train.")

    st.markdown('<div class="sec-hdr">⚡ Justification méthodologique</div>', unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    col1.info("**Médiane vs Moyenne**\n\nLa médiane résiste aux pics de capteurs (ex: vibration 10g). La moyenne serait biaisée.")
    col2.success("**Pipeline sklearn**\n\nL'imputation est `fit_transform` sur X_train. `transform` appliqué à X_test. Zéro contamination.")
    col3.warning("**Variables catégorielles**\n\n`operating_mode` et `machine_type` : aucun NaN → encodage direct sans imputation.")

# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Distributions":
    st.markdown("""
    <div class="tech-header">
        <div class="th">📊 Distributions des <span class="ab">Variables</span></div>
        <div class="ts">Analyse statistique · Variable cible · Capteurs par type</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None: st.stop()

    st.markdown('<div class="sec-hdr">🎯 Variable cible : rul_hours</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="rul_hours", nbins=60, color_discrete_sequence=[BLUE])
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.add_vline(x=df["rul_hours"].median(), line_dash="dash", line_color=GREEN, line_width=2,
                      annotation_text=f"Médiane: {df['rul_hours'].median():.1f}h", annotation_font_color=GREEN)
        fig.add_vline(x=df["rul_hours"].mean(), line_dash="dash", line_color=ORANGE, line_width=2,
                      annotation_text=f"Moyenne: {df['rul_hours'].mean():.1f}h", annotation_font_color=ORANGE)
        fig.update_layout(title="Distribution de rul_hours")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(df, x="machine_type", y="rul_hours", color="machine_type",
                      title="RUL par type de machine", notched=True,
                      color_discrete_sequence=[BLUE,GREEN,ORANGE,RED])
        dark_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

    sc = st.columns(4)
    sc[0].metric("Médiane RUL", f"{df['rul_hours'].median():.1f}h")
    sc[1].metric("Moyenne RUL", f"{df['rul_hours'].mean():.1f}h")
    sc[2].metric("Std RUL",     f"{df['rul_hours'].std():.1f}h")
    sc[3].metric("Max RUL",     f"{df['rul_hours'].max():.0f}h")

    st.markdown('<div class="sec-hdr">🎻 Capteurs par type de machine</div>', unsafe_allow_html=True)
    slbls = {"vibration_rms":"Vibration RMS (g)","temperature_motor":"Temp. moteur (°C)",
             "rpm":"RPM","pressure_level":"Pression (bar)"}
    sensor_sel = st.selectbox("Capteur", list(slbls.keys()), format_func=lambda x: slbls[x])
    fig3 = px.violin(df, x="machine_type", y=sensor_sel, color="machine_type", box=True, points=False,
                     title=f"{slbls[sensor_sel]} par type de machine",
                     color_discrete_sequence=[BLUE,GREEN,ORANGE,RED])
    dark_fig(fig3, h=420); st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sec-hdr">⚙️ RUL par mode opératoire</div>', unsafe_allow_html=True)
    fig4 = px.box(df, x="operating_mode", y="rul_hours", color="operating_mode", notched=True,
                  title="RUL médian selon le mode opératoire",
                  color_discrete_sequence=[GREEN,ORANGE,RED])
    dark_fig(fig4); st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Corrélations":
    st.markdown("""
    <div class="tech-header">
        <div class="th">🔗 Analyse des <span class="ab">Corrélations</span></div>
        <div class="ts">Pearson · Relations non-linéaires · Multicolinéarité</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None: st.stop()
    num_cols = ["vibration_rms","temperature_motor","current_phase_avg",
                "pressure_level","rpm","hours_since_maintenance","ambient_temp","rul_hours"]
    corr = df[num_cols].corr().round(2)

    c1, c2 = st.columns([3,2], gap="large")
    with c1:
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Matrice de corrélation de Pearson")
        fig.update_traces(textfont=dict(size=10, family="JetBrains Mono"))
        dark_fig(fig, h=480); st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="sec-hdr">🎯 Corrélation avec RUL</div>', unsafe_allow_html=True)
        corr_rul  = corr["rul_hours"].drop("rul_hours").sort_values(key=abs, ascending=True)
        bar_color = [RED if v<0 else GREEN for v in corr_rul.values]
        fig2 = go.Figure(go.Bar(x=corr_rul.values, y=corr_rul.index,
            orientation='h', marker_color=bar_color, marker_line_width=0,
            text=[f"{v:.3f}" for v in corr_rul.values], textposition="outside",
            textfont=dict(size=10,family="JetBrains Mono",color="#8b949e")))
        fig2.add_vline(x=0, line_color=BORDER, line_width=1)
        fig2.update_layout(title="Corrélation avec rul_hours",
                           xaxis_title="Corrélation de Pearson", bargap=0.35)
        dark_fig(fig2, h=320); st.plotly_chart(fig2, use_container_width=True)
        st.info("`hours_since_maintenance` : -0.307 (plus forte)\n\nCapteurs physiques : faibles (-0.08 à 0.02) → justifie RF/XGBoost\n\nMulticolinéarité capteurs : 0.74–0.88")

    st.markdown('<div class="sec-hdr">🔬 Relation capteur → RUL</div>', unsafe_allow_html=True)
    sel_opts = {"hours_since_maintenance":"Heures depuis maintenance","vibration_rms":"Vibration RMS",
                "temperature_motor":"Température moteur","rpm":"RPM"}
    sel = st.selectbox("Capteur", list(sel_opts.keys()), format_func=lambda x: sel_opts[x])
    sample = df.sample(3000, random_state=42)
    fig3 = px.scatter(sample, x=sel, y="rul_hours", color="machine_type",
                      opacity=0.45, trendline="lowess",
                      title=f"{sel_opts[sel]} vs rul_hours (tendance LOWESS)",
                      color_discrete_sequence=[BLUE,GREEN,ORANGE,RED])
    dark_fig(fig3, h=400); st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Feature Engineering":
    st.markdown("""
    <div class="tech-header">
        <div class="th">⚗️ Feature <span class="ag">Engineering</span></div>
        <div class="ts">5 nouvelles features · R² 0.671 → 0.955 (+42%)</div>
        <span class="tech-chip-g">+42% gain</span>
        <span class="tech-chip">R² final: 0.955</span>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("R² Baseline", "0.671")
    c2.metric("R² Après FE", "0.955", "+0.284")
    c3.metric("Gain relatif", "+42%", "Feature Engineering")

    st.markdown('<div class="sec-hdr">📊 Importance & détail des 5 nouvelles features</div>', unsafe_allow_html=True)

    features_info = [
        ("cumcount",    40.0, "🔴 Clé",
         "df.groupby('machine_id').cumcount()",
         "Nombre de mesures depuis le début du cycle.",
         "Proxy direct du vieillissement — disponible temps réel, zéro leakage."),
        ("hours_sq",    23.5, "🔴 Importante",
         "df['hours_since_maintenance'] ** 2",
         "Carré des heures depuis la dernière maintenance.",
         "Capture la relation non-linéaire : l'usure accélère exponentiellement."),
        ("vib_x_temp",  1.2, "🔵 Complémentaire",
         "df['vibration_rms'] * df['temperature_motor']",
         "Produit vibration × température moteur.",
         "Stress thermomécanique combiné — plus destructeur que chaque capteur seul."),
        ("vib_x_hours", 1.0, "🔵 Complémentaire",
         "df['vibration_rms'] * df['hours_since_maintenance']",
         "Produit vibration × heures depuis maintenance.",
         "Vibration cumulée dans le temps — fatigue vibratoire progressive."),
        ("temp_x_hours",1.0, "🔵 Complémentaire",
         "df['temperature_motor'] * df['hours_since_maintenance']",
         "Produit température × heures depuis maintenance.",
         "Chaleur cumulée dans le temps — fatigue thermique progressive."),
    ]

    # Animated HTML bars
    components.html(make_feat_bars_html(features_info), height=370, scrolling=False)

    st.markdown('<div class="sec-hdr">🔍 Détails et code</div>', unsafe_allow_html=True)
    for name, imp, tag, code, desc, justif in features_info:
        with st.expander(f"`{name}` — {imp:.1f}% d'importance — {tag}"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown(f"**Description :** {desc}")
                st.code(f"df['{name}'] = {code}", language="python")
            with col2:
                st.markdown(f"**Justification métier :** {justif}")

    if df is not None:
        st.markdown('<div class="sec-hdr">📈 Visualisation : cumcount vs RUL</div>', unsafe_allow_html=True)
        selected = st.selectbox("Machine", sorted(df["machine_id"].unique())[:5])
        m_df = df[df["machine_id"]==selected].copy()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["cumcount vs RUL","hours_since_maintenance vs RUL"])
        fig.add_trace(go.Scatter(x=m_df["cumcount"], y=m_df["rul_hours"],
            mode="markers", marker=dict(color=RED,size=4,opacity=0.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=m_df["hours_since_maintenance"], y=m_df["rul_hours"],
            mode="markers", marker=dict(color=BLUE,size=4,opacity=0.6)), row=1, col=2)
        fig.update_layout(height=360, paper_bgcolor=DARK, plot_bgcolor=CARD2,
                          font=dict(color="#e6edf3"), showlegend=False,
                          margin=dict(t=40,b=20,l=20,r=20))
        for i in [1,2]:
            fig.update_xaxes(gridcolor=BORDER, row=1, col=i)
            fig.update_yaxes(gridcolor=BORDER, row=1, col=i)
        st.plotly_chart(fig, use_container_width=True)
        cc = m_df["cumcount"].corr(m_df["rul_hours"])
        hm = m_df["hours_since_maintenance"].corr(m_df["rul_hours"])
        c1, c2 = st.columns(2)
        c1.metric("Corrélation cumcount ↔ RUL",              f"{cc:.3f}")
        c2.metric("Corrélation hours_since_maintenance ↔ RUL",f"{hm:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Pipeline Preprocessing":
    st.markdown("""
    <div class="tech-header">
        <div class="th">🔧 Pipeline <span class="ab">Preprocessing</span></div>
        <div class="ts">Architecture sklearn · Garantie anti-leakage · Reproductibilité</div>
        <span class="tech-chip">sklearn Pipeline</span>
        <span class="tech-chip-g">Zero Leakage</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🔄 Architecture visuelle du pipeline</div>', unsafe_allow_html=True)
    components.html(make_pipeline_html(), height=330, scrolling=True)

    st.markdown('<div class="sec-hdr">📋 Code complet</div>', unsafe_allow_html=True)
    st.code("""
# 1. Split AVANT tout preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ↑ Le test set ne voit JAMAIS les données d'entraînement

# 2. Transformer numérique : imputation + normalisation
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # NaN → médiane du TRAIN
    ('scaler',  StandardScaler())                    # z-score sur le TRAIN
])

# 3. ColumnTransformer (12 features numériques)
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
])

# 4. Pipeline complet
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=300, random_state=42))
])

# 5. Fit UNIQUEMENT sur X_train
pipeline.fit(X_train, y_train)
# → SimpleImputer et StandardScaler apprennent sur X_train seulement

# 6. Prédiction sur X_test (stats du train appliquées)
y_pred = pipeline.predict(X_test)
# → Métriques fiables, zéro contamination
    """, language="python")

    st.markdown('<div class="sec-hdr">🚨 Variables exclues — Data Leakage</div>', unsafe_allow_html=True)
    leakage_vars = [
        ("timestamp",             "Identifiant temporel",  "Non généralisable à de nouvelles machines"),
        ("machine_id",            "Identifiant machine",   "Apprentissage machine-spécifique → non généralisable"),
        ("failure_within_24h",    "🚨 Data Leakage",       "Dérivé directement de rul_hours : si RUL < 24 → 1"),
        ("failure_type",          "🚨 Data Leakage",       "Connu seulement APRÈS la panne — impossible en production"),
        ("estimated_repair_cost", "🚨 Data Leakage",       "Corrélé à la panne — connu seulement après l'événement"),
    ]
    for var, reason, expl in leakage_vars:
        is_leak = "🚨" in reason
        bg   = "#2d0d0d" if is_leak else "#161b22"
        bc   = "#f8514940" if is_leak else "#21262d"
        tc   = "#f85149"  if is_leak else "#58a6ff"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {bc};border-radius:10px;padding:12px 16px;margin:4px 0">
            <div style="color:{tc};font-family:JetBrains Mono,monospace;font-size:.85rem;font-weight:600">
                <code style="background:#0d1117;padding:2px 6px;border-radius:4px;border:1px solid #21262d;color:{tc}">{var}</code>
                &nbsp;—&nbsp;{reason}
            </div>
            <div style="color:#8b949e;font-size:.8rem;margin-top:5px">{expl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">📊 Résultats CV 5-fold</div>', unsafe_allow_html=True)
    cv_data = pd.DataFrame({
        "Modèle":      ["Ridge Regression","Random Forest","XGBoost","MLP"],
        "CV MAE Mean": [16.603,3.097,4.430,4.943],
        "CV MAE Std":  [0.114, 0.073,0.114,0.200],
        "Stabilité":   ["Faible","🏆 Très élevée","Élevée","Moyenne"],
    })
    fig_cv = go.Figure()
    pal = [RED, "#2a9d8f", BLUE, ORANGE]
    for i, row in cv_data.iterrows():
        fig_cv.add_trace(go.Bar(
            x=[row["Modèle"]], y=[row["CV MAE Mean"]], name=row["Modèle"],
            marker_color=pal[i], marker_line_width=0, width=0.5,
            error_y=dict(type="data",array=[row["CV MAE Std"]],color="rgba(255,255,255,0.4)"),
        ))
    fig_cv.update_layout(title="CV MAE ± Std (5-fold)", showlegend=False, bargap=0.3)
    dark_fig(fig_cv); st.plotly_chart(fig_cv, use_container_width=True)
    st.dataframe(cv_data, use_container_width=True, hide_index=True)
    st.info("**KFold(n_splits=5, shuffle=True)** — Preprocessing refit à chaque fold pour éliminer tout leakage inter-fold.")
