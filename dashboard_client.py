"""
Dashboard Client — Maintenance Prédictive Industrielle
Interface décisionnelle pour les responsables maintenance

Lancer : streamlit run dashboard_client.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os

st.set_page_config(page_title="Maintenance Prédictive — RUL",
                   page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
* { font-family: 'Source Sans 3', sans-serif; }
h1,h2,h3 { font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; }
.stApp { background: #111318; color: #f0f2f5; }
section[data-testid="stSidebar"] { background: #1a1d24 !important; border-right: 2px solid #2a9d8f33; }

.alert-critical {
    background: linear-gradient(135deg, #2d0a0a, #1a0606);
    border: 2px solid #e63946; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    box-shadow: 0 0 30px #e6394640;
}
.alert-critical .rul-val { font-size: 4rem; font-weight: 700; color: #e63946;
    font-family: 'Rajdhani'; line-height: 1; }
.alert-critical .rul-label { font-size: 0.85rem; color: #e6394699;
    text-transform: uppercase; letter-spacing: 2px; }
.alert-critical .alert-text { font-size: 1rem; color: #e63946; margin-top: 8px; font-weight: 600; }

.alert-warning {
    background: linear-gradient(135deg, #2d1a00, #1a1000);
    border: 2px solid #f4a261; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    box-shadow: 0 0 30px #f4a26140;
}
.alert-warning .rul-val { font-size: 4rem; font-weight: 700; color: #f4a261;
    font-family: 'Rajdhani'; line-height: 1; }
.alert-warning .rul-label { font-size: 0.85rem; color: #f4a26199;
    text-transform: uppercase; letter-spacing: 2px; }
.alert-warning .alert-text { font-size: 1rem; color: #f4a261; margin-top: 8px; font-weight: 600; }

.alert-ok {
    background: linear-gradient(135deg, #002d1a, #001a10);
    border: 2px solid #2a9d8f; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    box-shadow: 0 0 30px #2a9d8f40;
}
.alert-ok .rul-val { font-size: 4rem; font-weight: 700; color: #2a9d8f;
    font-family: 'Rajdhani'; line-height: 1; }
.alert-ok .rul-label { font-size: 0.85rem; color: #2a9d8f99;
    text-transform: uppercase; letter-spacing: 2px; }
.alert-ok .alert-text { font-size: 1rem; color: #2a9d8f; margin-top: 8px; font-weight: 600; }

.kpi-card { background: #1a1d24; border: 1px solid #2a2d35; border-radius: 10px;
            padding: 16px; text-align: center; }
.kpi-val { font-size: 2rem; font-weight: 700; font-family: 'Rajdhani'; }
.kpi-lbl { font-size: 0.7rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
.section-title { font-family: 'Rajdhani'; font-size: 1.3rem; font-weight: 600;
    color: #2a9d8f; border-left: 3px solid #2a9d8f;
    padding-left: 10px; margin: 20px 0 12px 0; }
div[data-testid="stMetric"] { background: #1a1d24; border: 1px solid #2a2d35;
    border-radius: 10px; padding: 12px; }

/* ── Sidebar : texte visible ── */

/* Titre et sous-titre sidebar */
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Labels de navigation (radio buttons) */
section[data-testid="stSidebar"] .stRadio label {
    color: #e2e8f0 !important;
}

/* Texte "Modèle actif", "R²", "MAE" */
section[data-testid="stSidebar"] p {
    color: #cbd5e1 !important;
}

/* Valeurs en gras (Random Forest, 0.955, 3.077h) */
section[data-testid="stSidebar"] strong {
    color: #2a9d8f !important;
}

/* Titres sidebar */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f0f2f5 !important;
}

/* Séparateurs --- */
section[data-testid="stSidebar"] hr {
    border-color: #2a2d35 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Maintenance Prédictive")
    st.markdown("**Interface Opérationnelle**")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Prédiction RUL",
        "Performances Modèles",
        "Données Capteurs",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Modèle actif**")
    st.markdown("Random Forest")
    st.markdown("R² = **0.955**")
    st.markdown("MAE = **3.077h**")

# ─── Chargement modèle & données ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "models/best_model_pipeline.pkl"
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("predictive_maintenance_v3.csv", parse_dates=["timestamp"])
        df = df.sort_values(["machine_id","timestamp"]).reset_index(drop=True)
        return df
    except: return None

pipeline = load_model()
df_raw   = load_data()

TEAL = "#2a9d8f"; RED = "#e63946"; ORANGE = "#f4a261"
DARK = "#111318"; CARD = "#1a1d24"; BORDER = "#2a2d35"

def dark_fig(fig, h=350):
    fig.update_layout(height=h, paper_bgcolor=DARK, plot_bgcolor=CARD,
        font=dict(color="#f0f2f5", family="Source Sans 3"),
        margin=dict(t=40,b=20,l=10,r=10),
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
    return fig

def prepare_input(vib, temp, curr, pres, rpm_v, hours, ambient, cumcount):
    return pd.DataFrame([{
        "vibration_rms"          : vib,
        "temperature_motor"      : temp,
        "current_phase_avg"      : curr,
        "pressure_level"         : pres,
        "rpm"                    : rpm_v,
        "hours_since_maintenance": hours,
        "ambient_temp"           : ambient,
        "cumcount"               : cumcount,
        "hours_sq"               : hours ** 2,
        "vib_x_temp"             : vib * temp,
        "vib_x_hours"            : vib * hours,
        "temp_x_hours"           : temp * hours,
    }])

# ══════════════════════════════════════════════════════════════════════════════
if page == "Prédiction RUL":

    st.title("Prédiction — Durée de Vie Restante")

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown('<div class="section-title">Paramètres Machine</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            machine_type  = st.selectbox("Type de machine", ["CNC","Pump","Compressor","Robotic Arm"])
            op_mode       = st.selectbox("Mode opératoire", ["idle","normal","peak"])
            cumcount      = st.number_input("Nb mesures depuis début du cycle", 0, 2000, 500,
                                            help="cumcount : proxy du vieillissement")
        with c2:
            hours         = st.number_input("Heures depuis maintenance", 0.0, 600.0, 250.0, 10.0)
            ambient       = st.number_input("Température ambiante (°C)", 8.0, 18.0, 13.0, 0.5)

        st.markdown("**Valeurs capteurs**")
        c3, c4 = st.columns(2)
        with c3:
            vib  = st.slider("Vibration RMS (g)",   0.0, 10.0, 1.2, 0.1)
            temp = st.slider("Température moteur (°C)", 28.0, 95.0, 55.0, 1.0)
            pres = st.slider("Pression (bar)",       10.0, 200.0, 45.0, 1.0)
        with c4:
            rpm_v = st.slider("RPM",                 100.0, 4100.0, 900.0, 50.0)
            curr  = st.slider("Courant de phase (A)", 2.0, 35.0, 6.0, 0.1)

        predict_btn = st.button("⚡ Lancer la Prédiction", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title">Résultat</div>', unsafe_allow_html=True)

        if predict_btn:
            if not pipeline:
                st.error("Modèle non disponible. Exécuter le notebook d'abord.")
            else:
                input_df = prepare_input(vib, temp, curr, pres, rpm_v, hours, ambient, cumcount)
                rul = float(max(0, pipeline.predict(input_df)[0]))

                # Affichage alerte
                if rul < 10:
                    cls = "critical"
                    msg = "INTERVENTION IMMÉDIATE REQUISE"
                elif rul < 30:
                    cls = "warning"
                    msg = "PLANIFIER UNE INTERVENTION SOUS 48H"
                else:
                    cls = "ok"
                    msg = "MACHINE EN ÉTAT NORMAL"

                st.markdown(f"""
                <div class="alert-{cls}">
                    <div class="rul-label">RUL estimé</div>
                    <div class="rul-val">{rul:.1f}h</div>
                    <div class="alert-text">{msg}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # Features calculées
                with st.expander("Features calculées par le modèle"):
                    fe_data = {
                        "Feature": ["cumcount", "hours_sq", "vib_x_temp", "vib_x_hours", "temp_x_hours"],
                        "Valeur": [cumcount, round(hours**2,1), round(vib*temp,2),
                                   round(vib*hours,2), round(temp*hours,2)],
                        "Description": ["Nb mesures", "Heures²", "Vib×Temp", "Vib×Heures", "Temp×Heures"]
                    }
                    st.dataframe(pd.DataFrame(fe_data), hide_index=True, use_container_width=True)

                # Radar capteurs
                cats = ["Vibration", "Température", "Pression", "RPM", "Courant"]
                vals = [vib/10, (temp-28)/67, pres/200, rpm_v/4100, curr/35]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals+[vals[0]], theta=cats+[cats[0]],
                    fill="toself",
                    fillcolor=f"rgba({42 if cls=='ok' else 230 if cls=='critical' else 244},{157 if cls=='ok' else 57 if cls=='critical' else 162},{143 if cls=='ok' else 70 if cls=='critical' else 97},0.25)",
                    line=dict(color=TEAL if cls=='ok' else RED if cls=='critical' else ORANGE, width=2),
                ))
                fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1], gridcolor=BORDER),
                                               bgcolor=CARD, angularaxis=dict(gridcolor=BORDER)),
                                    showlegend=False, height=250, paper_bgcolor=DARK,
                                    font=dict(color="#f0f2f5"), margin=dict(t=20,b=20,l=20,r=20),
                                    title="Profil capteurs (normalisé)")
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("👈 Configurez les paramètres machine et lancez la prédiction.")

    # Scénarios comparatifs
    st.markdown('<div class="section-title">Simulation Multi-Scénarios</div>', unsafe_allow_html=True)

    if pipeline:
        scenarios = [
            ("Machine Neuve",      0.8,  42.0, 5.0, 22.0,  900.0, 13.0,  50.0,  80),
            ("Machine Mi-vie",     1.5,  58.0, 7.0, 35.0, 1100.0, 13.0, 300.0, 450),
            ("Machine Dégradée",   4.2,  85.0, 8.5, 55.0, 1200.0, 13.0, 450.0, 750),
            ("Machine Critique",   7.5,  92.0,12.0, 80.0, 1500.0, 13.0, 250.0,1000),
        ]
        cols = st.columns(4)
        for i, (name, vib_s, temp_s, curr_s, pres_s, rpm_s, amb_s, hours_s, cc_s) in enumerate(scenarios):
            inp = prepare_input(vib_s, temp_s, curr_s, pres_s, rpm_s, hours_s, amb_s, cc_s)
            rul_s = float(max(0, pipeline.predict(inp)[0]))
            color = RED if rul_s < 10 else ORANGE if rul_s < 30 else TEAL
            icon  = "🔴" if rul_s < 10 else "🟠" if rul_s < 30 else "🟢"
            with cols[i]:
                st.markdown(f"""
                <div class="kpi-card" style="border-color: {color}33;">
                    <div class="kpi-lbl">{name}</div>
                    <div class="kpi-val" style="color:{color}">{icon} {rul_s:.1f}h</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Performances Modèles":

    st.title("Performances des Modèles")
    st.markdown("Comparaison rigoureuse des 4 modèles — métriques sur jeu de test (20%, 4 809 obs.).")

    # KPIs meilleur modèle
    st.markdown('<div class="section-title">🏆 Modèle Final : Random Forest</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("MAE",  "3.077h",  "-6.7h vs baseline")
    k2.metric("RMSE", "5.575h",  "-9.5h vs baseline")
    k3.metric("R²",   "0.955",   "+0.284 vs baseline")
    k4.metric("CV MAE", "3.097h ± 0.073h", "Très stable")
    k5.metric("Gain FE", "+42%", "Feature Engineering")

    # Tableau comparatif
    results = {
        "Ridge Regression"  : {"MAE": 16.288, "RMSE": 20.468, "R²": 0.394, "CV MAE": 16.603, "Interprétabilité": "⭐⭐⭐⭐", "Complexité": "🟢 Très faible"},
        "Random Forest ✅"   : {"MAE":  3.077, "RMSE":  5.575, "R²": 0.955, "CV MAE":  3.097, "Interprétabilité": "⭐⭐⭐",   "Complexité": "🟡 Moyenne"},
        "XGBoost"           : {"MAE":  4.402, "RMSE":  6.542, "R²": 0.938, "CV MAE":  4.430, "Interprétabilité": "⭐⭐",     "Complexité": "🟡 Moyenne"},
        "MLP (Deep Learning)": {"MAE": 4.911, "RMSE":  7.604, "R²": 0.916, "CV MAE":  4.943, "Interprétabilité": "⭐",       "Complexité": "🔴 Élevée"},
    }
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Modèle"})

    c1, c2 = st.columns(2)
    with c1:
        fig_r2 = go.Figure()
        colors = [RED, TEAL, "#58a6ff", ORANGE]
        for i, row in res_df.iterrows():
            fig_r2.add_trace(go.Bar(name=row["Modèle"], x=[row["Modèle"]],
                                    y=[row["R²"]], marker_color=colors[i]))
        fig_r2.add_hline(y=0.9, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                 annotation_text="Objectif R²=0.90")
        fig_r2.update_layout(title="R² par modèle (↑ meilleur)", showlegend=False,
                              yaxis=dict(range=[0,1.05]))
        dark_fig(fig_r2)
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        fig_mae = go.Figure()
        for i, row in res_df.iterrows():
            fig_mae.add_trace(go.Bar(name=row["Modèle"], x=[row["Modèle"]],
                                     y=[row["MAE"]], marker_color=colors[i]))
        fig_mae.update_layout(title="MAE par modèle (↓ meilleur)", showlegend=False)
        dark_fig(fig_mae)
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown('<div class="section-title">Tableau Comparatif Complet</div>', unsafe_allow_html=True)
    st.dataframe(res_df[["Modèle","MAE","RMSE","R²","CV MAE","Interprétabilité","Complexité"]],
                 use_container_width=True, hide_index=True)

    # Feature Importance
    st.markdown('<div class="section-title">Importance des Variables — Random Forest</div>', unsafe_allow_html=True)
    fi_data = {
        "Feature": ["cumcount","hours_sq","hours_since_maintenance","rpm",
                    "temperature_motor","machine_type_enc","current_phase_avg",
                    "vib_x_temp","pressure_level","vibration_rms"],
        "Importance": [0.400, 0.235, 0.080, 0.038, 0.027, 0.020, 0.017, 0.013, 0.012, 0.010],
        "Type": ["🔴 Nouvelle","🔴 Nouvelle","🔵 Originale","🔵 Originale",
                 "🔵 Originale","🔵 Originale","🔵 Originale","🔴 Nouvelle","🔵 Originale","🔵 Originale"]
    }
    fi_df = pd.DataFrame(fi_data).sort_values("Importance", ascending=True)
    bar_colors = [RED if "Nouvelle" in t else "#58a6ff" for t in fi_df["Type"]]
    fig_fi = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"],
                              orientation='h', marker_color=bar_colors))
    fig_fi.update_layout(title="Feature Importance (🔴 nouvelles features  🔵 originales)")
    dark_fig(fig_fi, h=380)
    st.plotly_chart(fig_fi, use_container_width=True)

    new_total = sum(fi_data["Importance"][i] for i in [0,1,7])
    st.metric("Contribution des nouvelles features", f"{new_total*100:.1f}%",
              "cumcount + hours_sq + vib_x_temp")

    # CV Stabilité
    st.markdown('<div class="section-title">Stabilité — Validation Croisée 5-fold</div>', unsafe_allow_html=True)
    cv_data = {
        "Modèle": ["Ridge","Random Forest","XGBoost","MLP"],
        "CV MAE Mean": [16.603, 3.097, 4.430, 4.943],
        "CV MAE Std":  [0.114,  0.073, 0.114, 0.200],
    }
    cv_df = pd.DataFrame(cv_data)

    fig_cv = go.Figure()
    cv_colors = [RED, TEAL, "#58a6ff", ORANGE]
    for i, row in cv_df.iterrows():
        fig_cv.add_trace(go.Bar(
            name=row["Modèle"], x=[row["Modèle"]],
            y=[row["CV MAE Mean"]], error_y=dict(type="data", array=[row["CV MAE Std"]]),
            marker_color=cv_colors[i]
        ))
    fig_cv.update_layout(title="CV MAE ± Std (5-fold) — plus petit = meilleur et plus stable",
                         showlegend=False)
    dark_fig(fig_cv)
    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("""
**Lecture :** La barre = MAE moyen sur 5 folds. La barre d'erreur = écart-type entre les folds.
- Random Forest : 3.097 ± 0.073 → **très stable** (0.073h de variation entre folds)
- MLP : 4.943 ± 0.200 → plus instable (0.200h de variation)
    """)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Données Capteurs":

    st.title("Analyse des Données Capteurs")

    if df_raw is None:
        st.error("Dataset non trouvé.")
        st.stop()

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Enregistrements", f"{len(df_raw):,}")
    k2.metric("Machines", df_raw["machine_id"].nunique())
    k3.metric("RUL moyen", f"{df_raw['rul_hours'].mean():.1f}h")
    k4.metric("Pannes < 24h", f"{df_raw['failure_within_24h'].sum():,}")

    # Sélection machine
    st.markdown('<div class="section-title">Évolution temporelle d\'une machine</div>', unsafe_allow_html=True)
    machine_sel = st.selectbox("Machine", sorted(df_raw["machine_id"].unique()))
    m_df = df_raw[df_raw["machine_id"] == machine_sel].sort_values("timestamp").head(800)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=m_df["timestamp"], y=m_df["rul_hours"],
                                mode="lines", name="RUL",
                                line=dict(color=TEAL, width=2)))
    fig_ts.add_hrect(y0=0, y1=10, fillcolor=RED, opacity=0.1, line_width=0,
                     annotation_text="Zone critique < 10h")
    fig_ts.add_hrect(y0=10, y1=30, fillcolor=ORANGE, opacity=0.07, line_width=0,
                     annotation_text="Zone alerte 10-30h")
    fig_ts.add_hline(y=10, line_dash="dash", line_color=RED, line_width=1)
    fig_ts.add_hline(y=30, line_dash="dash", line_color=ORANGE, line_width=1)
    fig_ts.update_layout(title=f"Évolution du RUL — Machine #{machine_sel}",
                         xaxis_title="Temps", yaxis_title="RUL (h)")
    dark_fig(fig_ts, h=300)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Capteurs
    st.markdown('<div class="section-title">Capteurs en temps réel</div>', unsafe_allow_html=True)
    sensor_sel = st.multiselect("Capteurs à afficher",
                                ["vibration_rms","temperature_motor","rpm","pressure_level","current_phase_avg"],
                                default=["vibration_rms","temperature_motor"])

    if sensor_sel:
        fig_s = make_subplots(rows=len(sensor_sel), cols=1,
                              subplot_titles=sensor_sel, shared_xaxes=True)
        colors_s = [TEAL, ORANGE, RED, "#58a6ff", "#a78bfa"]
        for i, col_s in enumerate(sensor_sel):
            fig_s.add_trace(go.Scatter(x=m_df["timestamp"], y=m_df[col_s],
                                       mode="lines", name=col_s,
                                       line=dict(color=colors_s[i % len(colors_s)], width=1.5)),
                            row=i+1, col=1)
        fig_s.update_layout(height=100+200*len(sensor_sel), paper_bgcolor=DARK,
                            plot_bgcolor=CARD, font=dict(color="#f0f2f5"),
                            showlegend=False, margin=dict(t=30,b=20,l=20,r=20))
        for i in range(1, len(sensor_sel)+1):
            fig_s.update_xaxes(gridcolor=BORDER, row=i, col=1)
            fig_s.update_yaxes(gridcolor=BORDER, row=i, col=1)
        st.plotly_chart(fig_s, use_container_width=True)

    # Distribution par type de machine
    st.markdown('<div class="section-title">Comparaison entre machines</div>', unsafe_allow_html=True)
    cap_sel = st.selectbox("Capteur", ["vibration_rms","temperature_motor","rpm","pressure_level"])
    fig_box = px.box(df_raw, x="machine_type", y=cap_sel, color="machine_type",
                     color_discrete_sequence=[TEAL, ORANGE, RED, "#58a6ff"],
                     title=f"{cap_sel} par type de machine")
    dark_fig(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)
