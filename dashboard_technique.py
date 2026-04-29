"""
Dashboard Technique — EDA & Preprocessing
Pour les Data Scientists / équipes techniques

Lancer : streamlit run dashboard_technique.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dashboard Technique — EDA & Preprocessing",
                   page_icon="", layout="wide")




st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
* { font-family: 'IBM Plex Sans', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }
.stApp { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
.metric-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 16px; text-align: center; margin: 4px; }
.metric-val { font-size: 2rem; font-weight: 700; color: #58a6ff; font-family: 'JetBrains Mono'; }
.metric-lbl { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.tag { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem;
       font-family: 'JetBrains Mono'; margin: 2px; }
.tag-new { background: #1f4a1f; color: #56d364; border: 1px solid #56d364; }
.tag-orig { background: #1a2f5a; color: #58a6ff; border: 1px solid #58a6ff; }
.tag-drop { background: #4a1f1f; color: #f85149; border: 1px solid #f85149; }
.section-header { font-size: 1.1rem; font-weight: 600; color: #58a6ff;
                  border-bottom: 1px solid #30363d; padding-bottom: 8px; margin: 16px 0 12px 0; }
div[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d;
                               border-radius: 8px; padding: 12px; }
            

/* Métriques — label et valeur bien visibles */
div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}

div[data-testid="stMetricLabel"] p {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 1.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricDelta"] {
    color: #56d364 !important;
}

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
    st.markdown("### Dashboard Technique")
    st.markdown("**EDA & Preprocessing**")
    st.markdown("---")
    page = st.radio("Section", [
        "Analyse du Dataset",
        "Valeurs Manquantes",
        "Distributions",
        "Corrélations",
        "Feature Engineering",
        "Pipeline Preprocessing",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Données**")
    st.markdown("- 24 042 enregistrements")
    st.markdown("- 15 variables")
    st.markdown("- 20 machines")
    st.markdown("- 14 jours")

# ─── Chargement données ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("predictive_maintenance_v3.csv", parse_dates=["timestamp"])
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        df["cumcount"]    = df.groupby("machine_id").cumcount()
        df["hours_sq"]    = df["hours_since_maintenance"] ** 2
        df["vib_x_temp"]  = df["vibration_rms"] * df["temperature_motor"]
        df["vib_x_hours"] = df["vibration_rms"] * df["hours_since_maintenance"]
        df["temp_x_hours"]= df["temperature_motor"] * df["hours_since_maintenance"]
        return df
    except:
        return None

df = load_data()
DARK = "#0d1117"; CARD = "#161b22"; BORDER = "#30363d"
BLUE = "#58a6ff"; GREEN = "#56d364"; RED = "#f85149"; ORANGE = "#d29922"

def dark_fig(fig, h=350):
    fig.update_layout(height=h, paper_bgcolor=DARK, plot_bgcolor=CARD,
        font=dict(color="#e6edf3", family="IBM Plex Sans"),
        margin=dict(t=40,b=20,l=20,r=20),
        xaxis=dict(gridcolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor=BORDER, showgrid=True))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
if page == "Analyse du Dataset":

    st.title("Analyse du Dataset")
    st.markdown("Vue d'ensemble du dataset industriel de maintenance prédictive.")

    if df is None:
        st.error("Dataset non trouvé. Placer `predictive_maintenance_v3.csv` dans le dossier.")
        st.stop()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Enregistrements", f"{len(df):,}")
    c2.metric("Variables", "15")
    c3.metric("Machines", df["machine_id"].nunique())
    c4.metric("Jours couverts", "14")
    c5.metric("Taux NaN moyen", "3.3%")

    st.markdown('<div class="section-header">Variables du dataset</div>', unsafe_allow_html=True)

    vars_info = {
        "timestamp"             : ("datetime", "drop",  "Horodatage : non généralisable"),
        "machine_id"            : ("int64",    "drop",  "ID machine : non généralisable"),
        "machine_type"          : ("object",   "keep",  "Type de machine : encodé"),
        "vibration_rms"         : ("float64",  "keep",  "Vibration RMS (g) : capteur clé"),
        "temperature_motor"     : ("float64",  "keep",  "Température moteur (°C)"),
        "current_phase_avg"     : ("float64",  "keep",  "Courant de phase moyen (A)"),
        "pressure_level"        : ("float64",  "keep",  "Pression (bar)"),
        "rpm"                   : ("float64",  "keep",  "Vitesse rotation (RPM)"),
        "operating_mode"        : ("object",   "keep",  "Mode opératoire : encodé"),
        "hours_since_maintenance": ("float64", "keep",  "Heures depuis maintenance : très important"),
        "ambient_temp"          : ("float64",  "keep",  "Température ambiante (°C)"),
        "rul_hours"             : ("float64",  "target","VARIABLE CIBLE"),
        "failure_within_24h"    : ("int64",    "drop",  "Dérivé du RUL : data leakage"),
        "failure_type"          : ("object",   "drop",  "Connu après panne : data leakage"),
        "estimated_repair_cost" : ("int64",    "drop",  "Corrélé à la panne : data leakage"),
    }

    rows = []
    for var, (dtype, status, desc) in vars_info.items():
        color = {"drop":"🔴","keep":"🟢","target":""}[status]
        action = {"drop":"Supprimée","keep":"Gardée","target":"Cible"}[status]
        rows.append({"Variable": var, "Type": dtype, "Statut": f"{color} {action}", "Description": desc})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Statistiques descriptives</div>', unsafe_allow_html=True)
    num_cols = ["vibration_rms","temperature_motor","current_phase_avg",
                "pressure_level","rpm","hours_since_maintenance","ambient_temp","rul_hours"]
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Valeurs Manquantes":

    st.title("Analyse des Valeurs Manquantes")

    if df is None: st.stop()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Variable": missing.index, "Count": missing.values,
                             "Pct (%)": missing_pct.values})
    miss_df = miss_df[miss_df["Count"] > 0].sort_values("Pct (%)", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Variables avec NaN</div>', unsafe_allow_html=True)
        fig = px.bar(miss_df, x="Variable", y="Pct (%)", color="Pct (%)",
                     color_continuous_scale=[[0,"#1f4a1f"],[0.5,"#d29922"],[1,"#f85149"]],
                     title="Pourcentage de valeurs manquantes")
        fig.add_hline(y=5, line_dash="dash", line_color=RED,
                      annotation_text="Seuil critique 5%")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Détail</div>', unsafe_allow_html=True)
        st.dataframe(miss_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        st.markdown("""
**Stratégie retenue : Imputation par la médiane**
- Toutes les variables ont < 5% de NaN → acceptable
- La médiane est robuste aux outliers
- Appliquée dans le pipeline sklearn (uniquement sur le train set)
        """)

    st.markdown('<div class="section-header">Justification méthodologique</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.info("**Médiane vs Moyenne**\n\nLa médiane n'est pas influencée par les pics de capteurs (outliers). Ex : un pic de vibration à 10g ne biaise pas l'imputation.")
    col2.success("**Pipeline sklearn**\n\nL'imputation est ajustée uniquement sur X_train, puis appliquée à X_test. Aucune information du test ne contamne l'entraînement.")
    col3.warning("**Variables catégorielles**\n\nOpérating_mode et machine_type n'ont pas de NaN → pas d'imputation nécessaire, encodage direct.")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Distributions":

    st.title("Distributions des Variables")

    if df is None: st.stop()

    st.markdown('<div class="section-header">Variable cible : rul_hours</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="rul_hours", nbins=60,
                           color_discrete_sequence=[BLUE],
                           title="Distribution de rul_hours")
        fig.add_vline(x=df["rul_hours"].median(), line_dash="dash",
                      line_color=GREEN, annotation_text="Médiane: 22.6h")
        fig.add_vline(x=df["rul_hours"].mean(), line_dash="dash",
                      line_color=ORANGE, annotation_text="Moyenne: 27.8h")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(df, x="machine_type", y="rul_hours", color="machine_type",
                      title="RUL par type de machine",
                      color_discrete_sequence=[BLUE, GREEN, ORANGE, RED])
        dark_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Capteurs par type de machine</div>', unsafe_allow_html=True)
    sensor_sel = st.selectbox("Capteur", ["vibration_rms","temperature_motor","rpm","pressure_level"])
    fig3 = px.violin(df, x="machine_type", y=sensor_sel, color="machine_type", box=True,
                     title=f"Distribution de {sensor_sel} par machine",
                     color_discrete_sequence=[BLUE, GREEN, ORANGE, RED])
    dark_fig(fig3, h=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">RUL par mode opératoire</div>', unsafe_allow_html=True)
    fig4 = px.box(df, x="operating_mode", y="rul_hours", color="operating_mode",
                  title="RUL médian selon le mode opératoire",
                  color_discrete_sequence=[GREEN, ORANGE, RED])
    dark_fig(fig4)
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Corrélations":

    st.title("Analyse des Corrélations")

    if df is None: st.stop()

    num_cols = ["vibration_rms","temperature_motor","current_phase_avg",
                "pressure_level","rpm","hours_since_maintenance","ambient_temp","rul_hours"]

    corr = df[num_cols].corr().round(2)

    c1, c2 = st.columns([3,2])
    with c1:
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Matrice de corrélation")
        dark_fig(fig, h=450)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Corrélation avec RUL</div>', unsafe_allow_html=True)
        corr_rul = corr["rul_hours"].drop("rul_hours").sort_values(key=abs, ascending=True)
        colors_bar = [RED if v < 0 else GREEN for v in corr_rul.values]
        fig2 = go.Figure(go.Bar(x=corr_rul.values, y=corr_rul.index,
                                orientation='h', marker_color=colors_bar))
        fig2.update_layout(title="Corrélation avec rul_hours",
                           xaxis_title="Corrélation de Pearson")
        dark_fig(fig2, h=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
**Observations clés :**
- `hours_since_maintenance` : -0.307 (plus forte corrélation)
- Capteurs physiques : corrélations faibles (-0.08 à 0.02)
- **Relations non-linéaires** → justifie RF/XGBoost
- Multicolinéarité entre capteurs (0.74-0.88)
        """)

    st.markdown('<div class="section-header">Relation capteur → RUL</div>', unsafe_allow_html=True)
    sel = st.selectbox("Capteur à analyser", ["hours_since_maintenance","vibration_rms","temperature_motor","rpm"])
    sample = df.sample(3000, random_state=42)
    fig3 = px.scatter(sample, x=sel, y="rul_hours", color="machine_type",
                      opacity=0.5, trendline="lowess",
                      title=f"{sel} vs rul_hours (avec tendance LOWESS)")
    dark_fig(fig3, h=380)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Engineering":

    st.title("Feature Engineering Temporel")
    st.markdown("Les nouvelles features ont permis de passer de R²=0.671 à **R²=0.955** (+42%).")

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Baseline", "0.671", "-")
    c2.metric("R² Après FE", "0.955", "+0.284")
    c3.metric("Gain", "+42%", "↑")

    st.markdown('<div class="section-header">Les 5 nouvelles features</div>', unsafe_allow_html=True)

    features_info = [
        ("cumcount", "40.0%", "🔴 Clé",
         "Nombre de mesures depuis le début du cycle de la machine.",
         "df['cumcount'] = df.groupby('machine_id').cumcount()",
         "Proxy direct du vieillissement. Disponible en temps réel → pas de data leakage."),
        ("hours_sq", "23.5%", "🔴 Importante",
         "Carré des heures depuis la dernière maintenance.",
         "df['hours_sq'] = df['hours_since_maintenance'] ** 2",
         "Capture la relation non-linéaire : l'usure accélère avec le temps."),
        ("vib_x_temp", "1.2%", "🔵 Complémentaire",
         "Produit vibration × température moteur.",
         "df['vib_x_temp'] = df['vibration_rms'] * df['temperature_motor']",
         "Stress thermomécanique combiné — plus destructeur que chaque capteur séparément."),
        ("vib_x_hours", "~1%", "🔵 Complémentaire",
         "Produit vibration × heures depuis maintenance.",
         "df['vib_x_hours'] = df['vibration_rms'] * df['hours_since_maintenance']",
         "Vibration cumulée dans le temps — fatigue vibratoire."),
        ("temp_x_hours", "~1%", "🔵 Complémentaire",
         "Produit température × heures depuis maintenance.",
         "df['temp_x_hours'] = df['temperature_motor'] * df['hours_since_maintenance']",
         "Chaleur cumulée dans le temps — fatigue thermique."),
    ]

    for name, importance, tag, desc, code, justif in features_info:
        with st.expander(f"**{name}** — {importance} d'importance — {tag}"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown(f"**Description :** {desc}")
                st.code(code, language="python")
            with col2:
                st.markdown(f"**Justification métier :** {justif}")

    st.markdown('<div class="section-header">Visualisation : cumcount vs RUL</div>', unsafe_allow_html=True)
    if df is not None:
        machine_ids = sorted(df["machine_id"].unique())
        selected = st.selectbox("Machine", machine_ids[:5])
        m_df = df[df["machine_id"] == selected].copy()

        fig = make_subplots(rows=1, cols=2, subplot_titles=["cumcount vs RUL", "hours_since_maintenance vs RUL"])
        fig.add_trace(go.Scatter(x=m_df["cumcount"], y=m_df["rul_hours"],
                                 mode="markers", marker=dict(color=RED, size=4, opacity=0.6),
                                 name="cumcount"), row=1, col=1)
        fig.add_trace(go.Scatter(x=m_df["hours_since_maintenance"], y=m_df["rul_hours"],
                                 mode="markers", marker=dict(color=BLUE, size=4, opacity=0.6),
                                 name="hours_since_maint"), row=1, col=2)
        fig.update_layout(height=350, paper_bgcolor=DARK, plot_bgcolor=CARD,
                          font=dict(color="#e6edf3"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        corr_cc = m_df["cumcount"].corr(m_df["rul_hours"])
        corr_hm = m_df["hours_since_maintenance"].corr(m_df["rul_hours"])
        st.info(f"Corrélation cumcount↔RUL : **{corr_cc:.3f}** vs hours_since_maintenance↔RUL : **{corr_hm:.3f}**")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Pipeline Preprocessing":

    st.title("Pipeline de Preprocessing")
    st.markdown("Architecture complète du pipeline sklearn garantissant l'absence de data leakage.")

    st.markdown('<div class="section-header">Schéma du pipeline</div>', unsafe_allow_html=True)
    st.code("""
# 1. Split AVANT tout preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ↑ Les données de test ne voient JAMAIS l'entraînement

# 2. Transformers numériques (12 features)
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # NaN → médiane du TRAIN
    ('scaler',  StandardScaler())                   # Normalisation du TRAIN
])

# 3. ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    # cat_features = [] car encodage catégoriel fait manuellement
])

# 4. Pipeline complet
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=300, ...))
])

# 5. Entraînement : fit UNIQUEMENT sur X_train
pipeline.fit(X_train, y_train)
# → scaler apprend mean/std sur X_train seulement

# 6. Prédiction : transform avec stats du train
pipeline.predict(X_test)
# → applique mean/std du train au test → pas de contamination
    """, language="python")

    st.markdown('<div class="section-header">Garanties anti-leakage</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.success("✅ **Split avant preprocessing**\nLe train/test split est fait sur les données brutes, avant toute transformation.")
    c2.success("✅ **Pipeline sklearn**\nSimpleImputer et StandardScaler sont ajustés uniquement sur X_train via `fit_transform`.")
    c3.success("✅ **Variables exclues**\n`failure_within_24h`, `failure_type`, `estimated_repair_cost` supprimées — fuites causales.")

    st.markdown('<div class="section-header">Variables exclues et pourquoi</div>', unsafe_allow_html=True)
    exclusions = [
        ("timestamp", "Identifiant temporel", "Non généralisable à de nouvelles machines"),
        ("machine_id", "Identifiant machine", "Le modèle apprendrait par machine — non généralisable"),
        ("failure_within_24h", "🚨 Data Leakage", "Dérivé directement de rul_hours : si RUL < 24 → 1"),
        ("failure_type", "🚨 Data Leakage", "Connu seulement APRÈS la panne — impossible en prédiction"),
        ("estimated_repair_cost", "🚨 Data Leakage", "Corrélé à la panne — connu après l'événement"),
    ]
    df_excl = pd.DataFrame(exclusions, columns=["Variable", "Raison", "Explication"])
    st.dataframe(df_excl, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Validation croisée 5-fold</div>', unsafe_allow_html=True)
    cv_data = {
        "Modèle": ["Ridge Regression", "Random Forest", "XGBoost", "MLP (Deep Learning)"],
        "CV MAE Mean": [16.603, 3.097, 4.430, 4.943],
        "CV MAE Std": [0.114, 0.073, 0.114, 0.200],
        "Stabilité": ["Faible", "🏆 Très élevée", "Élevée", "Moyenne"],
    }
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)
    st.info("**KFold(n_splits=5, shuffle=True)** — Le preprocessing est refit à chaque fold pour éviter tout leakage inter-fold.")
