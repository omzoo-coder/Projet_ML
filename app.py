# -*- coding: utf-8 -*-
"""
Application Streamlit : 4 modèles de Machine Learning
  - Classification : Car data, Téléphone
  - Régression     : Car dataset, Taxi trip
Pour chaque modèle : Prédiction simple (formulaire) et Prédiction multiple (fichier CSV).

Lancer en local :  streamlit run app.py
Les fichiers .joblib doivent être dans le dossier  models/
"""
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------
# CONFIGURATION DES 4 MODÈLES
#   features  : colonnes du modèle, DANS L'ORDRE du notebook (sans la cible)
#   enc_cols  : colonnes catégorielles, dans l'ordre des encodeurs sauvegardés
#               (la cible y figure pour les modèles de classification)
#   num spec  : (valeur par défaut, minimum, pas)
# ---------------------------------------------------------------------
MODELS = {
    "car_data": {
        "title": "Car data",
        "icon": "🚗",
        "task": "classification",
        "description": "La voiture est-elle « Venant » ou « D'occasion » ?",
        "target": "Venant_Occasion",
        "enc_cols": ["Marque", "Boite_Vitesse", "Adresse", "Venant_Occasion"],
        "features": [
            ("Marque", "cat"),
            ("Année", "int", (2015, 1980, 1)),
            ("Boite_Vitesse", "cat"),
            ("Adresse", "cat"),
            ("Prix", "int", (7800000, 0, 100000)),
        ],
        "files": {
            "model": ["car_data_model.joblib"],
            "encoders": ["car_data_encoders.joblib"],
            "scaler": ["car_data_scaler.joblib"],
        },
    },
    "telephone": {
        "title": "Téléphone",
        "icon": "📱",
        "task": "classification",
        "description": "État du téléphone : D'occasion, Neuf, Réconditionné ou Venant.",
        "target": "etat",
        "enc_cols": ["adresse", "marque", "etat"],
        "features": [
            ("prix", "int", (362043, 0, 5000)),
            ("adresse", "cat"),
            ("marque", "cat"),
            ("dim_ecr", "float", (6.0, 0.0, 0.1)),
            ("ram", "int", (12, 1, 1)),
            ("stockage", "int", (158, 1, 1)),
        ],
        "files": {
            "model": ["telephone_model.joblib"],
            "encoders": ["telephone_encoders.joblib"],
            "scaler": ["telephone_scaler.joblib"],
        },
    },
    "car_dataset": {
        "title": "Car dataset",
        "icon": "💰",
        "task": "regression",
        "description": "Prix de vente (Selling_Price) d'une voiture.",
        "target": "Selling_Price",
        "enc_cols": ["Fuel_Type", "Seller_Type", "Transmission"],
        "features": [
            ("Kms_Driven", "int", (71822, 0, 1000)),
            ("Present_Price", "float", (15.38, 0.0, 0.1)),
            ("Fuel_Type", "cat"),
            ("Seller_Type", "cat"),
            ("Transmission", "cat"),
            ("Age", "int", (8, 0, 1)),
        ],
        "files": {
            "model": ["car_dataset_model.joblib", "car_dataset_rf_model.joblib", "rf_model.joblib"],
            "encoders": ["car_dataset_encoders.joblib", "encoders.joblib"],
            "scaler": ["car_dataset_scaler.joblib", "scaler.joblib"],
        },
    },
    "taxi_trip": {
        "title": "Taxi trip",
        "icon": "🚕",
        "task": "regression",
        "description": "Prix d'une course de taxi (Trip_Price).",
        "target": "Trip_Price",
        "unit": "$",
        "enc_cols": ["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"],
        "features": [
            ("Trip_Distance_km", "float", (25.83, 0.0, 0.5)),
            ("Time_of_Day", "cat"),
            ("Day_of_Week", "cat"),
            ("Passenger_Count", "int", (2, 1, 1)),
            ("Traffic_Conditions", "cat"),
            ("Weather", "cat"),
            ("Base_Fare", "float", (3.52, 0.0, 0.1)),
            ("Per_Km_Rate", "float", (1.22, 0.0, 0.01)),
            ("Per_Minute_Rate", "float", (0.29, 0.0, 0.01)),
            ("Trip_Duration_Minutes", "float", (61.86, 0.0, 1.0)),
        ],
        "files": {
            "model": ["taxi_trip_rf_model.joblib", "taxi_trip_model.joblib"],
            "encoders": ["taxi_trip_encoders.joblib"],
            "scaler": ["taxi_trip_scaler.joblib"],
        },
    },
}
ORDER = ["car_data", "telephone", "car_dataset", "taxi_trip"]


# ---------------------------------------------------------------------
# CHARGEMENT DES FICHIERS .joblib
# ---------------------------------------------------------------------
@st.cache_resource
def _load_files(model_path, scaler_path, encoders_path, enc_cols, _stamp):
    """Charge les 3 fichiers (mis en cache uniquement quand le chargement réussit)."""
    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "encoders": dict(zip(enc_cols, joblib.load(encoders_path))),
    }


def load_artifacts(key):
    """Renvoie (dict, erreur). Les erreurs ne sont JAMAIS mises en cache."""
    cfg = MODELS[key]
    found, missing = {}, []
    for kind, candidates in cfg["files"].items():
        path = next((MODELS_DIR / n for n in candidates if (MODELS_DIR / n).exists()), None)
        if path is None:
            missing.append(candidates[0])
        else:
            found[kind] = path
    if missing:
        return None, "Fichier(s) introuvable(s) dans le dossier models/ : " + ", ".join(missing)

    stamp = tuple(p.stat().st_mtime for p in found.values())
    try:
        art = _load_files(
            str(found["model"]), str(found["scaler"]), str(found["encoders"]),
            tuple(cfg["enc_cols"]), stamp,
        )
    except Exception as e:  # version de scikit-learn différente, fichier abîmé...
        return None, f"Impossible de charger les fichiers de ce modèle : {type(e).__name__} : {e}"
    return art, None


# ---------------------------------------------------------------------
# PRÉDICTION (même préparation que dans les notebooks)
# ---------------------------------------------------------------------
def feature_names(cfg):
    return [f[0] for f in cfg["features"]]


def predict(df_raw, key):
    """df_raw : colonnes brutes du modèle. Renvoie (DataFrame résultat, erreur)."""
    cfg = MODELS[key]
    art, err = load_artifacts(key)
    if err:
        return None, err
    names = feature_names(cfg)

    X = df_raw[names].copy()

    # colonnes catégorielles -> valeurs connues + encodage
    for spec in cfg["features"]:
        col, kind = spec[0], spec[1]
        if kind == "cat":
            X[col] = X[col].astype(str).str.strip()
            known = set(art["encoders"][col].classes_)
            bad = sorted(set(X[col]) - known)
            if bad:
                return None, f"Valeur(s) inconnue(s) dans « {col} » : {', '.join(bad[:5])}"
            X[col] = art["encoders"][col].transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        cols = X.columns[X.isna().any()].tolist()
        return None, "Valeurs manquantes ou non numériques dans : " + ", ".join(cols)

    # normalisation puis prédiction
    Xs = art["scaler"].transform(X.values.astype(float))
    pred = art["model"].predict(Xs)

    out = pd.DataFrame(index=df_raw.index)
    if cfg["task"] == "classification":
        target_enc = art["encoders"][cfg["target"]]
        out["Prédiction"] = target_enc.inverse_transform(np.round(pred).astype(int))
        proba = art["model"].predict_proba(Xs)
        out["Confiance"] = proba.max(axis=1)
        classes = target_enc.inverse_transform(art["model"].classes_.astype(int))
        out.attrs["proba"] = pd.DataFrame(proba, columns=classes, index=df_raw.index)
    else:
        out["Prédiction"] = np.round(pred, 2)
    return out, None


# ---------------------------------------------------------------------
# STYLE (CSS)
# ---------------------------------------------------------------------
CSS = """
<style>
.block-container {max-width: 880px; padding-top: 2rem; padding-bottom: 3rem;}
footer {visibility: hidden;}

/* Bandeau principal */
.hero {background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: #fff;
       padding: 2rem 2rem; border-radius: 20px; margin: 0 0 1.4rem 0;
       box-shadow: 0 12px 30px rgba(79,70,229,.28);}
.hero h1 {color: #fff; margin: 0; padding: 0; font-size: 2.1rem; line-height: 1.2;}
.hero p {color: rgba(255,255,255,.92); margin: .5rem 0 0 0; font-size: 1.02rem;}
.hero-flex {display: flex; align-items: center; gap: 1.2rem;}
.hero-icon {font-size: 3rem; background: rgba(255,255,255,.18); border-radius: 18px;
            padding: .4rem .9rem; line-height: 1.2;}
.hero-pill {display: inline-block; margin-top: .6rem; padding: .15rem .8rem; border-radius: 999px;
            background: rgba(255,255,255,.22); font-size: .78rem; font-weight: 600;}

/* Cartes de la page d'accueil */
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}
.card-icon {font-size: 2.6rem; line-height: 1.1;}
.card-title {font-size: 1.3rem; font-weight: 700; margin: .3rem 0 .4rem 0;}
.badge {display: inline-block; padding: .15rem .75rem; border-radius: 999px;
        font-size: .75rem; font-weight: 600; background: rgba(99,102,241,.16); color: #6366F1;}
.badge.reg {background: rgba(16,185,129,.16); color: #10B981;}
.card-desc {opacity: .75; font-size: .92rem; margin: .6rem 0 .4rem 0; min-height: 2.6rem;}

/* Boutons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
    border-radius: 12px; font-weight: 600; padding: .55rem 1rem; transition: all .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 6px 16px rgba(79,70,229,.22);}
button[kind="primary"], button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important; border: none !important; color: #fff !important;}

/* Carte résultat */
.result {background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: #fff;
         padding: 1.5rem 1.6rem; border-radius: 18px; margin: 1rem 0 .8rem 0;
         box-shadow: 0 10px 26px rgba(79,70,229,.28);}
.result-label {font-size: .85rem; opacity: .85; text-transform: uppercase; letter-spacing: .06em;}
.result-value {font-size: 2.3rem; font-weight: 800; margin: .2rem 0; line-height: 1.2;}
.result-sub {font-size: .95rem; opacity: .9;}

/* Barres de probabilité */
.pb {margin: .55rem 0;}
.pb-top {display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: .25rem;}
.pb-track {background: rgba(128,128,128,.22); border-radius: 999px; height: 10px; overflow: hidden;}
.pb-fill {background: linear-gradient(90deg, #4F46E5, #7C3AED); height: 100%; border-radius: 999px;}

.section-title {font-size: 1.15rem; font-weight: 700; margin: 1.2rem 0 .4rem 0;}
.footer-note {text-align: center; opacity: .55; font-size: .82rem; margin-top: 2.5rem;}
</style>
"""


def esc(x):
    return html.escape(str(x))


def kind_label(cfg):
    return "Classification" if cfg["task"] == "classification" else "Régression"


def badge(cfg):
    css = "badge" if cfg["task"] == "classification" else "badge reg"
    return f'<span class="{css}">{kind_label(cfg)}</span>'


def result_card(label, value, sub=""):
    sub_html = f'<div class="result-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="result"><div class="result-label">{esc(label)}</div>'
        f'<div class="result-value">{value}</div>{sub_html}</div>'
    )


def prob_bars(proba):
    rows = []
    for name, p in proba.items():
        rows.append(
            f'<div class="pb"><div class="pb-top"><span>{esc(name)}</span><span>{p:.1%}</span></div>'
            f'<div class="pb-track"><div class="pb-fill" style="width:{p * 100:.1f}%"></div></div></div>'
        )
    return "".join(rows)


# ---------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------
def go(page, mode=None):
    st.session_state["page"] = page
    st.session_state["mode"] = mode


def cat_options(key, col):
    art, _ = load_artifacts(key)
    return [str(v) for v in art["encoders"][col].classes_]


def show_result(res, cfg):
    value = res["Prédiction"].iloc[0]
    if cfg["task"] == "classification":
        conf = res["Confiance"].iloc[0]
        st.markdown(
            result_card(cfg["target"], esc(value), f"Confiance du modèle : {conf:.0%}"),
            unsafe_allow_html=True,
        )
        proba = res.attrs["proba"].iloc[0].sort_values(ascending=False)
        st.markdown('<div class="section-title">Probabilité par classe</div>', unsafe_allow_html=True)
        st.markdown(prob_bars(proba), unsafe_allow_html=True)
    else:
        unit = cfg.get("unit", "")
        st.markdown(
            result_card(cfg["target"], f"{value:,.2f} {unit}".strip(), "Valeur prédite par le modèle"),
            unsafe_allow_html=True,
        )


def simple_page(key):
    cfg = MODELS[key]
    st.markdown('<div class="section-title">Prédiction simple</div>', unsafe_allow_html=True)
    st.caption("Remplis les champs puis appuie sur « Prédire ».")
    values = {}
    with st.form(f"form_{key}"):
        cols = st.columns(2, gap="medium")
        for i, spec in enumerate(cfg["features"]):
            name, kind = spec[0], spec[1]
            with cols[i % 2]:
                if kind == "cat":
                    values[name] = st.selectbox(name, cat_options(key, name), key=f"{key}_{name}")
                elif kind == "int":
                    default, minimum, step = spec[2]
                    values[name] = st.number_input(
                        name, min_value=int(minimum), value=int(default), step=int(step), key=f"{key}_{name}"
                    )
                else:
                    default, minimum, step = spec[2]
                    values[name] = st.number_input(
                        name, min_value=float(minimum), value=float(default), step=float(step), key=f"{key}_{name}"
                    )
        submitted = st.form_submit_button("🔮 Prédire", type="primary", use_container_width=True)

    if submitted:
        res, err = predict(pd.DataFrame([values]), key)
        if err:
            st.error(err)
        else:
            show_result(res, cfg)


def template_df(key):
    cfg = MODELS[key]
    row = {}
    for spec in cfg["features"]:
        name, kind = spec[0], spec[1]
        row[name] = cat_options(key, name)[0] if kind == "cat" else spec[2][0]
    return pd.DataFrame([row])


def multiple_page(key):
    cfg = MODELS[key]
    st.markdown('<div class="section-title">Prédiction multiple</div>', unsafe_allow_html=True)
    st.caption("Charge un fichier CSV contenant ces colonnes :")
    st.code(", ".join(feature_names(cfg)))

    st.download_button(
        "⬇️ Télécharger un fichier exemple",
        template_df(key).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"exemple_{key}.csv",
        mime="text/csv",
    )

    up = st.file_uploader("Fichier CSV", type="csv", key=f"upload_{key}")
    if up is None:
        return
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    missing = [c for c in feature_names(cfg) if c not in df.columns]
    if missing:
        st.error("Colonne(s) manquante(s) : " + ", ".join(missing))
        return

    res, err = predict(df, key)
    if err:
        st.error(err)
        return

    out = df.copy()
    out[f"{cfg['target']}_prédit"] = res["Prédiction"].values
    if cfg["task"] == "classification":
        out["Confiance"] = res["Confiance"].round(3).values

    st.success(f"{len(out)} prédiction(s) effectuée(s).")

    # petit résumé
    m1, m2, m3 = st.columns(3)
    m1.metric("Lignes", f"{len(out)}")
    if cfg["task"] == "classification":
        counts = res["Prédiction"].value_counts()
        m2.metric("Classe la plus fréquente", str(counts.index[0]))
        m3.metric("Confiance moyenne", f"{res['Confiance'].mean():.0%}")
        st.bar_chart(counts)
    else:
        unit = cfg.get("unit", "")
        m2.metric("Prédiction moyenne", f"{res['Prédiction'].mean():,.2f} {unit}".strip())
        m3.metric("Min – Max", f"{res['Prédiction'].min():,.0f} – {res['Prédiction'].max():,.0f}")

    st.dataframe(out, use_container_width=True)
    st.download_button(
        "⬇️ Télécharger les résultats",
        out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"predictions_{key}.csv",
        mime="text/csv",
        type="primary",
    )


def home_page():
    st.markdown(
        '<div class="hero"><h1>🤖 Application de Machine Learning</h1>'
        "<p>4 modèles de prédiction : 2 de classification et 2 de régression. "
        "Choisis un modèle pour commencer.</p></div>",
        unsafe_allow_html=True,
    )
    for row in (ORDER[:2], ORDER[2:]):
        cols = st.columns(2, gap="medium")
        for col, key in zip(cols, row):
            cfg = MODELS[key]
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="card-icon">{cfg["icon"]}</div>'
                        f'<div class="card-title">{esc(cfg["title"])}</div>'
                        f"{badge(cfg)}"
                        f'<div class="card-desc">{esc(cfg["description"])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Ouvrir →",
                        key=f"home_{key}",
                        use_container_width=True,
                        type="primary",
                        on_click=go,
                        args=(key, "simple"),
                    )


def model_page(key):
    cfg = MODELS[key]
    mode = st.session_state.get("mode") or "simple"

    st.button("← Retour à l'accueil", key="back", on_click=go, args=("home", None))
    st.markdown(
        f'<div class="hero"><div class="hero-flex"><div class="hero-icon">{cfg["icon"]}</div>'
        f'<div><h1>{esc(cfg["title"])}</h1><p>{esc(cfg["description"])}</p>'
        f'<span class="hero-pill">{kind_label(cfg)}</span></div></div></div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.button(
            "✍️ Prédiction simple",
            key="mode_simple",
            use_container_width=True,
            type="primary" if mode == "simple" else "secondary",
            on_click=go,
            args=(key, "simple"),
        )
    with c2:
        st.button(
            "📄 Prédiction multiple",
            key="mode_multiple",
            use_container_width=True,
            type="primary" if mode == "multiple" else "secondary",
            on_click=go,
            args=(key, "multiple"),
        )

    _, err = load_artifacts(key)
    if err:
        st.error(err)
        return

    if mode == "simple":
        simple_page(key)
    else:
        multiple_page(key)


def main():
    st.set_page_config(page_title="ML App", page_icon="🤖", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    page = st.session_state.get("page", "home")
    if page in MODELS:
        model_page(page)
    else:
        home_page()
    st.markdown('<div class="footer-note">Projet Machine Learning · 4 modèles</div>', unsafe_allow_html=True)


main()
