# -*- coding: utf-8 -*-
"""
Application Streamlit : 4 modèles de Machine Learning
  - Classification : Car data, Téléphone
  - Régression     : Car dataset, Taxi trip
Pour chaque modèle : Prédiction simple (formulaire) et Prédiction multiple (fichier CSV).

Lancer en local :  streamlit run app.py
Les fichiers .joblib doivent être dans le dossier  models/
"""
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
        st.success(f"{cfg['target']} prédit : **{value}**  (confiance : {res['Confiance'].iloc[0]:.0%})")
        proba = res.attrs["proba"].iloc[0].sort_values(ascending=False)
        st.bar_chart(proba)
    else:
        st.success(f"{cfg['target']} prédit : **{value:,.2f}**")


def simple_page(key):
    cfg = MODELS[key]
    st.subheader("Prédiction simple")
    values = {}
    with st.form(f"form_{key}"):
        cols = st.columns(2)
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
        submitted = st.form_submit_button("Prédire", type="primary")

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
    st.subheader("Prédiction multiple")
    st.write("Charge un fichier CSV avec ces colonnes :")
    st.code(", ".join(feature_names(cfg)))

    st.download_button(
        "Télécharger un fichier exemple",
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
    st.dataframe(out, use_container_width=True)
    st.download_button(
        "Télécharger les résultats",
        out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"predictions_{key}.csv",
        mime="text/csv",
    )


def home_page():
    st.title("🤖 Application de Machine Learning")
    st.write("Choisis un modèle :")
    for row in (ORDER[:2], ORDER[2:]):
        cols = st.columns(2)
        for col, key in zip(cols, row):
            cfg = MODELS[key]
            with col:
                st.button(
                    f"{cfg['icon']} {cfg['title']}",
                    key=f"home_{key}",
                    use_container_width=True,
                    on_click=go,
                    args=(key, "simple"),
                )
                kind = "Classification" if cfg["task"] == "classification" else "Régression"
                st.caption(f"{kind} — {cfg['description']}")


def model_page(key):
    cfg = MODELS[key]
    mode = st.session_state.get("mode") or "simple"

    st.button("← Retour à l'accueil", key="back", on_click=go, args=("home", None))
    st.title(f"{cfg['icon']} {cfg['title']}")
    st.caption("Classification" if cfg["task"] == "classification" else "Régression")

    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Prédiction simple",
            key="mode_simple",
            use_container_width=True,
            type="primary" if mode == "simple" else "secondary",
            on_click=go,
            args=(key, "simple"),
        )
    with c2:
        st.button(
            "Prédiction multiple",
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
    page = st.session_state.get("page", "home")
    if page in MODELS:
        model_page(page)
    else:
        home_page()


main()
