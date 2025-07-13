# demo/app.py  –  MOCK версия без PyTorch
# Стартиране: streamlit run demo/app.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Neural Scaling Predictor (mock)", page_icon="🔮")
st.title("🔮 Neural Scaling Predictor – Demo")

st.write(
    "Въведи трите основни параметъра или качи CSV/JSON с колони "
    "`params,data,compute` и натисни **Predict**."
)

uploaded = st.file_uploader("📂 Upload CSV / JSON", type=["csv", "json"])

# ---------------- Четене на входните данни ----------------
if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_json(uploaded)
else:
    N = st.number_input("Parameters N", 1e6, 1e14, 7e9, step=1e9, format="%.0f")
    D = st.number_input("Data tokens D", 1e9, 1e14, 1e12, step=1e11, format="%.0f")
    C = st.number_input("Compute FLOPs C", 1e15, 1e23, 1e21, step=1e20, format="%.0f")
    df = pd.DataFra
