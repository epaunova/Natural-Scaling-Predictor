# demo/app.py
"""
Neural Scaling Predictor – Streamlit demo
Стартиране локално:
    streamlit run demo/app.py
"""

import pathlib
import pandas as pd
import torch
import streamlit as st

# --- Импорт на модела от проекта -----------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(PROJECT_ROOT))
from src.models.nsp_model import ScalingLawPredictor  # noqa: E402

# ---------- UI -----------------------------------
st.set_page_config(page_title="Neural Scaling Predictor", page_icon="🔮")
st.title("🔮 Neural Scaling Predictor – Demo")
st.write(
    "Въведи параметри **N (params)**, **D (data)**, **C (compute)** "
    "или качи CSV/JSON с колони `params,data,compute`."
)

uploaded = st.file_uploader("📂 Upload CSV / JSON", type=["csv", "json"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_json(uploaded)
else:
    st.subheader("Ръчно въвеждане")
    params = st.number_input("Parameters N", 1e6, 1e14, 7e9, step=1e9, format="%.0f")
    data = st.number_input("Data tokens D", 1e9, 1e14, 1e12, step=1e11, format="%.0f")
    compute = st.number_input("Compute FLOPs C", 1e15, 1e23, 1e21, step=1e20, format="%.0f")
    df = pd.DataFrame({"params": [params], "data": [data], "compute": [compute]})

if st.button("🔍 Predict"):
    with st.spinner("Зареждам модела…"):
        model = ScalingLawPredictor()  # тук е зареден рандомно инициализиран модел
    x = torch.tensor(df.values, dtype=torch.float32)
    out = model(x)
    df["pred_loss"] = out["predictions"].detach().numpy().round(4)
    df["uncertainty"] = out["uncertainties"].detach().numpy().round(4)
    df["emergence"] = out["emergence_scores"].detach().numpy().round(4)
    st.success("Готово!")
    st.dataframe(df)

    st.line_chart(df.set_index("params")[["pred_loss"]])
