# demo/app.py  –  MOCK version (no PyTorch, random numbers for now)
# Run locally: streamlit run demo/app.py
# Live demo:   https://natural-scaling-predictor-…streamlit.app

import numpy as np
import pandas as pd
import streamlit as st

# ---------- page header ----------
st.set_page_config(page_title="Neural Scaling Predictor • Demo", page_icon="🔮")
st.title("🔮 Neural Scaling Predictor – Demo")

st.write(
    "Enter the three main scaling variables, **or** upload a CSV/JSON containing the "
    "columns **params, data, compute**, then click **Predict**."
)

# ---------- input section ----------
uploaded = st.file_uploader("📂 Upload CSV / JSON", type=["csv", "json"])

if uploaded:
    # Accept both .csv and .json
    df = (
        pd.read_csv(uploaded)
        if uploaded.name.endswith(".csv")
        else pd.read_json(uploaded)
    )
else:
    N = st.number_input(
        "Parameters N",
        min_value=1e6,
        max_value=1e14,
        value=7e9,
        step=1e9,
        format="%.0f",
    )
    D = st.number_input(
        "Data tokens D",
        min_value=1e9,
        max_value=1e14,
        value=1e12,
        step=1e11,
        format="%.0f",
    )
    C = st.number_input(
        "Compute FLOPs C",
        min_value=1e15,
        max_value=1e23,
        value=1e21,
        step=1e20,
        format="%.0f",
    )
    df = pd.DataFrame({"params": [N], "data": [D], "compute": [C]})

# ---------- prediction (mock) ----------
if st.button("🔍 Predict"):
    rng = np.random.default_rng(42)
    df["pred_loss"] = rng.normal(1.0, 0.1, len(df)).round(3)
    df["uncertainty"] = rng.uniform(0.05, 0.15, len(df)).round(3)
    df["emergence"] = rng.uniform(0, 1, len(df)).round(3)

    st.success("Results")
    st.dataframe(df)

    st.line_chart(df.set_index("params")[["pred_loss"]])
