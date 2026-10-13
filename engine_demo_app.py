"""
Example Streamlit page that uses apriori_engine.
Run:  python -m streamlit run engine_demo_app.py
(Place this file next to src/ and keep apriori_engine.py inside src/.)
"""
import pandas as pd
import streamlit as st

from src import apriori_engine as ae

st.title("Student Behaviour - Apriori Explorer")

# ---- 1. Data ----
up = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")
df = pd.read_csv(up) if up else ae.make_demo_data()
st.sidebar.caption("Using uploaded file" if up else "Using demo data")

# ---- 2. Choose variables to include / club ----
types = ae.detect_column_types(df)
columns = st.sidebar.multiselect("Variables to include", list(df.columns),
                                 default=list(df.columns)[:4])

# ---- 3. Binning options for numeric variables ----
st.sidebar.subheader("Binning")
method = st.sidebar.selectbox("Method", ["quantile", "equal_width"])
n_bins = st.sidebar.selectbox("Bins", [2, 3, 4, 5], index=1)
default_spec = ae.BinSpec(method=method, n_bins=n_bins)

# ---- 4. Thresholds ----
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.50, 0.10, 0.01)
min_conf = st.sidebar.slider("Minimum Confidence", 0.10, 1.00, 0.60, 0.05)
max_len = st.sidebar.slider("Max itemset size (k)", 2, 6, 4)

if st.sidebar.button("Run Apriori") and columns:
    res = ae.run_pipeline(df, columns, min_support, min_conf,
                          default_spec=default_spec, max_len=max_len)
    s = res["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", s["transactions"])
    c2.metric("Frequent itemsets", s["frequent_itemsets"])
    c3.metric("Rules", s["rules"])
    c4.metric("Runtime (s)", s["runtime_sec"])

    st.subheader("Transactions (first 5)")
    st.write(res["transactions"][:5])

    st.subheader("Frequent itemsets by size k")
    for k, g in ae.split_itemsets_by_length(res["freq"]).items():
        with st.expander(f"{k}-itemsets ({len(g)})"):
            st.dataframe(ae.itemsets_for_display(g))

    st.subheader("Association rules")
    rules = res["rules"]
    items = sorted({i for t in res["transactions"] for i in t})
    target = st.selectbox("Filter rules by consequent (optional)", ["(all)"] + items)
    if target != "(all)":
        rules = ae.filter_rules(rules, consequent_contains=target)
    st.dataframe(ae.rules_for_display(rules).drop(columns=["rule"], errors="ignore"))
    if len(rules):
        st.info(ae.explain_rule(rules.iloc[0]))

# ---- 5. Metric definitions, always visible ----
st.divider()
st.header("What do the metrics mean?")
for key in ["support", "confidence", "lift", "leverage", "conviction"]:
    info = ae.get_metric_info(key)
    with st.expander(info["name"]):
        st.write(info["definition"])
        st.latex(info["latex"])
        st.caption(f"Range: {info['range']}")
        st.write(info["interpretation"])
