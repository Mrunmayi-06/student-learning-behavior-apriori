"""Simple UI.  Run:  streamlit run app.py"""
import streamlit as st

from src import analysis as an

st.set_page_config(page_title="Student Behaviour Apriori", layout="wide")
st.title("Student Learning Behaviour -> Academic Outcome (Apriori)")

# ---------------- Controls ----------------
available = {label: path for label, (_, path) in an.PERIODS.items() if path.exists()}
if not available:
    st.error("No transactions file found in data/. Run src/preprocessing.py first.")
    st.stop()

dataset = st.sidebar.selectbox("Dataset (observation period)", list(available))
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.50, 0.10, 0.01)
min_conf = st.sidebar.slider("Minimum Confidence", 0.10, 1.00, 0.60, 0.05)
outcome = st.sidebar.selectbox("Outcome Filter", ["All"] + an.OUTCOMES)
run = st.sidebar.button("Run Apriori")


@st.cache_data(show_spinner=False)
def get_transactions(path):
    return an.load_transactions(path)


# ---------------- Run ----------------
if run:
    tx = get_transactions(str(available[dataset]))
    with st.spinner("Running Apriori and generating rules..."):
        res = an.mine(tx, min_support, min_conf)
    rules = an.sort_rules(an.filter_by_outcome(res["rules"], outcome))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", len(tx))
    c2.metric("Candidate itemsets", res["candidates"])
    c3.metric("Frequent itemsets", len(res["freq"]))
    c4.metric("Rules shown", len(rules))
    c5.metric("Runtime (s)", f"{res['runtime']:.2f}")

    base = an.outcome_baseline(tx)
    st.caption("Baseline outcome shares (lift is measured against these): "
               + ", ".join(f"{k} {v:.1%}" for k, v in base.items()))

    st.subheader("Frequent Itemsets")
    st.dataframe(an.freq_to_df(res["freq"]))

    st.subheader("Association Rules (Support / Confidence / Lift)")
    if rules.empty:
        st.warning("No rules for these settings. Lower the minimum support or confidence.")
    else:
        st.dataframe(rules)
else:
    st.info("Choose settings in the sidebar and click **Run Apriori**.")
st.divider()
st.header("Saved graphs (created by main.py)")
pngs = sorted(an.RESULTS_DIR.glob("*.png"))
if not pngs:
    st.info("No graphs yet. Run `python main.py` first.")
for p in pngs:
    st.image(str(p), caption=p.stem.replace("_", " "), width=550)
cat >> app.py <<'EOF'

st.divider()
st.header("Threshold experiment (created by main.py)")
_exp = an.RESULTS_DIR / "threshold_experiment.csv"
if _exp.exists():
    import pandas as pd
    st.dataframe(pd.read_csv(_exp))
else:
    st.info("Run `python main.py` first.")

st.header("Graphs (created by main.py)")
_pngs = sorted(an.RESULTS_DIR.glob("*.png"))
if not _pngs:
    st.info("No graphs yet. Run `python main.py` first.")
for _p in _pngs:
    st.image(str(_p), caption=_p.stem.replace("_", " "), width=550)
EOF
