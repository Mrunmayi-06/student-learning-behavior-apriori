"""
Integration + analysis layer (Member 4).

Connects:  preprocessing -> transactions -> Apriori -> association rules
and adds: threshold experiments, observation-period comparison, tables, graphs.

It reuses the teammates' functions as they are:
    src/apriori.py       -> load_transactions, apriori
    src/metrics.py       -> build_transaction_index (via rules.py)
    src/rules.py         -> generate_association_rules
    src/preprocessing.py -> build_dataset, create_outputs
"""
import contextlib
import io
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import apriori as ap
from src import preprocessing as prep
from src import rules as rl

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["Pass", "Fail", "Withdrawn"]          # shown to the user
OUTCOME_ITEM = {o: f"Result_{o}" for o in OUTCOMES}  # item names in transactions

# Observation periods: label -> (last day included, transactions file)
# Weeks are 7 days: Weeks 1-4 = days 0-27, Weeks 1-8 = days 0-55.
PERIODS = {
    "Weeks 1-4":   (27,      DATA_DIR / "transactions.csv"),
    "Weeks 1-8":   (55,      DATA_DIR / "transactions_weeks1_8.csv"),
    "Full course": (10**6,   DATA_DIR / "transactions_full_course.csv"),
}


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------
def load_transactions(path):
    """Uses Member 3's loader: returns a list of item sets, one per student."""
    return ap.load_transactions(path)


def outcome_baseline(transactions):
    """Share of students per outcome. Needed to interpret lift."""
    n = len(transactions)
    return {o: sum(OUTCOME_ITEM[o] in t for t in transactions) / n for o in OUTCOMES}


# --------------------------------------------------------------------------
# 2. Mining (Apriori -> frequent itemsets -> rules)
# --------------------------------------------------------------------------
def run_apriori(transactions, min_support):
    """
    Runs Member 3's apriori() and also counts candidate itemsets.
    apriori() only returns the frequent itemsets, so we wrap its two
    candidate-generating helpers to count how many candidates were created.
    Returns (frequent_itemsets_dict, total_candidates, runtime_seconds).
    """
    counter = {"n": 0}
    orig_c1, orig_ck = ap.generate_C1, ap.generate_candidates

    def counting_c1(tx):
        out = orig_c1(tx)
        counter["n"] += len(out)
        return out

    def counting_ck(prev, k):
        out = orig_ck(prev, k)
        counter["n"] += len(out)
        return out

    ap.generate_C1, ap.generate_candidates = counting_c1, counting_ck
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # silence prints
            freq = ap.apriori(transactions, min_support)
    finally:
        ap.generate_C1, ap.generate_candidates = orig_c1, orig_ck
    return freq, counter["n"], time.perf_counter() - start


def run_rules(transactions, freq, min_conf, index=None):
    """
    Runs Member 3's generate_association_rules().
    Returns (rules_dataframe, runtime_seconds).
    Only rules of the form  {behaviour items} -> Result_X  are kept
    (a single outcome item as the consequent).
    """
    if index is None:
        index = rl.build_transaction_index(transactions)
    itemsets = [sorted(i) for i in freq]

    # rules.get_outcome() accepts multi-item consequents but then reports only
    # the outcome name. Restrict it to single-item consequents so the printed
    # rule matches the numbers computed for it.
    orig_get_outcome = rl.get_outcome
    rl.get_outcome = lambda cons: orig_get_outcome(cons) if len(cons) == 1 else None

    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rules = rl.generate_association_rules(itemsets, transactions, index, min_conf)
    finally:
        rl.get_outcome = orig_get_outcome
    runtime = time.perf_counter() - start

    cols = ["antecedent", "consequent", "support", "confidence", "lift"]
    return pd.DataFrame(rules, columns=cols), runtime


def mine(transactions, min_support, min_conf):
    """Full mining run used by main.py and app.py."""
    freq, candidates, t_apriori = run_apriori(transactions, min_support)
    rules, t_rules = run_rules(transactions, freq, min_conf)
    return {"freq": freq, "candidates": candidates, "rules": rules,
            "runtime": t_apriori + t_rules}


# --------------------------------------------------------------------------
# 3. Display helpers
# --------------------------------------------------------------------------
def freq_to_df(freq):
    rows = [{"itemset": ", ".join(sorted(i)), "size": len(i), "support": round(s, 4)}
            for i, s in freq.items()]
    if not rows:
        return pd.DataFrame(columns=["itemset", "size", "support"])
    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def filter_by_outcome(rules, outcome):
    if outcome in (None, "All") or rules.empty:
        return rules
    return rules[rules["consequent"] == OUTCOME_ITEM[outcome]]


def sort_rules(rules):
    if rules.empty:
        return rules
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Experiment 1: support / confidence thresholds
# --------------------------------------------------------------------------
def threshold_experiment(transactions, supports, confidences):
    """
    Apriori depends only on min_support, so it is run once per support value.
    Rule generation is then repeated for every confidence value.
    runtime_sec = Apriori time + rule-generation time for that combination.
    """
    index = rl.build_transaction_index(transactions)
    rows = []
    for s in supports:
        freq, candidates, t_apriori = run_apriori(transactions, s)
        for c in confidences:
            rules, t_rules = run_rules(transactions, freq, c, index)
            rows.append({
                "min_support": s,
                "min_confidence": c,
                "candidate_itemsets": candidates,
                "frequent_itemsets": len(freq),
                "rules": len(rules),
                "pass_rules": int((rules["consequent"] == "Result_Pass").sum()),
                "fail_rules": int((rules["consequent"] == "Result_Fail").sum()),
                "withdrawn_rules": int((rules["consequent"] == "Result_Withdrawn").sum()),
                "runtime_sec": round(t_apriori + t_rules, 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "threshold_experiment.csv", index=False)
    return df


def plot_threshold_results(df):
    metrics = [("candidate_itemsets", "Candidate itemsets"),
               ("frequent_itemsets", "Frequent itemsets"),
               ("rules", "Number of rules"),
               ("runtime_sec", "Runtime (seconds)")]
    for col, label in metrics:
        plt.figure(figsize=(6, 4))
        for conf, g in df.groupby("min_confidence"):
            plt.plot(g["min_support"], g[col], marker="o", label=f"min conf = {conf}")
        plt.xlabel("Minimum support")
        plt.ylabel(label)
        plt.title(f"{label} vs minimum support")
        plt.xticks(sorted(df["min_support"].unique()))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{col}_vs_support.png", dpi=150)
        plt.close()


# --------------------------------------------------------------------------
# 5. Experiment 2: observation periods
# --------------------------------------------------------------------------
def build_period_transactions(end_day, out_csv):
    """
    Re-runs Member 1's preprocessing with a different cut-off day and saves
    only the transactions file. Needs the raw OULAD CSVs in data/raw/.
    (preprocessing.py reads EARLY_END_DAY and DATA_DIR as module globals, so we
    set them temporarily instead of editing the file.)
    """
    old_end, old_dir = prep.EARLY_END_DAY, prep.DATA_DIR
    tmp = Path(tempfile.mkdtemp())
    try:
        prep.EARLY_END_DAY, prep.DATA_DIR = end_day, tmp
        df = prep.build_dataset()
        prep.create_outputs(df)
        shutil.move(str(tmp / "transactions.csv"), str(out_csv))
    finally:
        prep.EARLY_END_DAY, prep.DATA_DIR = old_end, old_dir
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_period_files():
    """Builds any missing period transaction file (slow: reads raw studentVle)."""
    for label, (end_day, path) in PERIODS.items():
        if not path.exists():
            print(f"   building {label} (cut-off day {end_day}) ...")
            build_period_transactions(end_day, path)


def period_comparison(min_support, min_conf):
    rows, top = [], []
    for label, (_, path) in PERIODS.items():
        if not path.exists():
            print(f"   skipping {label}: {path.name} not found")
            continue
        tx = load_transactions(path)
        base = outcome_baseline(tx)
        res = mine(tx, min_support, min_conf)
        for o in OUTCOMES:
            sub = sort_rules(filter_by_outcome(res["rules"], o))
            rows.append({
                "period": label, "outcome": o,
                "baseline_share": round(base[o], 3),
                "rules": len(sub),
                "avg_confidence": round(sub["confidence"].mean(), 3) if len(sub) else 0,
                "max_confidence": round(sub["confidence"].max(), 3) if len(sub) else 0,
                "max_lift": round(sub["lift"].max(), 3) if len(sub) else 0,
            })
            if len(sub):
                best = sub.head(5).copy()
                best.insert(0, "outcome", o)
                best.insert(0, "period", label)
                top.append(best)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "period_comparison.csv", index=False)
    if top:
        pd.concat(top).to_csv(RESULTS_DIR / "top_rules_by_period.csv", index=False)
    return summary


def plot_period_comparison(summary):
    if summary.empty:
        return
    order = [p for p in PERIODS if p in set(summary["period"])]
    for col, label, fname in [("max_lift", "Max lift", "period_max_lift.png"),
                              ("max_confidence", "Max confidence", "period_max_confidence.png")]:
        pivot = summary.pivot(index="period", columns="outcome", values=col).reindex(order)
        pivot.plot(kind="bar", figsize=(7, 4))
        plt.ylabel(label)
        plt.title(f"{label} of rules by observation period")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / fname, dpi=150)
        plt.close()
