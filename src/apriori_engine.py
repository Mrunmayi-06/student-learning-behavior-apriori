"""
apriori_engine.py
=================
Reusable engine for Association Rule Mining on student learning-behaviour data.

Pipeline implemented here
-------------------------
    tabular DataFrame
        -> (1) choose variables to include
        -> (2) discretise numeric columns (Low / Medium / High, A/B/C, ...)
        -> (2b) optionally club (merge) categorical values
        -> (3) transactions  e.g. ["Attendance=High", "StudyHours=Medium", ...]
        -> (4) one-hot encoding (MLxtend TransactionEncoder)
        -> (5) Apriori  -> frequent itemsets (1-itemsets, 2-itemsets, ..., k-itemsets)
        -> (6) association rules with Support, Confidence, Lift, Leverage, Conviction
        -> (7) helper text / formulas so the UI can explain every metric

The module has NO Streamlit dependency, so it can be imported from app.py,
from main.py, or tested on its own:

    from src.apriori_engine import run_pipeline, BinSpec, get_metric_info

Requires:  pandas, numpy, mlxtend   (pip install mlxtend)
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ============================================================================
# 1. DISCRETISATION  (turning numbers into Low / Medium / High, A / B / C ...)
# ============================================================================

#: Default label sets used when the user only says "split into N bins".
DEFAULT_LABELS: Dict[int, List[str]] = {
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
    4: ["Very Low", "Low", "High", "Very High"],
    5: ["Very Low", "Low", "Medium", "High", "Very High"],
}


@dataclass
class BinSpec:
    """
    Describes HOW one numeric column should be discretised.

    method:
        "quantile"    -> equal-sized groups (each bin has ~the same number of
                         students). Ties are broken by rank, so bins never
                         collapse. Best default for skewed data.
        "equal_width" -> equal-sized value ranges (e.g. 0-33, 33-66, 66-100).
        "custom"      -> you give the cut points in `edges` (e.g. grades).
    n_bins:  number of bins (ignored for "custom", which uses len(edges)-1).
    labels:  names of the bins, lowest to highest. If None, DEFAULT_LABELS is
             used (or "Bin1", "Bin2", ... when n_bins is not 2-5).
    edges:   cut points for method="custom", e.g. [0, 40, 60, 75, 100] with
             labels ["F", "C", "B", "A"]. Values on a boundary go to the
             LOWER bin; the lowest edge is included.
    """
    method: str = "quantile"
    n_bins: int = 3
    labels: Optional[List[str]] = None
    edges: Optional[List[float]] = None


def _resolve_labels(spec: BinSpec) -> List[str]:
    """Pick labels for a BinSpec and validate their count."""
    n = (len(spec.edges) - 1) if spec.method == "custom" else spec.n_bins
    if spec.labels is not None:
        labels = list(spec.labels)
    else:
        labels = DEFAULT_LABELS.get(n, [f"Bin{i + 1}" for i in range(n)])
    if len(labels) != n:
        raise ValueError(f"Expected {n} labels but got {len(labels)}: {labels}")
    return labels


def discretize_numeric(series: pd.Series, spec: Optional[BinSpec] = None) -> pd.Series:
    """
    Convert one numeric Series into categorical labels using a BinSpec.
    Missing values stay missing (they are skipped when transactions are built).
    """
    spec = spec or BinSpec()
    s = pd.to_numeric(series, errors="coerce")

    if spec.method == "custom":
        if not spec.edges or len(spec.edges) < 3:
            raise ValueError("method='custom' needs `edges` with at least 3 values.")
        if sorted(spec.edges) != list(spec.edges):
            raise ValueError("`edges` must be in increasing order.")
        out = pd.cut(s, bins=spec.edges, labels=_resolve_labels(spec),
                     include_lowest=True)
    elif spec.method == "equal_width":
        out = pd.cut(s, bins=spec.n_bins, labels=_resolve_labels(spec))
    elif spec.method == "quantile":
        # rank(method="first") breaks ties, so qcut never fails on repeated values
        out = pd.qcut(s.rank(method="first"), q=spec.n_bins,
                      labels=_resolve_labels(spec))
    else:
        raise ValueError("method must be 'quantile', 'equal_width' or 'custom'.")

    return out.astype("object").where(s.notna())


def club_values(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """
    'Clubbing' of categories: merge several values into one group.
    Example: {"Video": "Digital", "Podcast": "Digital", "Textbook": "Print"}
    Values not listed in `mapping` are kept unchanged.
    """
    return series.map(lambda v: mapping.get(v, v) if pd.notna(v) else v)


def detect_column_types(df: pd.DataFrame, max_unique_for_categorical: int = 10) -> Dict[str, str]:
    """
    Suggest 'numeric' or 'categorical' for every column (useful to pre-fill UI
    widgets). A numeric column with <= max_unique_for_categorical distinct
    values (e.g. a 1-5 rating) is treated as categorical.
    """
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > max_unique_for_categorical:
            types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types


def build_categorical_frame(
    df: pd.DataFrame,
    columns: Sequence[str],
    bin_specs: Optional[Dict[str, BinSpec]] = None,
    default_spec: Optional[BinSpec] = None,
    club_maps: Optional[Dict[str, Dict[str, str]]] = None,
    max_unique_for_categorical: int = 10,
) -> pd.DataFrame:
    """
    Step 1+2: keep ONLY the chosen `columns` and turn each into categories.

    - Column in `bin_specs`            -> discretised with that BinSpec.
    - Other numeric column (many values) -> discretised with `default_spec`
                                           (3 quantile bins: Low/Medium/High).
    - Categorical column               -> kept as text.
    - Column in `club_maps`            -> categories merged with club_values().
    """
    bin_specs = bin_specs or {}
    club_maps = club_maps or {}
    default_spec = default_spec or BinSpec()

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")

    out = pd.DataFrame(index=df.index)
    for col in columns:
        s = df[col]
        if col in bin_specs:
            cat = discretize_numeric(s, bin_specs[col])
        elif pd.api.types.is_numeric_dtype(s) and s.nunique() > max_unique_for_categorical:
            cat = discretize_numeric(s, default_spec)
        else:
            cat = s.astype("object").where(s.isna(), s.astype(str).str.strip())
        if col in club_maps:
            cat = club_values(cat, club_maps[col])
        out[col] = cat
    return out


# ============================================================================
# 2. TRANSACTIONS
# ============================================================================

def to_transactions(cat_df: pd.DataFrame, sep: str = "=") -> List[List[str]]:
    """
    One row (student) -> one transaction (list of 'Column=Value' items).
    Missing values are skipped. Example row ->
        ["Attendance=High", "StudyHours=Medium", "Grade=A"]
    """
    cols = list(cat_df.columns)
    transactions = []
    for row in cat_df.itertuples(index=False, name=None):
        transactions.append([f"{c}{sep}{v}" for c, v in zip(cols, row) if pd.notna(v)])
    return transactions


def transactions_from_items_column(df: pd.DataFrame, column: str = "items",
                                   sep: str = ",") -> List[List[str]]:
    """
    Adapter for the project's existing data/transactions.csv, where each row
    already holds a text such as "Forum_High,Quiz_Low,Result_Pass".
    """
    return [[i.strip() for i in str(v).split(sep) if i.strip()]
            for v in df[column].dropna()]


def encode_transactions(transactions: List[List[str]]) -> pd.DataFrame:
    """List of transactions -> one-hot (True/False) DataFrame for MLxtend."""
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


# ============================================================================
# 3. FREQUENT ITEMSETS + ASSOCIATION RULES  (MLxtend)
# ============================================================================

def mine_frequent_itemsets(onehot: pd.DataFrame, min_support: float,
                           max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Run Apriori. Returns columns: support, itemsets (frozenset), length.
    `length` is k, so you can filter 1-itemsets, 2-itemsets, ... k-itemsets.
    """
    fi = apriori(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
    if fi.empty:
        return pd.DataFrame(columns=["support", "itemsets", "length"])
    fi["length"] = fi["itemsets"].apply(len)
    return fi.sort_values(["length", "support"], ascending=[True, False]).reset_index(drop=True)


def split_itemsets_by_length(freq: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """{1: 1-itemsets, 2: 2-itemsets, ...} - handy for one table/tab per k."""
    return {k: g.sort_values("support", ascending=False).reset_index(drop=True)
            for k, g in freq.groupby("length")}


RULE_COLUMNS = ["antecedents", "consequents", "antecedent support",
                "consequent support", "support", "confidence", "lift",
                "leverage", "conviction"]


def generate_rules(freq: pd.DataFrame, n_transactions: int,
                   min_confidence: float = 0.6,
                   min_lift: Optional[float] = None) -> pd.DataFrame:
    """
    Association rules with Support, Confidence, Lift, Leverage, Conviction.

    Works with old and new MLxtend (>=0.23 needs `num_itemsets`).
    Conviction is `inf` when confidence = 1 (the rule never fails).
    """
    if freq.empty or (freq["length"] < 2).all():
        return pd.DataFrame(columns=RULE_COLUMNS + ["rule"])

    kwargs = dict(metric="confidence", min_threshold=min_confidence)
    if "num_itemsets" in inspect.signature(association_rules).parameters:
        kwargs["num_itemsets"] = n_transactions
    rules = association_rules(freq[["support", "itemsets"]], **kwargs)

    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS + ["rule"])
    if min_lift is not None:
        rules = rules[rules["lift"] >= min_lift]

    rules = rules[RULE_COLUMNS].copy()
    rules["rule"] = rules.apply(
        lambda r: f"{format_itemset(r['antecedents'])}  =>  {format_itemset(r['consequents'])}", axis=1)
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def format_itemset(items: Iterable[str]) -> str:
    """frozenset({'B=1','A=2'}) -> 'A=2, B=1' (sorted, readable)."""
    return ", ".join(sorted(items))


def rules_for_display(rules: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Rules with frozensets converted to text and numbers rounded (UI/CSV)."""
    if rules.empty:
        return rules
    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(format_itemset)
    out["consequents"] = out["consequents"].apply(format_itemset)
    num_cols = out.select_dtypes("number").columns
    out[num_cols] = out[num_cols].round(decimals)
    return out


def itemsets_for_display(freq: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Frequent itemsets with frozensets converted to text (UI/CSV)."""
    if freq.empty:
        return freq
    out = freq.copy()
    out["itemsets"] = out["itemsets"].apply(format_itemset)
    out["support"] = out["support"].round(decimals)
    return out[["itemsets", "length", "support"]]


def filter_rules(rules: pd.DataFrame, consequent_contains: Optional[str] = None,
                 antecedent_contains: Optional[str] = None) -> pd.DataFrame:
    """
    Keep rules that mention an item (exact match) in the consequent and/or
    antecedent. Example: consequent_contains="Grade=A" (outcome filter).
    """
    out = rules
    if consequent_contains:
        out = out[out["consequents"].apply(lambda s: consequent_contains in s)]
    if antecedent_contains:
        out = out[out["antecedents"].apply(lambda s: antecedent_contains in s)]
    return out.reset_index(drop=True)


# ============================================================================
# 4. ONE-CALL PIPELINE (what the UI should call)
# ============================================================================

def run_pipeline(
    df: pd.DataFrame,
    columns: Sequence[str],
    min_support: float = 0.1,
    min_confidence: float = 0.6,
    bin_specs: Optional[Dict[str, BinSpec]] = None,
    default_spec: Optional[BinSpec] = None,
    club_maps: Optional[Dict[str, Dict[str, str]]] = None,
    max_len: Optional[int] = None,
    min_lift: Optional[float] = None,
) -> dict:
    """
    Whole pipeline in one call. Returns a dict with:
        categorical_df, transactions, onehot, freq, rules, stats
    `stats` holds counts and the runtime in seconds.
    """
    start = time.perf_counter()
    cat_df = build_categorical_frame(df, columns, bin_specs, default_spec, club_maps)
    transactions = to_transactions(cat_df)
    onehot = encode_transactions(transactions)
    freq = mine_frequent_itemsets(onehot, min_support, max_len)
    rules = generate_rules(freq, len(transactions), min_confidence, min_lift)
    runtime = time.perf_counter() - start
    return {
        "categorical_df": cat_df,
        "transactions": transactions,
        "onehot": onehot,
        "freq": freq,
        "rules": rules,
        "stats": {
            "transactions": len(transactions),
            "distinct_items": onehot.shape[1],
            "frequent_itemsets": len(freq),
            "rules": len(rules),
            "runtime_sec": round(runtime, 4),
        },
    }


# ============================================================================
# 5. METRIC DEFINITIONS + FORMULAS (for dynamic display in the UI)
# ============================================================================

METRIC_INFO: Dict[str, dict] = {
    "support": {
        "name": "Support",
        "definition": "How often the whole pattern (A and B together) appears "
                      "among ALL students. It tells you how common a pattern is.",
        "formula": "Support(A => B) = count(A and B) / N",
        "latex": r"\text{Support}(A \Rightarrow B)=\frac{\sigma(A \cup B)}{N}",
        "range": "0 to 1 (higher = more common)",
        "interpretation": "Support 0.20 means 20% of all students show both A and B.",
    },
    "confidence": {
        "name": "Confidence",
        "definition": "Among students who have A, the share who also have B. "
                      "It is the conditional probability P(B | A).",
        "formula": "Confidence(A => B) = count(A and B) / count(A) = Support(A and B) / Support(A)",
        "latex": r"\text{Confidence}(A \Rightarrow B)=\frac{\sigma(A \cup B)}{\sigma(A)}=P(B\mid A)",
        "range": "0 to 1 (higher = stronger rule)",
        "interpretation": "Confidence 0.80 means 80% of students with A also have B. "
                          "Warning: it can be high just because B is very common - check lift.",
    },
    "lift": {
        "name": "Lift",
        "definition": "How much more often A and B occur together than expected if "
                      "they were independent. It corrects confidence for how common B is.",
        "formula": "Lift(A => B) = Confidence(A => B) / Support(B)",
        "latex": r"\text{Lift}(A \Rightarrow B)=\frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}",
        "range": "0 to infinity",
        "interpretation": "Lift > 1: positive association (A makes B more likely). "
                          "Lift = 1: independent. Lift < 1: negative association.",
    },
    "leverage": {
        "name": "Leverage",
        "definition": "The difference between how often A and B occur together and "
                      "how often we would expect if they were independent.",
        "formula": "Leverage(A => B) = Support(A and B) - Support(A) x Support(B)",
        "latex": r"\text{Leverage}(A \Rightarrow B)=\text{Supp}(A \cup B)-\text{Supp}(A)\,\text{Supp}(B)",
        "range": "-0.25 to 0.25 (0 = independent)",
        "interpretation": "Leverage 0.05 means 5% more students than expected have both A and B.",
    },
    "conviction": {
        "name": "Conviction",
        "definition": "How much more often the rule would be wrong if A and B were "
                      "independent. High conviction = B almost always follows A.",
        "formula": "Conviction(A => B) = (1 - Support(B)) / (1 - Confidence(A => B))",
        "latex": r"\text{Conviction}(A \Rightarrow B)=\frac{1-\text{Supp}(B)}{1-\text{Conf}(A \Rightarrow B)}",
        "range": "0 to infinity (1 = independent; infinity when confidence = 1)",
        "interpretation": "Conviction 1.5 means the rule is wrong 1.5x less often than it "
                          "would be by chance. Infinity means it is never wrong.",
    },
}


def get_metric_info(metric: str) -> dict:
    """Definition, formula (plain + LaTeX), range and meaning of ONE metric."""
    key = metric.lower().strip()
    if key not in METRIC_INFO:
        raise KeyError(f"Unknown metric '{metric}'. Choose from {list(METRIC_INFO)}")
    return METRIC_INFO[key]


def get_all_metric_info() -> Dict[str, dict]:
    """All metric descriptions (loop over this to build an 'About metrics' page)."""
    return METRIC_INFO


def get_support_info() -> dict:
    return get_metric_info("support")


def get_confidence_info() -> dict:
    return get_metric_info("confidence")


def get_lift_info() -> dict:
    return get_metric_info("lift")


def worked_example(n_total: int, count_a: int, count_b: int, count_ab: int) -> Dict[str, dict]:
    """
    Plug real counts into every formula and show the arithmetic step by step.
    Great for a live demo: the UI can display `steps` text next to the result.

        n_total  = N, all students
        count_a  = students with A (antecedent)
        count_b  = students with B (consequent)
        count_ab = students with both A and B
    """
    if not (0 < n_total and 0 <= count_ab <= min(count_a, count_b) and max(count_a, count_b) <= n_total):
        raise ValueError("Counts are inconsistent (need count_ab <= count_a, count_b <= N).")
    if count_a == 0 or count_b == 0:
        raise ValueError("count_a and count_b must be greater than 0.")

    sup_a, sup_b, sup_ab = count_a / n_total, count_b / n_total, count_ab / n_total
    conf = count_ab / count_a
    lift = conf / sup_b
    lev = sup_ab - sup_a * sup_b
    conv = float("inf") if conf >= 1 else (1 - sup_b) / (1 - conf)

    return {
        "support": {"value": sup_ab,
                    "steps": f"{count_ab} / {n_total} = {sup_ab:.4f}"},
        "confidence": {"value": conf,
                       "steps": f"{count_ab} / {count_a} = {conf:.4f}"},
        "lift": {"value": lift,
                 "steps": f"{conf:.4f} / {sup_b:.4f} = {lift:.4f}"},
        "leverage": {"value": lev,
                     "steps": f"{sup_ab:.4f} - ({sup_a:.4f} x {sup_b:.4f}) = {lev:.4f}"},
        "conviction": {"value": conv,
                       "steps": ("infinity (confidence = 1)" if conv == float("inf")
                                 else f"(1 - {sup_b:.4f}) / (1 - {conf:.4f}) = {conv:.4f}")},
    }


def explain_rule(rule_row: pd.Series) -> str:
    """Plain-English sentence for one rule row (as returned by generate_rules)."""
    a, c = format_itemset(rule_row["antecedents"]), format_itemset(rule_row["consequents"])
    lift = rule_row["lift"]
    if lift > 1.05:
        direction = f"{lift:.2f}x MORE likely than average (positive association)"
    elif lift < 0.95:
        direction = f"only {lift:.2f}x as likely as average (negative association)"
    else:
        direction = "about as likely as average (no real association)"
    return (f"Students with [{a}] also have [{c}] in {rule_row['confidence']:.0%} of cases "
            f"(confidence); this pattern covers {rule_row['support']:.0%} of all students "
            f"(support). Having [{a}] makes [{c}] {direction} (lift). "
            "This shows association, not cause and effect.")


# ============================================================================
# 6. DEMO DATA  (so the module can be tried without any file)
# ============================================================================

def make_demo_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic student table with realistic correlations (for testing/demos)."""
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(3.5, 1.8, n), 0, 10)                  # hours/day
    attend = np.clip(55 + study * 6 + rng.normal(0, 9, n), 20, 100)   # %
    sleep = np.clip(rng.normal(6.8, 1.2, n), 3, 10)                   # hours
    quiz = np.clip(30 + study * 5 + attend * 0.25 + rng.normal(0, 8, n), 0, 100)
    exam = np.clip(20 + study * 4 + attend * 0.3 + sleep * 1.5 + rng.normal(0, 8, n), 0, 100)
    speed = np.clip(2 - study * 0.4 + rng.normal(0, 1.5, n), -3, 7)   # days late (neg = early)
    resource = rng.choice(["Video", "Textbook", "Podcast", "Slides", "Online Notes"], n)
    return pd.DataFrame({
        "StudyHours": study.round(1), "Attendance": attend.round(1),
        "ExamScore": exam.round(1), "QuizScore": quiz.round(1),
        "PreferredResource": resource, "SleepHours": sleep.round(1),
        "SubmissionDaysLate": speed.round(1),
    })


if __name__ == "__main__":
    # Quick self-test:  python apriori_engine.py
    data = make_demo_data()
    res = run_pipeline(
        data,
        columns=["StudyHours", "Attendance", "ExamScore", "PreferredResource"],
        bin_specs={"ExamScore": BinSpec("custom", edges=[0, 40, 55, 70, 100],
                                        labels=["D", "C", "B", "A"])},
        club_maps={"PreferredResource": {"Video": "Digital", "Podcast": "Digital",
                                         "Online Notes": "Digital",
                                         "Textbook": "Print", "Slides": "Print"}},
        min_support=0.10, min_confidence=0.60,
    )
    print(res["stats"])
    print(rules_for_display(res["rules"]).head(5).to_string(index=False))
