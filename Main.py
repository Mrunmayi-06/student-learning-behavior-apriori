"""
Run the complete project:   python main.py
Also build Weeks 1-8 / Full-course data (needs data/raw/, slow): python main.py --periods
"""
import argparse

from src import analysis as an

SUPPORTS = [0.05, 0.10, 0.15, 0.20]
CONFIDENCES = [0.50, 0.60, 0.70, 0.80]
DEFAULT_SUPPORT, DEFAULT_CONF = 0.10, 0.60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", action="store_true",
                        help="build missing Weeks 1-8 / Full-course transaction files")
    args = parser.parse_args()
    out = an.RESULTS_DIR

    print("1) Loading transactions (Weeks 1-4) ...")
    tx = an.load_transactions(an.PERIODS["Weeks 1-4"][1])
    print(f"   {len(tx)} students; baseline outcome shares:",
          {k: round(v, 3) for k, v in an.outcome_baseline(tx).items()})

    print(f"2) Mining with support={DEFAULT_SUPPORT}, confidence={DEFAULT_CONF} ...")
    res = an.mine(tx, DEFAULT_SUPPORT, DEFAULT_CONF)
    an.freq_to_df(res["freq"]).to_csv(out / "frequent_itemsets.csv", index=False)
    an.sort_rules(res["rules"]).to_csv(out / "all_rules.csv", index=False)
    print(f"   {res['candidates']} candidates, {len(res['freq'])} frequent itemsets, "
          f"{len(res['rules'])} rules, {res['runtime']:.2f}s")
    for o in an.OUTCOMES:
        sub = an.sort_rules(an.filter_by_outcome(res["rules"], o))
        sub.to_csv(out / f"rules_{o.lower()}.csv", index=False)
        print(f"   {o}: {len(sub)} rules")

    print("3) Threshold experiment ...")
    df = an.threshold_experiment(tx, SUPPORTS, CONFIDENCES)
    an.plot_threshold_results(df)
    print(df.to_string(index=False))

    if args.periods:
        print("4) Building period datasets ...")
        an.ensure_period_files()
    print("4) Observation period comparison ...")
    summary = an.period_comparison(DEFAULT_SUPPORT, DEFAULT_CONF)
    an.plot_period_comparison(summary)
    print(summary.to_string(index=False))

    print(f"\nDone. Results saved in {out}")


if __name__ == "__main__":
    main()
