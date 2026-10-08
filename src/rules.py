import csv
import ast
import os
import sys

from itertools import combinations


# =========================================================
# IMPORT METRICS
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from metrics import (
    build_transaction_index,
    calculate_support,
    calculate_confidence,
    calculate_lift
)


# =========================================================
# CONSTANTS
# =========================================================

OUTCOMES = {
    "Result_Pass",
    "Result_Fail",
    "Result_Withdrawn"
}

MIN_CONFIDENCE = 0.50


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

TRANSACTIONS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "transactions.csv"
)

FREQUENT_ITEMSETS_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "frequent_itemsets.csv"
)

RESULTS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results"
)

ASSOCIATION_RULES_FILE = os.path.join(
    RESULTS_FOLDER,
    "association_rules.csv"
)

SUCCESS_RULES_FILE = os.path.join(
    RESULTS_FOLDER,
    "success_rules.csv"
)

RISK_RULES_FILE = os.path.join(
    RESULTS_FOLDER,
    "risk_rules.csv"
)


# =========================================================
# LOAD TRANSACTIONS
# =========================================================

def load_transactions(filename):

    transactions = []

    if not os.path.exists(filename):

        print("ERROR: transactions.csv not found.")
        print(filename)

        return transactions

    with open(
        filename,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.reader(file)

        for row in reader:

            if not row:
                continue

            row = [
                item.strip()
                for item in row
                if item.strip()
            ]

            if not row:
                continue

            # Skip header
            first = row[0].lower()

            if first in [
                "transaction",
                "items",
                "itemset"
            ]:
                continue

            transactions.append(row)

    return transactions


# =========================================================
# PARSE ITEMSET
# =========================================================

def parse_itemset(value):

    value = value.strip()

    # Example:
    # ['Forum_Low', 'Result_Pass']
    try:

        parsed = ast.literal_eval(value)

        if isinstance(
            parsed,
            (list, tuple, set)
        ):

            return [
                str(item).strip()
                for item in parsed
            ]

    except Exception:
        pass

    # Example:
    # Forum_Low|Result_Pass
    if "|" in value:

        return [
            item.strip()
            for item in value.split("|")
            if item.strip()
        ]

    # Example:
    # Forum_Low,Result_Pass
    if "," in value:

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    return [value]


# =========================================================
# LOAD FREQUENT ITEMSETS
# =========================================================

def load_frequent_itemsets(filename):

    itemsets = []

    if not os.path.exists(filename):

        print(
            "ERROR: frequent_itemsets.csv not found."
        )

        print(filename)

        return itemsets

    with open(
        filename,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:

            print(
                "ERROR: CSV has no header."
            )

            return itemsets

        print(
            "Frequent itemset columns:",
            reader.fieldnames
        )

        itemset_column = None

        for column in reader.fieldnames:

            if column.lower().strip() in [
                "itemset",
                "itemsets",
                "items",
                "item"
            ]:

                itemset_column = column

                break

        if itemset_column is None:

            print(
                "ERROR: itemset column not found."
            )

            return itemsets

        for row in reader:

            value = row.get(
                itemset_column,
                ""
            )

            if not value:
                continue

            items = parse_itemset(value)

            if len(items) > 0:

                itemsets.append(items)

    return itemsets


# =========================================================
# GENERATE POSSIBLE RULES
# =========================================================

def generate_rules(itemset):

    rules = []

    n = len(itemset)

    if n < 2:
        return rules

    for size in range(1, n):

        for combination in combinations(
            itemset,
            size
        ):

            antecedent = list(combination)

            consequent = [
                item
                for item in itemset
                if item not in antecedent
            ]

            rules.append(
                (
                    antecedent,
                    consequent
                )
            )

    return rules


# =========================================================
# CHECK OUTCOME
# =========================================================

def get_outcome(consequent):

    for item in consequent:

        if item in OUTCOMES:

            return item

    return None


# =========================================================
# GENERATE ASSOCIATION RULES
# =========================================================

def generate_association_rules(
    frequent_itemsets,
    transactions,
    transaction_index,
    min_confidence
):

    rules = []

    total_transactions = len(
        transactions
    )

    # Cache values so the same support
    # is never calculated repeatedly.
    support_cache = {}
    confidence_cache = {}

    total_itemsets = len(
        frequent_itemsets
    )

    print()
    print(
        "Processing",
        total_itemsets,
        "frequent itemsets..."
    )

    for number, itemset in enumerate(
        frequent_itemsets,
        start=1
    ):

        if len(itemset) < 2:
            continue

        possible_rules = generate_rules(
            itemset
        )

        for antecedent, consequent in possible_rules:

            outcome = get_outcome(
                consequent
            )

            # Only outcome-focused rules
            if outcome is None:
                continue

            # -------------------------------------------------
            # SUPPORT
            # -------------------------------------------------

            full_key = frozenset(
                itemset
            )

            if full_key not in support_cache:

                support_cache[full_key] = calculate_support(
                    itemset,
                    transaction_index,
                    total_transactions
                )

            support_value = support_cache[
                full_key
            ]

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            antecedent_key = frozenset(
                antecedent
            )

            consequent_key = frozenset(
                consequent
            )

            confidence_key = (
                antecedent_key,
                consequent_key
            )

            if confidence_key not in confidence_cache:

                confidence_cache[
                    confidence_key
                ] = calculate_confidence(
                    antecedent,
                    consequent,
                    transaction_index,
                    total_transactions
                )

            confidence_value = confidence_cache[
                confidence_key
            ]

            # -------------------------------------------------
            # CONFIDENCE FILTER
            # -------------------------------------------------

            if confidence_value < min_confidence:
                continue

            # -------------------------------------------------
            # LIFT
            # -------------------------------------------------

            lift_value = calculate_lift(
                confidence_value,
                consequent,
                transaction_index,
                total_transactions
            )

            rule = {
                "antecedent":
                    " + ".join(antecedent),

                "consequent":
                    outcome,

                "support":
                    round(
                        support_value,
                        4
                    ),

                "confidence":
                    round(
                        confidence_value,
                        4
                    ),

                "lift":
                    round(
                        lift_value,
                        4
                    )
            }

            rules.append(rule)

        # Progress
        if number % 50 == 0:

            print(
                "Processed:",
                number,
                "/",
                total_itemsets
            )

    return rules


# =========================================================
# SEPARATE RULES
# =========================================================

def separate_rules(rules):

    success_rules = []
    risk_rules = []

    for rule in rules:

        if rule["consequent"] == "Result_Pass":

            success_rules.append(rule)

        elif rule["consequent"] in [
            "Result_Fail",
            "Result_Withdrawn"
        ]:

            risk_rules.append(rule)

    return (
        success_rules,
        risk_rules
    )


# =========================================================
# SAVE CSV
# =========================================================

def save_rules(
    filename,
    rules
):

    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True
    )

    fieldnames = [
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for rule in rules:

            writer.writerow(rule)


# =========================================================
# MAIN
# =========================================================

def main():

    

    # -----------------------------------------------------
    # LOAD TRANSACTIONS
    # -----------------------------------------------------

    print()
    print(
        "Loading transactions..."
    )

    transactions = load_transactions(
        TRANSACTIONS_FILE
    )

    print(
        "Transactions loaded:",
        len(transactions)
    )

    if len(transactions) == 0:

        print(
            "ERROR: No transactions found."
        )

        return

    # -----------------------------------------------------
    # LOAD FREQUENT ITEMSETS
    # -----------------------------------------------------

    print()
    print(
        "Loading frequent itemsets..."
    )

    frequent_itemsets = load_frequent_itemsets(
        FREQUENT_ITEMSETS_FILE
    )

    print(
        "Frequent itemsets loaded:",
        len(frequent_itemsets)
    )

    if len(frequent_itemsets) == 0:

        print(
            "ERROR: No frequent itemsets found."
        )

        return

    # -----------------------------------------------------
    # BUILD INDEX
    # -----------------------------------------------------

    print()
    print(
        "Building transaction index..."
    )

    transaction_index = build_transaction_index(
        transactions
    )

    print(
        "Transaction index created."
    )

    # -----------------------------------------------------
    # GENERATE RULES
    # -----------------------------------------------------

    print()
    print(
        "Generating association rules..."
    )

    print(
        "Minimum confidence:",
        MIN_CONFIDENCE
    )

    rules = generate_association_rules(
        frequent_itemsets,
        transactions,
        transaction_index,
        MIN_CONFIDENCE
    )

    # -----------------------------------------------------
    # SEPARATE
    # -----------------------------------------------------

    success_rules, risk_rules = separate_rules(
        rules
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    print()
    print(
        "Saving result files..."
    )

    save_rules(
        ASSOCIATION_RULES_FILE,
        rules
    )

    save_rules(
        SUCCESS_RULES_FILE,
        success_rules
    )

    save_rules(
        RISK_RULES_FILE,
        risk_rules
    )

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

   

   
    print(
        "Total association rules:",
        len(rules)
    )

    print(
        "Pass-associated rules:",
        len(success_rules)
    )

    print(
        "Fail/Withdrawal rules:",
        len(risk_rules)
    )

    print()
    print(
        "Created:"
    )

    print(
        "results/association_rules.csv"
    )

    print(
        "results/success_rules.csv"
    )

    print(
        "results/risk_rules.csv"
    )

    print()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()