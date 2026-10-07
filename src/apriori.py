import pandas as pd
from itertools import combinations

# load_transactions()
def load_transactions(file_path):
    """Read transactions.csv and convert the items column into sets."""
    df = pd.read_csv(file_path)

    transactions = []

    for item_string in df["items"]:
        items = set(item_string.split(","))
        transactions.append(items)

    return transactions

# generate_C1()
def generate_C1(transactions):
    """Generate candidate 1-itemsets from all transactions."""
    items = set()

    for transaction in transactions:
        for item in transaction:
            items.add(item)

    C1 = [frozenset([item]) for item in sorted(items)]

    return C1

# calculate_support()
def calculate_support(candidate, transactions):
    """Calculate the support of a candidate itemset."""
    count = 0

    for transaction in transactions:
        if candidate.issubset(transaction):
            count += 1

    support = count / len(transactions)

    return support

# prune_itemsets()
def prune_itemsets(candidates, transactions, min_support):
    """Keep only candidates whose support meets minimum support."""
    frequent_itemsets = {}

    for candidate in candidates:
        support = calculate_support(candidate, transactions)

        if support >= min_support:
            frequent_itemsets[candidate] = support

    return frequent_itemsets

# generate_candidates()
def generate_candidates(previous_frequent_itemsets, k):
    """Generate candidate k-itemsets using the Apriori property."""
    previous_itemsets = list(previous_frequent_itemsets.keys())
    previous_itemsets_set = set(previous_itemsets)

    candidates = set()

    for i in range(len(previous_itemsets)):
        for j in range(i + 1, len(previous_itemsets)):
            union = previous_itemsets[i] | previous_itemsets[j]

            if len(union) != k:
                continue

            # Check the Apriori property:
            # every (k-1)-subset must be frequent
            all_subsets_frequent = True

            for subset in combinations(union, k - 1):
                subset = frozenset(subset)

                if subset not in previous_itemsets_set:
                    all_subsets_frequent = False
                    break

            if all_subsets_frequent:
                candidates.add(union)

    return sorted(candidates, key=lambda x: sorted(x))

# apriori()
def apriori(transactions, min_support=0.05):
    """Run the complete Apriori algorithm."""

    all_frequent_itemsets = {}

    # Step 1: Generate C1 and L1
    Ck = generate_C1(transactions)
    Lk = prune_itemsets(Ck, transactions, min_support)

    all_frequent_itemsets.update(Lk)

    print("\nC1:", len(Ck))
    print("L1:", len(Lk))

    # Start generating 2-itemsets
    k = 2

    while Lk:

        # Generate Ck using previous Lk
        Ck = generate_candidates(Lk, k)

        if not Ck:
            break

        print(f"\nC{k}: {len(Ck)} candidates")

        # Calculate support and prune
        Lk = prune_itemsets(Ck, transactions, min_support)

        print(f"L{k}: {len(Lk)} frequent itemsets")

        # Store frequent itemsets
        all_frequent_itemsets.update(Lk)

        k += 1

    return all_frequent_itemsets

# save_frequent_itemsets()
def save_frequent_itemsets(frequent_itemsets, output_path):
    """Save all frequent itemsets and their support values to CSV."""

    rows = []

    for itemset, support in frequent_itemsets.items():
        rows.append({
            "itemset": ", ".join(sorted(itemset)),
            "support": support
        })

    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False)

    print(f"\nFrequent itemsets saved to: {output_path}")



if __name__ == "__main__":
    transactions = load_transactions("data/transactions.csv")

    C1 = generate_C1(transactions)

    print("Number of transactions:", len(transactions))
    print("Number of candidate 1-itemsets:", len(C1))

    print("\nC1:")
    for itemset in C1:
        print(itemset)

    min_support = 0.05

    L1 = prune_itemsets(C1, transactions, min_support)

    print("\nMinimum support:", min_support)
    print("Number of frequent 1-itemsets:", len(L1))

    print("\nL1:")
    for itemset, support in L1.items():
        print(itemset, "Support:", round(support, 4))

    C2 = generate_candidates(L1, 2)

    print("\nNumber of candidate 2-itemsets:", len(C2))

    print("\nFirst 10 C2 candidates:")
    for itemset in C2[:10]:
        print(itemset)

    L2 = prune_itemsets(C2, transactions, min_support)

    print("\nNumber of frequent 2-itemsets:", len(L2))

    print("\nL2:")
    for itemset, support in L2.items():
        print(itemset, "Support:", round(support, 4))

print("\n==============================")
print("Running complete Apriori")
print("==============================")

frequent_itemsets = apriori(transactions, min_support)

print("\nTotal frequent itemsets:", len(frequent_itemsets))

save_frequent_itemsets(
    frequent_itemsets,
    "results/frequent_itemsets.csv"
)