def build_transaction_index(transactions):
    """
    Creates an index:
    item -> set of transaction IDs containing that item
    """

    index = {}

    for transaction_id, transaction in enumerate(transactions):

        for item in transaction:

            if item not in index:
                index[item] = set()

            index[item].add(transaction_id)

    return index


def calculate_itemset_count(itemset, transaction_index):

    itemset = list(itemset)

    if len(itemset) == 0:
        return 0

    if itemset[0] not in transaction_index:
        return 0

    transaction_ids = set(
        transaction_index[itemset[0]]
    )

    for item in itemset[1:]:

        if item not in transaction_index:
            return 0

        transaction_ids.intersection_update(
            transaction_index[item]
        )

        if len(transaction_ids) == 0:
            return 0

    return len(transaction_ids)


def calculate_support(
    itemset,
    transaction_index,
    total_transactions
):

    if total_transactions == 0:
        return 0.0

    count = calculate_itemset_count(
        itemset,
        transaction_index
    )

    return count / total_transactions


def calculate_confidence(
    antecedent,
    consequent,
    transaction_index,
    total_transactions
):

    antecedent_count = calculate_itemset_count(
        antecedent,
        transaction_index
    )

    if antecedent_count == 0:
        return 0.0

    complete_itemset = list(antecedent) + list(consequent)

    complete_count = calculate_itemset_count(
        complete_itemset,
        transaction_index
    )

    return complete_count / antecedent_count


def calculate_lift(
    confidence_value,
    consequent,
    transaction_index,
    total_transactions
):

    consequent_support = calculate_support(
        consequent,
        transaction_index,
        total_transactions
    )

    if consequent_support == 0:
        return 0.0

    return confidence_value / consequent_support