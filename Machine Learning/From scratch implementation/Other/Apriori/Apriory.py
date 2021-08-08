# Importing  libraries
import numpy as np
import pandas as pd
from apyori import apriori


# Get Data & Initialize variables
filename = ""
dataset = pd.read_csv(filename, header = None)
transactions = []
n_transactions = dataset.shape[0]
n_items = 20 # Movies / Supermarket Items / etc ..
n_common_items = 3 # How many Top items occurrences considered | playing around is recommended
min_support = round((n_common_items) / n_transactions, 3) # min support = n_items occured / n_transactions
min_confidence = 0.2 # Arbitrary .. may be 50% , 80% , try different values
min_lift = 3 # How strong the rules are (likelihood) (lift > 1 is good likelihood)

# Preprocessing data (make list of lists(transactions) of items)
for trans in range(0, n_transactions):
    transactions.append([ str(dataset.values[trans, item]) for item in range(0, n_items)])

# Run algorithm
rules = apriori(transactions, min_support = min_support, min_confidence = min_confidence, min_lift = min_lift, min_length = 2)

# Visualization
results = list(rules)