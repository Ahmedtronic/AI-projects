import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
n_items = 10
n_trials = 10000

n_ones = [0] * n_items # Number of rewards of each item
n_zeros = [0] * n_items # Number of punishments of each item
total_rewards = 0 # Total gained rewards
items_selected = [] # a History contains the selected item at each N_trial

for trial in range(n_trials):
    max_probability = 0
    # Randomly select first item and try it
    selected_item = 0
    for item in range(n_items):
        # Calculate Beta distribution
        random_beta = random.betavariate(n_ones[item] + 1, n_zeros[item] + 1)
        if random_beta > max_probability :
            # Update to higher probability & keep selected item index
            max_probability = random_beta
            selected_item = item
    # Update selected item's data
    reward = dataset.values[trial, selected_item]
    total_rewards += reward
    # Keep a history of which item is selected
    items_selected.append(selected_item + 1) # +1 because selected_item is index
    # Add a reward(1) and punishment(0) to selected item
    if reward:
        n_ones[selected_item] += 1
    else:
        n_zeros[selected_item] += 1
    
# Visualization
plt.hist(items_selected)

# Most selected item (as value, not as index)
top_item = np.bincount(np.array(items_selected)).argmax()