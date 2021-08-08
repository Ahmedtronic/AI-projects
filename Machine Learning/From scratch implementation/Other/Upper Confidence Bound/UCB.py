import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
n_items = 10
n_trials = 10000

sum_reward = [0] * n_items # Number of rewards of each item
sum_selection = [0] * n_items # Number of selections of each item
total_rewards = 0 # Total gained rewards
items_selected = [] # a History contains the selected item at each N_trial

for trial in range(n_trials):
    max_UCB = 0
    # Randomly select first item and try it
    selected_item = 0
    for item in range(n_items):
        # If current item has been choosen before -> calculate upper bound
        if sum_selection[item] > 0 :
            average_reward = sum_reward[item] / sum_selection[item]
            delta = math.sqrt( 2 * (math.log(trial + 1) / sum_selection[item]  ) )
            upper_bound = average_reward + delta
        # If current item has never been choosen -> 
        # use a big number to break the later condition to select that item
        else :
            upper_bound = 10e100
        if upper_bound > max_UCB :
            # Update max upper bound & keep selected item index
            max_UCB = upper_bound
            selected_item = item
    # Update selected item's data
    # Add a reward(1) or punishment(0) to selected item
    reward = dataset.values[trial, selected_item]
    sum_reward[selected_item] += reward
    total_rewards += reward
    # Add 1 to selected times of that item
    sum_selection[selected_item] += 1
    # Keep a history of which item is selected
    items_selected.append(selected_item + 1) # +1 because selected_item is index
    
# Visualization
plt.hist(items_selected)

# Most selected item (as value, not as index)
top_item = np.bincount(np.array(items_selected)).argmax()