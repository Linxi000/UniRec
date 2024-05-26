import pandas as pd
import numpy as np
import pickle

data_file = 'db_ml1m_mapped_internal-ids_interactions_10core-users_5core-items.csv'
data = pd.read_csv(data_file)

grouped_data = data.sort_values(['item', 'timestamp']).groupby('item')

variance_dict = {}

for item, group in grouped_data:
    time_intervals = group['timestamp'].diff().dropna()

    variance = time_intervals.var()

    if pd.isna(variance):
        variance = 0

    variance_dict[item] = variance

sorted_items = sorted(variance_dict.items(), key=lambda x: x[1])
threshold = int(0.3 * len(sorted_items))
uniformity_dict = {item: (i < threshold) for i, (item, _) in enumerate(sorted_items)}

with open('db_ml1m_item_head_tail.pkl', 'wb') as file:
    pickle.dump(uniformity_dict, file)

max_variance = max(variance_dict.values())
min_variance = min(variance_dict.values())

normalized_variance_dict = {}
for item, variance in variance_dict.items():
    if max_variance - min_variance > 0:  
        normalized_score = ((variance - min_variance) / (max_variance - min_variance)) * 100
    else:
        normalized_score = 0  
    normalized_variance_dict[item] = normalized_score

with open('db_ml1m_item_var.pkl', 'wb') as file:
    pickle.dump(normalized_variance_dict, file)

with open('db_ml1m_item_uni.pkl', 'wb') as file:
    pickle.dump(normalized_variance_dict, file)


