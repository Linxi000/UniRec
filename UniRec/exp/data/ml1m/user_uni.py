import pandas as pd
import numpy as np
import pickle

data_file = 'db_ml1m_mapped_internal-ids_interactions_10core-users_5core-items.csv'
data = pd.read_csv(data_file)

grouped_data = data.sort_values('timestamp').groupby('user')

variance_dict = {}

for user, group in grouped_data:

    time_intervals = group['timestamp'].diff().dropna()

    variance = time_intervals.var()

    if pd.isna(variance):
        variance = 0

    variance_dict[user] = variance

sorted_users = sorted(variance_dict.items(), key=lambda x: x[1], reverse=True)
threshold = int(0.4 * len(sorted_users))
uniformity_dict = {user: (i < threshold) for i, (user, _) in enumerate(sorted_users)}

with open('db_ml1m_user_uni.pkl', 'wb') as file:
    pickle.dump(uniformity_dict, file)

