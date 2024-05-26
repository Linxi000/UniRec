
from collections import defaultdict
import pandas as pd
import numpy as np
import csv
import math
import pickle  

def load_train_data(data_file):
    train_data = defaultdict(lambda: defaultdict(list))
    with open(data_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            userid = row['user']
            itemid = row['item']
            timestamp = int(row['timestamp'])  
            train_data[userid][itemid].append(timestamp)
    return train_data


def generate_item_similarity(train_data):
    C = dict()
    N = dict()
    for user, items in train_data.items():
        for i in items.keys():
            N.setdefault(i, 0)
            N[i] += 1
            for j in items.keys():
                if i == j:
                    continue
                C.setdefault(i, {})
                C[i].setdefault(j, 0)
                C[i][j] += 1 / math.log(1 + len(items))

    item_sim = dict()
    for i, related_items in C.items():
        item_sim[i] = dict()
        for j, score in related_items.items():
            similarity_score = score / math.sqrt(N[i] * N[j])
            item_sim[i][j] = similarity_score  

    return item_sim

def calculate_time_intervals(train_data_with_time):
    raw_time_intervals = defaultdict(lambda: defaultdict(list))

    for user, items_with_timestamps in train_data_with_time.items():
        for item_i, timestamps_i in items_with_timestamps.items():
            for item_j, timestamps_j in items_with_timestamps.items():
                if item_i == item_j:
                    continue

                for t_i in timestamps_i:
                    for t_j in timestamps_j:
                        interval = abs(t_i - t_j)
                        raw_time_intervals[item_i][item_j].append(interval)

    time_intervals = defaultdict(dict)
    for item_i, related_items in raw_time_intervals.items():
        for item_j, intervals in related_items.items():
            if intervals:
                average_interval = sum(intervals) / len(intervals)
                time_intervals[item_i][item_j] = average_interval
            else:
                time_intervals[item_i][item_j] = float('inf') 

    return time_intervals

def calculate_variance(data_file):
    data = pd.read_csv(data_file)
    grouped_data = data.sort_values(['item', 'timestamp']).groupby('item')
    variance_dict = {}
    for item, group in grouped_data:
        time_intervals = group['timestamp'].diff().dropna()
        variance = time_intervals.var()
        if pd.isna(variance):
            variance = 99999999
        variance_dict[item] = variance
    return variance_dict

def calculate_score(similarity, time_interval, variance):
    normalized_similarity = similarity / (1 + similarity)

    if time_interval == float('inf'):
        time_score = 0 
    else:
        time_score = 1 / (1 + np.log(1 + time_interval)) 
    normalized_variance = 1 / (1 + np.log(1 + variance))
    score = (0.4 * normalized_similarity + 0.8 * time_score + 0.5 * normalized_variance)
    return score


def find_top_similar_items(item_similarity, time_intervals, variance_dict):
    top_similar_items = {}
    for item_i, sim_items in item_similarity.items():
        scores = [(item_j, calculate_score(sim, time_intervals.get(item_i, {}).get(item_j, float('inf')),
                                           variance_dict.get(item_j, 0)))
                  for item_j, sim in sim_items.items()]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
        top_similar_items[item_i] = [item_j for item_j, score in sorted_scores if score > 0]
        if len(top_similar_items[item_i]) < 10:
            top_similar_items[item_i] += [item_i] * (10 - len(top_similar_items[item_i]))
    return top_similar_items


if __name__ == "__main__":
    data_file = 'db_ml1m_mapped_internal-ids_interactions_10core-users_5core-items.csv'
    train_data_with_time = load_train_data(data_file)
    
    item_similarity = generate_item_similarity(train_data_with_time)
    time_intervals = calculate_time_intervals(train_data_with_time)
    variance_dict = calculate_variance(data_file)
    top_similar_items = find_top_similar_items(item_similarity, time_intervals, variance_dict)

    with open('db_ml1m_item_neighbors.pkl', 'wb') as f:
        pickle.dump(top_similar_items, f)

    print("Item neighbors saved to 'db_ml1m_item_neighbors.pkl'")
    with open('db_ml1m_item_neighbors.pkl', 'rb') as f:
        item_neighbors = pickle.load(f)

    i=0
    for item, neighbors in enumerate(item_neighbors.items()):
        print(f"Item ID: {item}, Neighbors: {neighbors}")
        i = i+1
        if i == 4:
            break

