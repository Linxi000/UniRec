import csv
import os
import math
import pickle

def convert_data_to_dict(data):
    train_data_dict = {}
    for user, item in data:
        train_data_dict.setdefault(user, {})
        train_data_dict[user][item] = 1 
    return train_data_dict

def load_train_data(data_file):
    train_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            userid = row['user']
            itemid = row['item']
            train_data.append((userid, itemid))
    return convert_data_to_dict(train_data)

def generate_item_similarity(train_data, save_path):
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
        max_similarity = 0
        most_similar_item = None
        for j, score in related_items.items():
            similarity_score = score / math.sqrt(N[i] * N[j])
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                most_similar_item = j
        item_sim[i] = most_similar_item

    with open(save_path, 'wb') as write_file:
        pickle.dump(item_sim, write_file)

    return item_sim


def main():
    data_file = 'db_ml1m_mapped_internal-ids_interactions_10core-users_5core-items.csv'
    similarity_file = 'db_ml1m_item_similarity.pkl'

    train_data_dict = load_train_data(data_file)
    item_similarity = generate_item_similarity(train_data_dict, save_path=similarity_file)
    for item, most_similar in list(item_similarity.items())[:10]:
        print(f"Item {item} most similar to: {most_similar}")


if __name__ == "__main__":
    main()

