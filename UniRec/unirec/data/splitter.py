import json
import os
def split_data(sorted_user_interactions):

    train_set = {}
    valid_set = {}
    test_set = {}
    for uid, interactions in sorted_user_interactions.items():
        nfeedback = len(interactions)
        if nfeedback < 3:
            train_set[uid] = interactions
            valid_set[uid] = []
            test_set[uid] = []
        else:
            train_set[uid] = interactions[:-2]
            valid_set[uid] = [interactions[-2]]
            test_set[uid] = [interactions[-1]]

    return train_set, valid_set, test_set
