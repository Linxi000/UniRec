import random
import numpy as np
import pickle


from unirec.utils import random_neg


def train_sample(uid, nxt_idx, dataset, seqlen, n_items,
                 **kwargs):
    """
    Sampling train data for a given user
    :param uid: user id
    :param nxt_idx: next interaction index
    :param dataset: dataset
    :param seqlen: sequence length
    :param n_items: number of items
    :param kwargs: additional parameters
    :return:
    """
    seq = np.zeros([seqlen], dtype=np.int32)
    in_ts_seq = np.zeros([seqlen], dtype=np.int32)
    nxt_ts_seq = np.zeros([seqlen], dtype=np.int32)
    pos = np.zeros([seqlen], dtype=np.int32)
    neg = np.zeros([seqlen], dtype=np.int32)
    uneven_uniform_seq = np.zeros([seqlen], dtype=np.int32)
    uneven_uniform_ts_seq = np.zeros([seqlen], dtype=np.int32)

    item_head_tail_values = np.zeros([seqlen], dtype=np.bool_)
    item_neighbors_sequences = np.zeros([seqlen, 3], dtype=np.int32)  
    item_var_values = np.zeros([seqlen], dtype=np.float32)

    nxt = dataset[uid][nxt_idx][0]
    nxt_time = dataset[uid][nxt_idx][1]

    activate_sse = kwargs['activate_sse']
    sse_type = kwargs['sse_type']
    item_uni=kwargs['item_uni']
    user_uni=kwargs['user_uni']
    item_head_tail=kwargs['item_head_tail']
    item_neighbors=kwargs['item_neighbors']
    item_var=kwargs['item_var']

    threshold_item = kwargs['threshold_item'] \
        if 'threshold_item' in kwargs else 1.
    threshold_favs = kwargs['threshold_favs'] \
        if 'threshold_favs' in kwargs else 1.
    threshold_user = kwargs['threshold_user'] \
        if 'threshold_user' in kwargs else 1.
    idx = seqlen - 1
    uneven_idx = seqlen - 1
    # list of historic items
    favs = set(map(lambda x: x[0], dataset[uid]))
    favs_list = list(favs)
    flag=0
    for interaction in reversed(dataset[uid][:nxt_idx]):
        iid, ts = interaction
        flag+=1
        if activate_sse is True:
            if sse_type == 'uniform':
                if random.random() > threshold_item:
                    iid = np.random.randint(1, n_items + 1)
                    nxt = np.random.randint(1, n_items + 1)
            else:
                p_favs = random.random()
                if p_favs > threshold_favs:
                    iid = np.random.choice(favs_list)
                    nxt = np.random.choice(favs_list)
                elif random.random() > threshold_item:
                    iid = np.random.randint(1, n_items + 1)
                    nxt = np.random.randint(1, n_items + 1)

        seq[idx] = iid
        in_ts_seq[idx] = ts
        pos[idx] = nxt
        nxt_ts_seq[idx] = nxt_time
        #itembrach
        item_head_tail_values[idx] = item_head_tail.get(iid)
        item_var_values[idx] = item_var.get(iid,100)
        neighbors = item_neighbors.get(iid, [iid] * 5)       
        selected_neighbors = random.sample(neighbors, min(len(neighbors), 3))
        item_neighbors_sequences[idx, :len(selected_neighbors)] = selected_neighbors

        if nxt != 0:
            neg[idx] = random_neg(1, n_items + 1, favs)
        if flag<3:
            uneven_uniform_seq[uneven_idx] = iid
            uneven_uniform_ts_seq[uneven_idx] = ts
            uneven_idx -= 1
        else:
            if item_uni.get(iid):
                uneven_uniform_seq[uneven_idx] = iid
                uneven_uniform_ts_seq[uneven_idx] = ts
                uneven_idx -= 1
        nxt = iid
        nxt_time = ts
        idx -= 1
        if idx == -1:
            break
    if random.random() > threshold_user:
        uid = np.random.randint(1, kwargs['n_users'] + 1)
    out = uid, seq, pos, neg
    if 'time_dict' in kwargs and kwargs['time_dict'] is not None:
        seq_year = np.zeros([seqlen], dtype=np.int32)
        seq_month = np.zeros([seqlen], dtype=np.int32)
        seq_day = np.zeros([seqlen], dtype=np.int32)
        seq_dayofweek = np.zeros([seqlen], dtype=np.int32)
        seq_dayofyear = np.zeros([seqlen], dtype=np.int32)
        seq_week = np.zeros([seqlen], dtype=np.int32)
        seq_hour = np.zeros([seqlen], dtype=np.int32)
        nxt_year = np.zeros([seqlen], dtype=np.int32)
        nxt_month = np.zeros([seqlen], dtype=np.int32)
        nxt_day = np.zeros([seqlen], dtype=np.int32)
        nxt_dayofweek = np.zeros([seqlen], dtype=np.int32)
        nxt_dayofyear = np.zeros([seqlen], dtype=np.int32)
        nxt_week = np.zeros([seqlen], dtype=np.int32)
        nxt_hour = np.zeros([seqlen], dtype=np.int32)
        un_year = np.zeros([seqlen], dtype=np.int32)
        un_month = np.zeros([seqlen], dtype=np.int32)
        un_day = np.zeros([seqlen], dtype=np.int32)
        un_dayofweek = np.zeros([seqlen], dtype=np.int32)
        un_dayofyear = np.zeros([seqlen], dtype=np.int32)
        un_week = np.zeros([seqlen], dtype=np.int32)
        un_hour = np.zeros([seqlen], dtype=np.int32)
        for i, ts in enumerate(in_ts_seq):
            if ts > 0:
                seq_year[i], seq_month[i], seq_day[i], seq_dayofweek[i], \
                    seq_dayofyear[i], seq_week[i], seq_hour[i] = kwargs['time_dict'][ts]
        for i, ts in enumerate(nxt_ts_seq):
            if ts > 0:
                nxt_year[i], nxt_month[i], nxt_day[i], nxt_dayofweek[i], \
                    nxt_dayofyear[i], nxt_week[i], nxt_hour[i] = kwargs['time_dict'][ts]
        for i, ts in enumerate(uneven_uniform_ts_seq):
            if ts > 0:
                un_year[i], un_month[i], un_day[i], un_dayofweek[i], \
                    un_dayofyear[i], un_week[i], un_hour[i] = kwargs['time_dict'][ts]
        out = out + (seq_year, seq_month, seq_day,
                     seq_dayofweek, seq_dayofyear, seq_week, seq_hour,
                     nxt_year, nxt_month, nxt_day,
                     nxt_dayofweek, nxt_dayofyear, nxt_week, nxt_hour)
        out = out + ()
    if 'item_fism' in kwargs:
        item_fism_seq = kwargs['item_fism'][uid]
        if len(item_fism_seq) < kwargs['n_fism_items']:
            zeros = [0] * (kwargs['n_fism_items'] - len(item_fism_seq))
            item_fism_seq = item_fism_seq + tuple(zeros)
        out = out + (item_fism_seq,)

    in_ts_intervals = calculate_intervals(in_ts_seq)
    nxt_ts_intervals = calculate_intervals(nxt_ts_seq)
    out += (in_ts_intervals, nxt_ts_intervals,)

    new_seq = np.copy(seq)
    new_new_seq = np.copy(seq)
    if np.max(seq) > np.min(seq):
        min_interval_index=find_min_interval_index(in_ts_intervals)
        if min_interval_index<0:
            random_index = np.random.randint(0, seqlen - 1) 
            while seq[random_index] == 0:
                random_index = np.random.randint(0, seqlen - 1)
            new_value = (kwargs['item_similarity'].get(seq[random_index],seq[random_index]))
            new_seq[random_index] = new_value
            new_new_seq[random_index], new_new_seq[random_index + 1] = new_new_seq[random_index + 1], new_new_seq[
                random_index]
        else:
            new_seq[min_interval_index]= (kwargs['item_similarity'].get(seq[min_interval_index],seq[min_interval_index]))
            if min_interval_index+1>=seqlen-1:
                new_new_seq[min_interval_index], new_new_seq[min_interval_index - 1] = \
                new_new_seq[min_interval_index - 1], new_new_seq[min_interval_index]
            else:
                new_new_seq[min_interval_index], new_new_seq[min_interval_index + 1] = \
                    new_new_seq[min_interval_index + 1], new_new_seq[min_interval_index]

        out = out + (new_seq, new_new_seq)
    else:
        out = out + (seq,seq)
    uneven_index= 49-uneven_idx

    # userbranch
    u_value = user_uni.get(uid, False)
    out = out + (u_value, uneven_uniform_seq,)
    out = out + (un_year, un_month, un_day,un_dayofweek, un_dayofyear, un_week, un_hour)
    out = out + (uneven_index,)
    out = out + (item_head_tail_values, item_neighbors_sequences,item_var_values,)

    return out


def test_sample(uid, dataset, seqlen, n_items, **kwargs):
    """
    Sampling test data for a given user
    :param uid:
    :param dataset:
    :param seqlen:
    :param n_items:
    :param kwargs:
    :return:
    """
    train_set = kwargs['train_set']
    num_negatives = kwargs['num_test_negatives']
    item_head_tail=kwargs['item_head_tail']
    item_neighbors=kwargs['item_neighbors']
    item_head_tail_values = np.zeros([seqlen], dtype=np.bool_)
    item_neighbors_sequences = np.zeros([seqlen, 3], dtype=np.int32)  
    test_item_neighbors = np.zeros([seqlen, 3], dtype=np.int32) 

    seq = np.zeros([seqlen], dtype=np.int32)
    ts_seq = np.zeros([seqlen], dtype=np.int32)


    idx = seqlen - 1
    for interaction in reversed(train_set[uid]):
        seq[idx] = interaction[0]
        ts_seq[idx] = interaction[1]

        item_head_tail_values[idx] = item_head_tail.get(interaction[0], False)

        neighbors = item_neighbors.get(interaction[0], [interaction[0]] * 5)

        selected_neighbors = random.sample(neighbors, min(len(neighbors), 3))
        item_neighbors_sequences[idx, :len(selected_neighbors)] = selected_neighbors

        idx -= 1
        if idx == -1:
            break
    rated = set([x[0] for x in train_set[uid]])
    rated.add(dataset[uid][0][0])
    rated.add(0)

    test_item_ids = [dataset[uid][0][0]]
    test_ts_seq = [dataset[uid][0][1]]
    test_ts_timestamp = dataset[uid][0][1]

    neg_sampling = kwargs['neg_sampling']
    if neg_sampling == 'uniform':
        for _ in range(num_negatives):
            t = np.random.randint(1, n_items + 1)
            while t in rated:
                t = np.random.randint(1, n_items + 1)
            test_item_ids.append(t)
            test_ts_seq.append(test_ts_timestamp)
    else:
        zeros = np.array(list(rated)) - 1
        p = kwargs['train_item_popularities'].copy()
        p[zeros] = 0.0
        p = p / p.sum()
        neg_item_ids = np.random.choice(range(1, n_items + 1),
                                        size=num_negatives,
                                        p=p,
                                        replace=False)
        test_item_ids = test_item_ids + neg_item_ids.tolist()
        test_ts_seq.extend(test_ts_timestamp * num_negatives)

    out = uid, seq, test_item_ids
    if 'time_dict' in kwargs and kwargs['time_dict'] is not None:
        test_ts = dataset[uid][0][1]
        seq_year = np.zeros([seqlen], dtype=np.int32)
        seq_month = np.zeros([seqlen], dtype=np.int32)
        seq_day = np.zeros([seqlen], dtype=np.int32)
        seq_dayofweek = np.zeros([seqlen], dtype=np.int32)
        seq_dayofyear = np.zeros([seqlen], dtype=np.int32)
        seq_week = np.zeros([seqlen], dtype=np.int32)
        seq_hour = np.zeros([seqlen], dtype=np.int32)
        for i, ts in enumerate(ts_seq):
            if ts > 0:
                seq_year[i], seq_month[i], seq_day[i], seq_dayofweek[i], \
                    seq_dayofyear[i], seq_week[i], seq_hour[i] = kwargs['time_dict'][ts]
        test_year, test_month, test_day, test_dayofweek, \
            test_dayofyear, test_week, test_hour = kwargs['time_dict'][test_ts]
        out = out + (seq_year, seq_month, seq_day,
                     seq_dayofweek, seq_dayofyear, seq_week, seq_hour,
                     test_year, test_month, test_day,
                     test_dayofweek, test_dayofyear, test_week, test_hour)
    if 'item_fism' in kwargs:
        item_fism_seq = kwargs['item_fism'][uid]
        if len(item_fism_seq) < kwargs['n_fism_items']:
            zeros = [0] * (kwargs['n_fism_items'] - len(item_fism_seq))
            item_fism_seq = item_fism_seq + tuple(zeros)
        out = out + (item_fism_seq,)

    ts_intervals = calculate_intervals(ts_seq)

    if len(ts_seq) > 0 and ts_seq[-1] > 0:
        last_interval = (test_ts_timestamp - ts_seq[-1]) // 86400
        last_interval = min(last_interval, 365)  # Cap at 365 days
    else:
        last_interval = 0

    out = out + (ts_intervals, last_interval,)
    out = out + (item_head_tail_values, item_neighbors_sequences,)
    

    return out

def calculate_intervals(ts_seq):
    days_seq = ts_seq // 3600

    intervals = np.diff(days_seq, prepend=days_seq[0])
   
    intervals = np.clip(intervals, 0, 8760)
    return intervals

def find_min_interval_index(ts_seq):
    if len(ts_seq) < 2:
        return -1
    intervals = np.diff(ts_seq, prepend=ts_seq[0])
    
    if not np.any(intervals > 0):
        return -1 

    positive_intervals = intervals[intervals > 0]
    min_interval = np.min(positive_intervals)
    min_interval_index=np.where(intervals == min_interval)[0][0]
    return min_interval_index

def find_most_similar_item(item_id, similarity_dict):
    return similarity_dict.get(item_id, item_id)
