import numpy as np
import scipy.sparse as sps

def precision(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score

def recall(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score

def MAP(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score

def AP_ten(relevant_items, recommended_items):

    summation = 0.0
    mask = np.arange(10)
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique = True)
    number_of_relevant_items = np.count_nonzero(is_relevant)

    for k in mask:
        if(np.count_nonzero(is_relevant[:k]!= 0)):
            precision_k = precision(is_relevant[:k], relevant_items) * is_relevant[k]
        else:
            precision_k = 0
        summation += precision_k / 10
    return summation    

def evaluate_algorithm(URM_test, recommender_object):

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    cumulative_MAP_10 = 0.0

    num_eval = 0
    URM_test = sps.csr_matrix(URM_test)
    n_users = URM_test.shape[0]

    for user_id in range(n_users):

        if user_id % 10000 == 0:
            print("Evaluated user {} of {}".format(user_id, n_users))

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id+1]

        if end_pos-start_pos>0:

            relevant_items = URM_test.indices[start_pos:end_pos]
            recommended_items = recommender_object.recommend(user_id)
            num_eval+=1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)
            cumulative_MAP_10 += AP_ten(relevant_items, recommended_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    cumulative_MAP_10 /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}, MAP@10 = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP, cumulative_MAP_10))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
        "MAP@10": cumulative_MAP_10
    }

    return result_dict
