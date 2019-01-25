import numpy as np
import scipy.sparse as sps

def train_test_holdout(URM_matrix, train_perc):

    URM_all = sps.coo_matrix(URM_matrix)
    numInteractions = URM_all.nnz

    train_mask = np.random.choice([True,False], numInteractions, p=[train_perc, 1-train_perc])
    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
    URM_test = URM_test.tocsr()
    return URM_train, URM_test

def split_sequential(sequential_list, URM_test):

    mask = np.array([])
    user_to_test = np.unique(sequential_list[:,0])

    for user_id in user_to_test:
        item_test = URM_test[user_id].indices
        items_seq = sequential_list[sequential_list[:,0] == user_id][:,1]
        mask = np.append(mask, np.isin(items_seq, item_test))
    
    sequential_list_test = sequential_list[np.array(mask, dtype=bool)]
    sequential_list_train = sequential_list[np.logical_not(np.array(mask, dtype=bool))]
    return sequential_list_train, sequential_list_test
