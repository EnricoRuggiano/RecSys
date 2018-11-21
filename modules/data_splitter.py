import numpy as np
import scipy.sparse as sps

def train_test_holdout(URM_matrix, train_perc):

    URM_all = sps.coo_matrix(URM_matrix)
    numInteractions = URM_all.nnz

    train_mask = np.random.choice([True,False], numInteractions, [train_perc, 1-train_perc])
    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
    URM_test = URM_test.tocsr()
    return URM_train, URM_test
