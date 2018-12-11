import numpy as np
import scipy.sparse as sps
import sklearn.utils.extmath as skl
import time as time

from .recommender import Recommender

class PureSVD(Recommender):

    def fit(self, URM):
        
        self.u, self.sigma, self.v = skl.randomized_svd(URM, n_components = 100)
        self.s_Vt = sps.diags(self.sigma) * self.v

    def recommend_all(self, URM, test_set, at=10):
        
        submission = []
        time_start = time_curr = time.time() 
        
        for user in test_set:
            
            #time
            if(time.time() - time_curr > 120):
                print("Time spent:\t {}sec Percentage of users seen:\t {:2.0f}%".format(\
                        (time.time() - time_start),(100 * (user / len(test_set)))))
                time_curr = time.time()
            
            # calculate ranks
            rank = self.u[user,:].dot(self.s_Vt)

            # put zero the items of users
            np.put(rank, URM[user].indices, 0)

            # sort by ratings and get the top ten items to recommend
            score = np.argsort(rank)[::-1][:10]
            submission.append([user, score])
        
        return submission

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
