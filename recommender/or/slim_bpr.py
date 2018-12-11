from .recommender import Recommender
import modules.Compute_Similarity_Python as sim
#import recommender.Cython.SLIM_BPR_Cython_Epoch as cyt

import time as time
import numpy as np
import scipy.sparse as sps

class SLIM_BPR_recommender(Recommender):
    
    def extract_eligible_users(self, URM):
        
        eligibleUsers = []
        
        for user_id in np.arange(URM.shape[0]):
            start_pos = URM.indptr[user_id]
            end_pos = URM.indptr[user_id+1]

            if len(URM.indices[start_pos:end_pos]) > 0:
                eligibleUsers.append(user_id)
        return eligibleUsers

    def sampleTriplet(self, eligibleUsers, URM):

        user_id = np.random.choice(eligibleUsers)

        # Get user seen items and choose one
        userSeenItems = URM[user_id].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, URM.shape[1])

            if (neg_item_id not in userSeenItems):

                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def updateFactors(self, user_id, positive_item_id, negative_item_id, URM, learning_rate):

            userSeenItems = URM[user_id].indices
            
            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

           
            for userSeenItem in userSeenItems:
                if(positive_item_id != userSeenItem):
                    update = gradient - self.lambda_i * self.similarity_matrix[positive_item_id, userSeenItems]
                    self.similarity_matrix[positive_item_id, userSeenItems] += learning_rate * update

                if(negative_item_id != userSeenItem):
                    update = gradient - self.lambda_j * self.similarity_matrix[negative_item_id, userSeenItems]
                    self.similarity_matrix[negative_item_id, userSeenItems] += learning_rate * update 

    def epochIteration(self, URM, learning_rate):

        # Get number of available interactions
        numPositiveIteractions = URM.nnz
        eligibleUsers = self.extract_eligible_users(URM)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet(eligibleUsers, URM)
            self.updateFactors(user_id, positive_item_id, negative_item_id, URM, learning_rate)

            if(time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0* float(num_sample)/numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()
                
                
    def fit(self, URM, learning_rate = 0.1, epochs = 10, lambda_i = 0.0025, lambda_j = 0.0025):
      #  self.cythonEpoch = SLIM_BPR_Cython_Epoch(URM,
      #                                           learning_rate=learning_rate,
      #                                           li_reg = lambda_i,
      #                                           lj_reg = lambda_j)

        self.similarity_matrix = np.zeros((URM.shape[1], URM.shape[1]))
        
        #regularization par
        self.lambda_i = lambda_i 
        self.lambda_j = lambda_j

        for numEpoch in range(epochs):
            self.epochIteration(URM, learning_rate) 
            print("Epoch {} of {} complete in {:.2f} minutes".format(numEpoch+1, epochs, float(time.time()-start_time_epoch)/60))

    def recommend_all(self, URM, test_set, at=10):
        
        submission = []
        S = self.similarity_matrix
        time_start = time_curr = time.time() 
       
        # Calculate matrix product: 
        # elements are sij where i is the item, j the user and
        # sij is the sum between the similarity score of item i with all the others.
        # the sum is done only on the coefficients of items with interactions.
        matrix_product = np.dot(S, URM.T)
        matrix_product_csc = sps.csc_matrix(matrix_product)
    
        # number of item interactions of the users
        # 50000 operations
        corated_items = [len(URM[user].indices) for user \
                in np.arange(URM.shape[0])]
        
        for user in test_set:
            
            #time
            if(time.time() - time_curr > 120):
                print("Time spent:\t {}sec Percentage of users seen:\t {:2.0f}%".format(\
                        (time_start - time.time()),(100 * (user / len(test_set)))))
                time_curr = time.time()
            
            # take a single column of product matrix. column_product has size item numbers.
            column_product = matrix_product_csc[:,user].toarray().squeeze()
            
            # all sij are < 1 and weighted by the occurency of the items
            column_norm = np.divide(column_product, corated_items[user])
            
            # put to zero all nan values; it happens only on testing
            column_norm_no_nan = np.where(np.isnan(column_norm), [0], column_norm) 
            
            # put to zero the items score already chosen by the user 
            mask_one_interaction = np.logical_not(URM[user].toarray().squeeze()) 
            column_ratings = np.multiply(column_norm_no_nan, mask_one_interaction)

            # sort by ratings and get the top ten items to recommend
            column_rank = np.argsort(column_ratings)[::-1][:10]
            submission.append([user, column_rank])
        
        return submission

    def recommend(self, playlist):

        submission = self.submission[playlist][1]
        return submission
