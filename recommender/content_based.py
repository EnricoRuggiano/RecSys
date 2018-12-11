from .recommender import Recommender
from modules.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np

class CBRecommender(Recommender):
   

    def fit(self, URM_matrix, shrink=100, topK=50, normalize=True,):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink, \
                topK=topK, normalize=normalize,\
                similarity = 'cosine')

        self.W_sparse = similarity_object.compute_similarity() 
    
    def filter_seen(self, user_id, URM_matrix, scores):
       
        start_pos = URM_matrix.indptr[user_id]
        end_pos = URM_matrix.indptr[user_id+1]
      
        user_profile = URM_matrix.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
      
        return scores
  
    def recommend_all(self, URM_matrix, test_set_size, at, exclude_seen=True):

        submission = []
                
        for playlist in test_set_size:
            
            user_profile = URM_matrix[playlist]
            scores = user_profile.dot(self.W_sparse).toarray().ravel()

            if exclude_seen:
                scores = self.filter_seen(playlist, URM_matrix, scores)
            ranking = scores.argsort()[::-1]
            ranking = list(ranking[:at])

            submission.append([playlist, ranking])
        return submission   

    def recommend(self, playlist):
        submission = self.submission[playlist][1]
        return submission
