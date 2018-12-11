from .recommender import Recommender
from .default.Matrix_Factorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython
from .default.Matrix_Factorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from .default.Matrix_Factorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from .default.Matrix_Factorization.Cython.MF_RMSE import FunkSVD_sgd
from .default.Base.Evaluation.Evaluator import SequentialEvaluator

import numpy as np
import scipy.sparse as sps

class MF_BPR(Recommender):
    
    def __init__(self, ICM, URM, URM_train, URM_test, target_playlists):
        
        Recommender.__init__(self,ICM, URM, URM_train, URM_test, target_playlists) 

    def fit(self, URM):
 
        evaluator_test = SequentialEvaluator(self.URM_test, cutoff_list=[10])
        evaluator_train = SequentialEvaluator(self.URM_train, cutoff_list=[10], exclude_seen=False)

   #     self.recommender = MatrixFactorization_BPR_Cython(URM, positive_threshold=1, URM_validation=URM, recompile_cython=False)
   #     self.recommender.fit(epochs=100, num_factors=10,
   #             learning_rate = 0.05, sgd_mode='adagrad', user_reg = 0.01, positive_reg = 0.01, negative_reg = 0.005,
   #             stop_on_validation = True, lower_validatons_allowed = 5, validation_metric = "MAP",
   #             evaluator_object = evaluator_train, validation_every_n = 10)
        self.U, self.V = FunkSVD_sgd(URM, num_factors=10, lrate=0.1e-1, reg=0.1e-3, n_iterations=100, init_mean=0.0, init_std=0.1, lrate_decay=1, rnd_seed=42)
        self.URM_setted = URM

    def recommend(self, user_id):
        
      #  score = self.recommender.compute_score_MF(user_id)
        score = np.dot(self.U[user_id], self.V.T)
        score[self.URM_setted[user_id].indices] = -np.inf
        
        submission = np.argsort(score)[::-1][:10]
        return submission

    def get_URM_train(self):
        return self.URM_train.copy()


