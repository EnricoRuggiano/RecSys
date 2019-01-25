import sys

import modules.importer as imp
import modules.data_splitter as splitter
import modules.evaluation_function as evaluation 
import recommender.hybrid_recommender as hybrid

def run(flag):
    ICM_matrix = imp.create_ICM()
    URM_matrix = imp.create_URM()
    target_playlists = imp.import_target_playlist()
    sequential_list = imp.import_sequential_playlist()

    (URM_train, URM_test) = splitter.train_test_holdout(URM_matrix, train_perc = 0.8)
    (sequential_list_train, sequential_list_test) = splitter.split_sequential(sequential_list, URM_test)
    
    recommender = hybrid.HybridRecommender(ICM_matrix, URM_matrix, URM_train, URM_test, \
            target_playlists, sequential_list, sequential_list_train, sequential_list_test)
    
    if(flag):
        recommender.test()
        evaluation.evaluate_algorithm(URM_test, recommender)
    else:
        recommender.execute()

if __name__ == '__main__':

    if(sys.argv[1] == '--test'):
        print ("Testing of recommender started.")
        flag_test = True
        run(flag_test)

    elif(sys.argv[1] == '--execute'):
        print ("Execution of recommendations started.")
        flag_test = False
        run (flag_test)
    else:
        print('Error: invalid arguments')
        print('To run this program:\npython3 main.py --test (to test it.)')
        print('python3 main.py --execute (to execute all the recommendations.)')
